"""Fill missing abstracts and arXiv links, remembering results across runs."""

import json
import re
import time
from pathlib import Path
from xml.etree import ElementTree

import requests
from loguru import logger

from .conference_programs import parse_abstract
from .openreview_state import project_root
from .protocol import Paper

_ATOM = {"atom": "http://www.w3.org/2005/Atom"}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
CACHE_PATH = Path("data") / "paper_cache.json"
LOOKUP_PAUSE_SECONDS = 3.001
MAX_LOOKUPS = 1000
MAX_RATE_LIMITS = 3


class ArxivRateLimit(Exception):
    """arXiv asked us to slow down, so this title was not actually looked up."""


def normalize_title(title: str) -> str:
    return _NON_ALNUM.sub("", title.lower())


def titles_match(left: str, right: str) -> bool:
    normalized = normalize_title(left)
    return bool(normalized) and normalized == normalize_title(right)


def _cache_file(root: Path | None = None) -> Path:
    return (root or project_root()) / CACHE_PATH


def load_cache(root: Path | None = None) -> dict:
    path = _cache_file(root)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_cache(cache: dict, root: Path | None = None) -> None:
    path = _cache_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2) + "\n")


def _retry_delay(response: requests.Response) -> int:
    raw_delay = response.headers.get("Retry-After") or "60"
    try:
        delay = int(float(raw_delay))
    except ValueError:
        delay = 60
    return min(max(delay, 1), 600)


def lookup_arxiv(title: str) -> tuple[str, str] | None:
    """Return ``(abs_url, abstract)`` only when the hit title matches exactly."""
    response = _arxiv_get(title)
    if response.status_code == 429:
        delay = _retry_delay(response)
        logger.warning(f"arXiv rate limit while matching a title. Waiting {delay}s.")
        time.sleep(delay)
        response = _arxiv_get(title)
        if response.status_code == 429:
            raise ArxivRateLimit(title)
    response.raise_for_status()
    root = ElementTree.fromstring(response.text)
    entry = root.find("atom:entry", _ATOM)
    if entry is None:
        return None
    found = "".join(entry.findtext("atom:title", default="", namespaces=_ATOM).split())
    if not titles_match(title, found):
        return None
    raw_id = entry.findtext("atom:id", default="", namespaces=_ATOM)
    paper_id = raw_id.rsplit("/abs/", 1)[-1]
    summary = " ".join(entry.findtext("atom:summary", default="", namespaces=_ATOM).split())
    return f"https://arxiv.org/abs/{paper_id}", summary


def _arxiv_get(title: str):
    return requests.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": f'ti:"{title}"', "start": 0, "max_results": 1},
        headers={"User-Agent": "zotero-arxiv-daily"},
        timeout=60,
    )


def _detail_abstract(url: str) -> str:
    if not url or "accepted_papers.php" in url or "dblp.org" in url:
        return ""
    response = requests.get(url, headers={"User-Agent": "zotero-arxiv-daily"}, timeout=60)
    if response.status_code >= 400:
        return ""
    return parse_abstract(response.text)


def _apply_cached(paper: Paper, cached: dict) -> None:
    if not paper.abstract and cached.get("abstract"):
        paper.abstract = cached["abstract"]
    if not paper.arxiv_url and cached.get("arxiv_url"):
        paper.arxiv_url = cached["arxiv_url"]


def _remember(cache: dict, key: str, abstract: str, arxiv_url: str, root: Path | None, arxiv_checked: bool) -> None:
    cache[key] = {"abstract": abstract, "arxiv_url": arxiv_url, "arxiv_checked": arxiv_checked}
    save_cache(cache, root)


def enrich_papers(
    papers: list[Paper],
    root: Path | None = None,
    pause_seconds: float = LOOKUP_PAUSE_SECONDS,
    max_lookups: int = MAX_LOOKUPS,
) -> None:
    total = len(papers)
    if total == 0:
        return
    logger.info(f"Enriching papers: 0% (0/{total})")
    cache = load_cache(root)
    last_percent = 0
    lookups = 0
    rate_limits = 0
    for index, paper in enumerate(papers, start=1):
        percent = index * 100 // total
        if percent != last_percent or index % 25 == 0:
            title = " ".join(paper.title.split())
            logger.info(f"Enriching papers: {percent}% ({index}/{total}) {title[:80]}")
            last_percent = percent
        key = normalize_title(paper.title)
        if not key:
            continue
        cached = cache.get(key)
        if cached is not None:
            _apply_cached(paper, cached)
            continue
        abstract = paper.abstract or _detail_abstract(paper.url)
        arxiv_url = paper.arxiv_url or ""
        if "arxiv.org/abs/" in (paper.url or "") and not arxiv_url:
            arxiv_url = paper.url
        checked = bool(arxiv_url)
        if not abstract and not arxiv_url:
            if lookups >= max_lookups:
                logger.info(f"Enriching paused after {max_lookups} arXiv lookups. Remaining papers continue next run.")
                break
            try:
                found = lookup_arxiv(paper.title)
            except ArxivRateLimit:
                rate_limits += 1
                lookups += 1
                if rate_limits >= MAX_RATE_LIMITS:
                    logger.warning("Stopping enrichment after repeated arXiv rate limits. Remaining papers continue next run.")
                    break
                continue
            except requests.RequestException:
                continue
            rate_limits = 0
            lookups += 1
            checked = True
            time.sleep(pause_seconds)
            if found:
                arxiv_url, arxiv_abstract = found
                abstract = abstract or arxiv_abstract
        _remember(cache, key, abstract, arxiv_url, root, checked)
        _apply_cached(paper, cache[key])


def attach_arxiv_links(
    papers: list[Paper],
    root: Path | None = None,
    pause_seconds: float = LOOKUP_PAUSE_SECONDS,
) -> None:
    """Fill arXiv links for a small mailed slice without blocking on a long backlog."""
    cache = load_cache(root)
    rate_limits = 0
    for paper in papers:
        if paper.arxiv_url or "arxiv.org/abs/" in (paper.url or ""):
            if not paper.arxiv_url:
                paper.arxiv_url = paper.url
            continue
        key = normalize_title(paper.title)
        cached = cache.get(key) if key else None
        if cached is not None and cached.get("arxiv_checked"):
            _apply_cached(paper, cached)
            continue
        try:
            found = lookup_arxiv(paper.title)
        except ArxivRateLimit:
            rate_limits += 1
            if rate_limits >= MAX_RATE_LIMITS:
                logger.warning("Stopped filling arXiv links after repeated rate limits.")
                break
            continue
        except requests.RequestException:
            continue
        rate_limits = 0
        time.sleep(pause_seconds)
        arxiv_url = ""
        abstract = paper.abstract or ""
        if found:
            arxiv_url, arxiv_abstract = found
            abstract = abstract or arxiv_abstract
            paper.arxiv_url = arxiv_url
            if not paper.abstract:
                paper.abstract = arxiv_abstract
        if key:
            _remember(cache, key, abstract, arxiv_url, root, True)

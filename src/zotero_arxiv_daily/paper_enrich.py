"""Fill missing abstracts and arXiv links, remembering results across runs."""

import json
import re
import time
from pathlib import Path
from xml.etree import ElementTree

import requests

from .conference_programs import parse_abstract
from .openreview_state import project_root
from .protocol import Paper

_ATOM = {"atom": "http://www.w3.org/2005/Atom"}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
CACHE_PATH = Path("data") / "paper_cache.json"


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


def lookup_arxiv(title: str) -> tuple[str, str] | None:
    """Return ``(abs_url, abstract)`` only when the hit title matches exactly."""
    response = _arxiv_get(title)
    if response.status_code == 429:
        delay = int(response.headers.get("Retry-After") or 60)
        time.sleep(min(delay, 600))
        response = _arxiv_get(title)
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


def enrich_papers(papers: list[Paper], root: Path | None = None, pause_seconds: float = 3.0) -> None:
    cache = load_cache(root)
    dirty = False
    for paper in papers:
        key = normalize_title(paper.title)
        if not key:
            continue
        cached = cache.get(key)
        if cached is None:
            abstract = paper.abstract or _detail_abstract(paper.url)
            arxiv_url = paper.arxiv_url or ""
            if "arxiv.org/abs/" in (paper.url or "") and not arxiv_url:
                arxiv_url = paper.url
            if not arxiv_url:
                try:
                    found = lookup_arxiv(paper.title)
                except requests.RequestException:
                    found = None
                time.sleep(pause_seconds)
                if found:
                    arxiv_url, arxiv_abstract = found
                    abstract = abstract or arxiv_abstract
            cached = {"abstract": abstract, "arxiv_url": arxiv_url}
            cache[key] = cached
            save_cache(cache, root)
            dirty = True
        if not paper.abstract and cached.get("abstract"):
            paper.abstract = cached["abstract"]
        if not paper.arxiv_url and cached.get("arxiv_url"):
            paper.arxiv_url = cached["arxiv_url"]
    if dirty:
        save_cache(cache, root)

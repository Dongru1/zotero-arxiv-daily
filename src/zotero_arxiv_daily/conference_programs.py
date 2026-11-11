"""Official accepted-paper lists that are not fully represented on OpenReview."""

import json
import re
import time
from dataclasses import dataclass
from typing import Callable

import requests
from loguru import logger

from .deadline import TimeBudgetExceeded, expired

_TAG = re.compile(r"<[^>]+>")
_ANTHOLOGY_PAPER = re.compile(
    r'href="(/(?P<id>20\d{2}\.[a-z0-9-]+\.\d+)/)"[^>]*>\s*(?P<title>[^<]+)',
    re.I,
)
_ABSTRACT = re.compile(
    r'class="[^"]*abstract[^"]*"[^>]*>(?P<body>.*?)</(?:div|span|p)>',
    re.I | re.S,
)
_ISCA_PAPER = re.compile(
    r'href="(?P<href>[^"]*interspeech[^"]*)"[^>]*>\s*(?P<title>.*?)</a>(?:\s*<br\s*/?>\s*(?P<authors>[^<]*))?',
    re.I | re.S,
)
_ICASSP_ROW = re.compile(
    r"<tr>\s*<td[^>]*>\s*(?P<number>\d+)\s*</td>\s*<td[^>]*>\s*(?P<title>.*?)\s*</td>",
    re.I | re.S,
)
_KEYNOTE = re.compile(r"keynote|invited talk|welcome|opening ceremony|chair:", re.I)


@dataclass
class ListedPaper:
    title: str
    authors: list[str]
    year: str
    venue: str
    url: str
    external_id: str
    abstract: str = ""


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", _TAG.sub(" ", text)).strip()


def parse_anthology_volume(html: str, venue: str, year: str, volume_id: str) -> list[ListedPaper]:
    papers = []
    seen = set()
    for match in _ANTHOLOGY_PAPER.finditer(html):
        paper_id = match.group("id")
        if paper_id in seen or paper_id == volume_id:
            continue
        title = _clean(match.group("title"))
        if not title or title.lower() in {"pdf", "bib", "abs"}:
            continue
        seen.add(paper_id)
        papers.append(ListedPaper(
            title=title,
            authors=[],
            year=year,
            venue=venue,
            url=f"https://aclanthology.org/{paper_id}/",
            external_id=paper_id,
        ))
    return papers


def parse_abstract(html: str) -> str:
    match = _ABSTRACT.search(html)
    if not match:
        return ""
    return _clean(match.group("body"))


def parse_isca_index(html: str, year: str) -> list[ListedPaper]:
    papers = []
    pieces = re.split(r"(<h[1-4][^>]*>.*?</h[1-4]>)", html, flags=re.I | re.S)
    skip = False
    for piece in pieces:
        if re.match(r"<h[1-4]", piece, re.I):
            skip = bool(_KEYNOTE.search(_clean(piece)))
            continue
        if skip:
            continue
        for match in _ISCA_PAPER.finditer(piece):
            title = _clean(match.group("title"))
            if not title or _KEYNOTE.search(title):
                continue
            href = match.group("href")
            if href.startswith("http"):
                url = href
            else:
                url = f"https://www.isca-archive.org/interspeech_{year}/{href.lstrip('/')}"
            authors = [item.strip() for item in _clean(match.group("authors")).split(",") if item.strip()]
            papers.append(ListedPaper(
                title=title,
                authors=authors,
                year=year,
                venue="Interspeech",
                url=url,
                external_id=href,
            ))
    return papers


def parse_icassp_accepted(html: str, year: str) -> list[ListedPaper]:
    papers = []
    for match in _ICASSP_ROW.finditer(html):
        title = _clean(match.group("title"))
        number = match.group("number")
        if not title:
            continue
        papers.append(ListedPaper(
            title=title,
            authors=[],
            year=year,
            venue="ICASSP",
            url=f"https://cmsworkshops.com/ICASSP{year}/papers/accepted_papers.php",
            external_id=f"icassp-{year}-{number}",
        ))
    return papers


def parse_dblp_hits(payload: dict, venue: str, year: str) -> list[ListedPaper]:
    hits = ((payload.get("result") or {}).get("hits") or {}).get("hit") or []
    if isinstance(hits, dict):
        hits = [hits]
    papers = []
    for hit in hits:
        info = hit.get("info") or {}
        title = _clean(info.get("title"))
        if not title:
            continue
        authors = info.get("authors", {}).get("author") or []
        if isinstance(authors, dict):
            authors = [authors]
        names = []
        for author in authors:
            if isinstance(author, dict):
                names.append(str(author.get("text") or author.get("@text") or "").strip())
            else:
                names.append(str(author).strip())
        ee = info.get("ee") or ""
        if isinstance(ee, list):
            ee = ee[0] if ee else ""
        if isinstance(ee, dict):
            ee = str(ee.get("text") or ee.get("@text") or "")
        key = str(info.get("key") or title)
        papers.append(ListedPaper(
            title=title.rstrip("."),
            authors=[name for name in names if name],
            year=str(info.get("year") or year),
            venue=venue,
            url=ee or f"https://dblp.org/db/conf/slt/slt{year}.html",
            external_id=key,
        ))
    return papers


def candidate_years(now_year: int, extra_back: int = 0) -> list[int]:
    years = [now_year + 1, now_year, now_year - 1]
    for offset in range(2, extra_back + 1):
        years.append(now_year - offset)
    return years


def biennial_years(now_year: int, editions: int = 2) -> list[int]:
    """Even-year venues such as SLT, newest edition first."""
    newest = now_year if now_year % 2 == 0 else now_year + 1
    return [newest - 2 * index for index in range(editions)]


def select_newest(years: list[int], available: Callable[[int], bool]) -> int | None:
    for year in years:
        if available(year):
            return year
    return None


def program_sources(year: int) -> list[tuple[str, str, str]]:
    """Return ``(venue, kind, url)`` for one concrete year."""
    return [
        ("EMNLP", "anthology", f"https://aclanthology.org/volumes/{year}.emnlp-main/"),
        ("EMNLP", "anthology", f"https://aclanthology.org/volumes/{year}.findings-emnlp/"),
        ("ACL", "anthology", f"https://aclanthology.org/volumes/{year}.acl-long/"),
        ("ACL", "anthology", f"https://aclanthology.org/volumes/{year}.findings-acl/"),
        ("ICASSP", "icassp", f"https://cmsworkshops.com/ICASSP{year}/papers/accepted_papers.php"),
        ("Interspeech", "isca", f"https://www.isca-archive.org/interspeech_{year}/index.html"),
        ("SLT", "dblp", f"https://dblp.org/search/publ/api?q=toc:db/conf/slt/slt{year}.bht:&h=1000&format=json"),
    ]


def _retry_delay(response: requests.Response) -> float:
    raw = response.headers.get("Retry-After") or "10"
    try:
        delay = float(raw)
    except ValueError:
        delay = 10
    return min(max(delay, 1), 120)


def _get(url: str) -> str:
    if expired():
        raise TimeBudgetExceeded(url)
    for attempt in range(3):
        response = requests.get(
            url,
            headers={"User-Agent": "zotero-arxiv-daily (conference list)"},
            timeout=60,
        )
        if response.status_code == 404:
            return ""
        if response.status_code == 429 and attempt < 2:
            delay = _retry_delay(response)
            logger.warning(f"Rate limited by {url}. Waiting {int(delay)}s.")
            time.sleep(delay)
            continue
        response.raise_for_status()
        return response.text
    return ""


def papers_for_venue_year(venue: str, kind: str, url: str, year: int) -> list[ListedPaper]:
    body = _get(url)
    if not body:
        return []
    if kind == "anthology":
        volume_id = url.rstrip("/").rsplit("/", 1)[-1]
        return parse_anthology_volume(body, venue, str(year), volume_id)
    if kind == "isca":
        return parse_isca_index(body, str(year))
    if kind == "icassp":
        return parse_icassp_accepted(body, str(year))
    if kind == "dblp":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return []
        return parse_dblp_hits(payload, venue, str(year))
    return []


def load_conference_programs(now_year: int, only_venue: str | None = None) -> list[ListedPaper]:
    """Use the newest year in the automatic window that actually lists papers."""
    found: list[ListedPaper] = []
    seen: set[tuple[str, str]] = set()
    templates = (
        ("EMNLP", "anthology", "https://aclanthology.org/volumes/{year}.emnlp-main/"),
        ("EMNLP", "anthology", "https://aclanthology.org/volumes/{year}.findings-emnlp/"),
        ("ACL", "anthology", "https://aclanthology.org/volumes/{year}.acl-long/"),
        ("ACL", "anthology", "https://aclanthology.org/volumes/{year}.findings-acl/"),
        ("ICASSP", "icassp", "https://cmsworkshops.com/ICASSP{year}/papers/accepted_papers.php"),
        ("Interspeech", "isca", "https://www.isca-archive.org/interspeech_{year}/index.html"),
        ("SLT", "dblp", "https://dblp.org/search/publ/api?q=toc:db/conf/slt/slt{year}.bht:&h=1000&format=json"),
    )
    for venue, kind, template in templates:
        if only_venue and venue.lower() != only_venue.lower():
            continue
        years = biennial_years(now_year) if venue == "SLT" else candidate_years(now_year)
        for year in years:
            if kind == "dblp":
                time.sleep(1)
            try:
                papers = papers_for_venue_year(venue, kind, template.format(year=year), year)
            except Exception as exc:
                logger.warning(f"Skipping {venue} {year} program list: {exc}")
                break
            if not papers:
                continue
            for paper in papers:
                key = (paper.venue, paper.external_id)
                if key in seen:
                    continue
                seen.add(key)
                found.append(paper)
            break
    return found

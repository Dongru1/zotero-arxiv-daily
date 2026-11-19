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
_ANTHOLOGY_ANCHOR = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a>", re.I | re.S)
_ANTHOLOGY_HREF = re.compile(r"""href=(?:"(?P<quoted>[^"]+)"|(?P<bare>[^\s>]+))""", re.I)
_ANTHOLOGY_ID = re.compile(r"(?:^|/)(?P<id>20\d{2}\.[a-z0-9-]+\.\d+)/?$", re.I)
_ANTHOLOGY_ABSTRACT = re.compile(
    r"""id=(?:"(?P<qid>abstract-[^"]+)"|(?P<bid>abstract-[^\s">]+))[^>]*>\s*(?:<div\b[^>]*>)?(?P<body>.*?)</div>""",
    re.I | re.S,
)
_ABSTRACT = re.compile(
    r'class="[^"]*abstract[^"]*"[^>]*>(?P<body>.*?)</(?:div|span|p)>',
    re.I | re.S,
)
_ISCA_ANCHOR = re.compile(
    r'<a\b[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<body>.*?)</a>(?:\s*<br\s*/?>\s*(?P<outside>[^<]*))?',
    re.I | re.S,
)
_ISCA_FILE = re.compile(r"(?P<file>[A-Za-z0-9]+\d{2}_interspeech\.html)$", re.I)
_ISCA_AUTHORS = re.compile(
    r'class="[^"]*w3-text-theme[^"]*"[^>]*>(?P<names>.*?)</span>',
    re.I | re.S,
)
_ICASSP_ROW = re.compile(
    r"<tr>\s*<td[^>]*>\s*(?P<number>\d+)\s*</td>\s*<td[^>]*>\s*(?P<title>.*?)\s*</td>",
    re.I | re.S,
)
_SLT_ROW = re.compile(
    r"<tr>\s*<td[^>]*>\s*#(?P<number>\d+)\s*</td>\s*<td[^>]*>\s*(?P<title>.*?)\s*</td>",
    re.I | re.S,
)
_KEYNOTE = re.compile(r"keynote|invited talk|welcome|opening ceremony|chair:", re.I)
_PROGRAM_TEMPLATES = (
    ("EMNLP", "anthology", "https://aclanthology.org/volumes/{year}.emnlp-main/"),
    ("EMNLP", "anthology", "https://aclanthology.org/volumes/{year}.findings-emnlp/"),
    ("ACL", "anthology", "https://aclanthology.org/volumes/{year}.acl-long/"),
    ("ACL", "anthology", "https://aclanthology.org/volumes/{year}.acl-short/"),
    ("ACL", "anthology", "https://aclanthology.org/volumes/{year}.findings-acl/"),
    ("ICASSP", "icassp", "https://cmsworkshops.com/ICASSP{year}/papers/accepted_papers.php"),
    ("Interspeech", "isca", "https://www.isca-archive.org/interspeech_{year}/index.html"),
    ("SLT", "slt", "https://attend.ieee.org/slt-{year}/program/"),
)


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
    # Drop tags without inserting spaces so Anthology case spans stay glued,
    # for example <span class=acl-fixed-case>BIP</span>ro -> BIPro.
    return re.sub(r"\s+", " ", _TAG.sub("", text)).strip()


def _anthology_href(attrs: str) -> str:
    match = _ANTHOLOGY_HREF.search(attrs)
    if not match:
        return ""
    return match.group("quoted") or match.group("bare") or ""


def _anthology_abstracts(html: str) -> dict[str, str]:
    found = {}
    for match in _ANTHOLOGY_ABSTRACT.finditer(html):
        key = match.group("qid") or match.group("bid") or ""
        abstract = _clean(match.group("body"))
        if key and abstract:
            found[key] = abstract
    return found


def parse_anthology_volume(html: str, venue: str, year: str, volume_id: str) -> list[ListedPaper]:
    papers = []
    seen = set()
    abstracts = _anthology_abstracts(html)
    for match in _ANTHOLOGY_ANCHOR.finditer(html):
        href = _anthology_href(match.group("attrs"))
        paper_match = _ANTHOLOGY_ID.search(href.split("?", 1)[0].split("#", 1)[0])
        if not paper_match:
            continue
        paper_id = paper_match.group("id")
        if paper_id in seen or paper_id == volume_id or paper_id.endswith(".0"):
            continue
        title = _clean(match.group("body"))
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
            abstract=abstracts.get("abstract-" + paper_id.replace(".", "--"), ""),
        ))
    return papers


def parse_abstract(html: str) -> str:
    match = _ABSTRACT.search(html)
    if not match:
        return ""
    return _clean(match.group("body"))


def _isca_title_and_authors(body: str, outside: str | None) -> tuple[str, list[str]]:
    parts = re.split(r"<br\s*/?>", body, maxsplit=1, flags=re.I)
    title = _clean(parts[0])
    author_html = parts[1] if len(parts) > 1 else ""
    author_match = _ISCA_AUTHORS.search(author_html)
    raw_authors = _clean(author_match.group("names") if author_match else "")
    if not raw_authors:
        raw_authors = _clean(outside or author_html)
    authors = [item.strip() for item in raw_authors.split(",") if item.strip()]
    return title, authors


def parse_isca_index(html: str, year: str) -> list[ListedPaper]:
    papers = []
    seen = set()
    pieces = re.split(r"(<h[1-4][^>]*>.*?</h[1-4]>)", html, flags=re.I | re.S)
    skip = False
    for piece in pieces:
        if re.match(r"<h[1-4]", piece, re.I):
            skip = bool(_KEYNOTE.search(_clean(piece)))
            continue
        if skip:
            continue
        for match in _ISCA_ANCHOR.finditer(piece):
            href = match.group("href")
            filename = href.rstrip("/").rsplit("/", 1)[-1]
            if filename in seen or not _ISCA_FILE.search(filename):
                continue
            title, authors = _isca_title_and_authors(match.group("body"), match.group("outside"))
            if not title or _KEYNOTE.search(title):
                continue
            seen.add(filename)
            if href.startswith("http"):
                url = href
            else:
                url = f"https://www.isca-archive.org/interspeech_{year}/{href.lstrip('/')}"
            papers.append(ListedPaper(
                title=title,
                authors=authors,
                year=year,
                venue="Interspeech",
                url=url,
                external_id=filename,
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


def parse_slt_program(html: str, year: str) -> list[ListedPaper]:
    papers = []
    seen = set()
    for match in _SLT_ROW.finditer(html):
        number = match.group("number")
        title = _clean(match.group("title"))
        if not title or number in seen or _KEYNOTE.search(title):
            continue
        seen.add(number)
        papers.append(ListedPaper(
            title=title,
            authors=[],
            year=year,
            venue="SLT",
            url=f"https://attend.ieee.org/slt-{year}/program/",
            external_id=f"slt-{year}-{number}",
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
        (venue, kind, template.format(year=year))
        for venue, kind, template in _PROGRAM_TEMPLATES
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
    if kind == "slt":
        return parse_slt_program(body, str(year))
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
    for venue, kind, template in _PROGRAM_TEMPLATES:
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

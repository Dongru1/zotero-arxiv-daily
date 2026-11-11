import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .openreview_state import PAGES_DIR, SITE_DIR, project_root
from .protocol import Paper

_DATED_PAGE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<kind>.+)\.html$")
_MONTHLY_PAGE = re.compile(r"^(?P<month>\d{4}-\d{2})-monthly\.html$")
_VENUE_PAGE = re.compile(r"^(?P<venue>.+)-(?P<year>\d{4})-(?P<kind>digest|under_review)\.html$")
DAILY_ARCHIVE = Path("data") / "daily_archive"

_PAGE_STYLE = """
body { margin: 0; background: #f4f5f7; color: #1f2430; font-family: Georgia, "Iowan Old Style", Palatino, serif; }
main { max-width: 760px; margin: 0 auto; padding: 40px 20px 72px; }
a { color: #16325c; }
h1 { font-size: 1.9rem; letter-spacing: -0.03em; margin-bottom: 0.3rem; }
.lead { color: #5c6570; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.card { background: #fff; border: 1px solid #e4e6eb; border-radius: 14px; padding: 20px 22px; margin: 16px 0; }
.card h2 { font-size: 1.15rem; line-height: 1.35; margin: 0 0 8px; }
.card h2 a { text-decoration: none; }
.card h2 a:hover { text-decoration: underline; }
.meta, .index, .actions { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.meta { color: #5c6570; font-size: 0.92rem; line-height: 1.5; }
.tldr, .abstract { margin: 12px 0; line-height: 1.55; }
.abstract { color: #333; }
.rating { display: inline-block; background: #e8f6ec; color: #17693a; border-radius: 999px; padding: 2px 8px; font-weight: 650; }
.actions { display: flex; gap: 8px; margin-top: 14px; }
.btn { display: inline-block; text-decoration: none; font-size: 0.88rem; font-weight: 650; padding: 8px 14px; border-radius: 999px; }
.btn.primary { background: #16325c; color: #fff; }
.btn.secondary { background: #fff; color: #16325c; border: 1px solid #c9d0da; }
.nav { display: flex; gap: 18px; margin: 8px 0 28px; }
.nav a { font-weight: 650; text-decoration: none; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin: 12px 0 8px; }
.cards.home { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
a.entry { display: block; background: #fff; border: 1px solid #e4e6eb; border-radius: 14px; padding: 18px 20px; color: inherit; text-decoration: none; }
a.entry strong { display: block; font-size: 1.15rem; }
a.entry span { display: block; margin-top: 6px; color: #5c6570; font-size: 0.88rem; }
details { margin: 8px 0 16px; }
summary { cursor: pointer; font-weight: 650; }
"""


@dataclass
class PageRecord:
    filename: str
    html: str
    kind: str
    run_date: str
    venue: str | None = None
    conference_year: str | None = None


def render_site_page(title: str, papers: list[Paper], intro: str | None = None) -> str:
    cards = [_paper_card(paper) for paper in papers] or ["<p class=\"lead\">No papers in this batch.</p>"]
    intro_html = f"<p class=\"lead\">{html.escape(intro)}</p>" if intro else ""
    return f"""<!DOCTYPE HTML>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{_PAGE_STYLE}</style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  {intro_html}
  {"".join(cards)}
  <p class="meta"><a href="index.html">All results</a></p>
</main>
</body>
</html>
"""


def _paper_card(paper: Paper) -> str:
    label, page_url = _source_link(paper)
    title = html.escape(paper.title)
    title_html = f'<a href="{html.escape(page_url)}">{title}</a>' if page_url else title
    authors = html.escape(", ".join(paper.authors))
    bits = [authors]
    if paper.score is not None:
        bits.append(f"Relevance {paper.score:.1f}")
    if paper.rating_summary:
        bits.append(f'<span class="rating">Rating {html.escape(paper.rating_summary)}</span>')
    meta = " · ".join(bits)
    abstract = f'<p class="abstract">{html.escape(paper.abstract)}</p>' if paper.abstract else ""
    tldr = f'<p class="tldr"><strong>TLDR:</strong> {html.escape(paper.tldr)}</p>' if paper.tldr else ""
    buttons = []
    if page_url and label:
        buttons.append(f'<a class="btn primary" href="{html.escape(page_url)}">{html.escape(label)}</a>')
    arxiv_url = paper.arxiv_url or ""
    if not arxiv_url and "arxiv.org/abs/" in (paper.url or ""):
        arxiv_url = paper.url
    if arxiv_url and arxiv_url != page_url:
        buttons.append(f'<a class="btn primary" href="{html.escape(arxiv_url)}">arXiv</a>')
    if paper.pdf_url:
        buttons.append(f'<a class="btn secondary" href="{html.escape(paper.pdf_url)}">PDF</a>')
    actions = f'<div class="actions">{"".join(buttons)}</div>' if buttons else ""
    return f"""<article class="card">
  <h2>{title_html}</h2>
  <p class="meta">{meta}</p>
  {abstract}
  {tldr}
  {actions}
</article>
"""


def _source_link(paper: Paper) -> tuple[str, str]:
    url = paper.url or ""
    if "openreview.net" in url:
        return "OpenReview", url
    if "arxiv.org" in url:
        return "arXiv", url
    return ("Paper", url) if url else ("", "")


def save_daily_archive(day: str, papers: list[Paper], root: Path | None = None) -> None:
    """Remember one day's mailed papers so the next month can rank them again."""
    path = (root or project_root()) / DAILY_ARCHIVE / f"{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "arxiv_url": paper.arxiv_url,
        }
        for paper in papers
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_month_archive(month: str, root: Path | None = None) -> list[Paper]:
    folder = (root or project_root()) / DAILY_ARCHIVE
    if not folder.is_dir():
        return []
    papers = []
    seen = set()
    for path in sorted(folder.glob(f"{month}-*.json")):
        for item in json.loads(path.read_text()):
            key = item.get("url") or item.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            papers.append(Paper(
                source="arxiv",
                title=item.get("title") or "",
                authors=list(item.get("authors") or []),
                abstract=item.get("abstract") or "",
                url=item.get("url") or "",
                pdf_url=item.get("pdf_url"),
                arxiv_url=item.get("arxiv_url"),
            ))
    return papers


def _infer_record(filename: str) -> dict:
    monthly = _MONTHLY_PAGE.match(filename)
    if monthly:
        return {
            "file": filename,
            "kind": "monthly",
            "run_date": monthly.group("month"),
            "venue": None,
            "conference_year": None,
        }
    venue_page = _VENUE_PAGE.match(filename)
    if venue_page:
        kind = venue_page.group("kind")
        return {
            "file": filename,
            "kind": "under_review" if kind == "under_review" else "digest",
            "run_date": "",
            "venue": venue_page.group("venue"),
            "conference_year": venue_page.group("year"),
        }
    match = _DATED_PAGE.match(filename)
    if not match:
        return {"file": filename, "kind": "daily", "run_date": "", "venue": None, "conference_year": None}
    run_date = match.group("date")
    kind_name = match.group("kind")
    if kind_name == "daily":
        return {
            "file": filename,
            "kind": "daily",
            "run_date": run_date,
            "venue": None,
            "conference_year": None,
        }
    return {
        "file": filename,
        "kind": "digest",
        "run_date": run_date,
        "venue": kind_name,
        "conference_year": run_date[:4],
    }


def _record_payload(page: PageRecord) -> dict:
    return {
        "file": page.filename,
        "kind": page.kind,
        "run_date": page.run_date,
        "venue": page.venue,
        "conference_year": page.conference_year or (page.run_date[:4] if page.kind == "digest" and page.run_date else None),
    }


def _load_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {item["file"]: item for item in payload if item.get("file")}


def _shell(title: str, body: str) -> str:
    return f"""<!DOCTYPE HTML>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{_PAGE_STYLE}</style>
</head>
<body>
<main class="index">
  {body}
</main>
</body>
</html>
"""


def _index_html(entries: list[dict]) -> str:
    counts = {
        "daily": sum(1 for item in entries if item.get("kind") == "daily"),
        "digest": sum(1 for item in entries if item.get("kind") == "digest"),
        "under_review": sum(1 for item in entries if item.get("kind") == "under_review"),
    }
    cards = [
        ("daily.html", "每日", f"{counts['daily']} 天"),
        ("digest.html", "会议精选", f"{counts['digest']} 个会议"),
        ("under-review.html", "Under Review", f"{counts['under_review']} 个会议"),
    ]
    body = '<div class="cards home">' + "".join(
        f'<a class="entry" href="{href}"><strong>{html.escape(title)}</strong><span>{html.escape(meta)}</span></a>'
        for href, title, meta in cards
    ) + "</div>"
    return _shell("Zotero arXiv Daily", f"<h1>Zotero arXiv Daily</h1>{body}")


def _section_page(title: str, body: str) -> str:
    return _shell(title, f"<h1>{html.escape(title)}</h1>{body}<p class=\"meta\"><a href=\"index.html\">首页</a></p>")


def _entry_card(item: dict, title: str) -> str:
    date = item.get("run_date") or ""
    meta = f"<span>{html.escape(date)}</span>" if date and date != title else ""
    return (
        f'<a class="entry" href="{html.escape(item["file"])}">'
        f"<strong>{html.escape(title)}</strong>{meta}</a>"
    )


def _daily_index(entries: list[dict], monthly: list[dict] | None = None) -> str:
    if not entries and not monthly:
        return "<p class=\"lead\">No daily results yet.</p>"
    by_month: dict[str, list[dict]] = {}
    for item in entries:
        month = (item.get("run_date") or "")[:7] or "Unknown"
        by_month.setdefault(month, []).append(item)
    monthly_by_month = {(item.get("run_date") or ""): item for item in monthly or []}
    months = sorted(set(by_month) | set(monthly_by_month), reverse=True)
    parts = []
    for index, month in enumerate(months):
        cards = []
        highlight = monthly_by_month.get(month)
        if highlight:
            cards.append(_entry_card({**highlight, "run_date": ""}, f"{month} 月度精选"))
        cards.extend(
            _entry_card(item, item.get("run_date") or "Daily")
            for item in sorted(by_month.get(month, []), key=lambda entry: entry.get("run_date") or "", reverse=True)
        )
        opened = " open" if index == 0 else ""
        parts.append(
            f"<details{opened}><summary>{html.escape(month)}</summary>"
            f'<div class="cards">{"".join(cards)}</div></details>'
        )
    return "".join(parts)


def _grouped_index(entries: list[dict], suffix: str) -> str:
    if not entries:
        return f"<p class=\"lead\">No {html.escape(suffix)} yet.</p>"
    grouped: dict[str, dict[str, list[dict]]] = {}
    for item in entries:
        year = item.get("conference_year") or "Unknown"
        venue = item.get("venue") or suffix
        grouped.setdefault(year, {}).setdefault(venue, []).append(item)
    years = sorted(grouped, reverse=True)
    parts = []
    for index, year in enumerate(years):
        cards = []
        for venue in sorted(grouped[year]):
            papers = sorted(grouped[year][venue], key=lambda entry: entry.get("run_date") or "", reverse=True)
            for item in papers:
                cards.append(_entry_card(item, venue))
        opened = " open" if index == 0 else ""
        parts.append(
            f"<details{opened}><summary>{html.escape(year)}</summary>"
            f'<div class="cards">{"".join(cards)}</div></details>'
        )
    return "".join(parts)


def publish_results(pages: list[PageRecord], root: Path | None = None) -> Path:
    """Archive this run's site pages and rebuild the GitHub Pages index.

    Older HTML files stay in ``data/pages``. Digests are grouped by conference
    year, then venue. A digest without a conference year uses the run year.
    """
    root = root or project_root()
    archive = root / PAGES_DIR
    archive.mkdir(parents=True, exist_ok=True)
    manifest_path = archive / "manifest.json"
    by_file = _load_manifest(manifest_path)
    for page in pages:
        (archive / page.filename).write_text(page.html)
        by_file[page.filename] = _record_payload(page)
    for archived_page in archive.glob("*.html"):
        inferred = _infer_record(archived_page.name)
        current = by_file.get(archived_page.name)
        if current is None or (
            inferred["kind"] in {"digest", "under_review"} and current.get("kind") == "daily"
        ):
            by_file[archived_page.name] = inferred
    manifest_path.write_text(json.dumps(list(by_file.values()), indent=2) + "\n")

    site = root / SITE_DIR
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    for item in by_file.values():
        source = archive / item["file"]
        if source.is_file():
            shutil.copyfile(source, site / item["file"])
    values = list(by_file.values())
    (site / "index.html").write_text(_index_html(values))
    daily = [item for item in values if item.get("kind") == "daily"]
    monthly = [item for item in values if item.get("kind") == "monthly"]
    digest = [item for item in values if item.get("kind") == "digest"]
    reviewing = [item for item in values if item.get("kind") == "under_review"]
    (site / "daily.html").write_text(_section_page("每日", _daily_index(daily, monthly)))
    (site / "digest.html").write_text(_section_page("会议精选", _grouped_index(digest, "精选")))
    (site / "under-review.html").write_text(_section_page("Under Review", _grouped_index(reviewing, "Under Review")))
    return site

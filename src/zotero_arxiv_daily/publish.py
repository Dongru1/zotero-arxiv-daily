import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .openreview_state import PAGES_DIR, SITE_DIR, project_root
from .protocol import Paper

_FILENAME = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<kind>.+)\.html$")

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
.year { margin: 22px 0 4px; }
.venue { margin: 10px 0 4px; }
.index ul { margin-top: 6px; }
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


def _infer_record(filename: str) -> dict:
    match = _FILENAME.match(filename)
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


def _index_html(entries: list[dict]) -> str:
    sections = [
        ("精选", "digest", "精选"),
        ("Under Review", "under_review", "Under Review"),
        ("每日", "daily", "Daily"),
    ]
    body = []
    for heading, kind, suffix in sections:
        items = [item for item in entries if item.get("kind") == kind or (kind == "daily" and item.get("kind") not in {"digest", "under_review"})]
        if kind != "daily":
            items = [item for item in entries if item.get("kind") == kind]
        html_block = _grouped_index(items, suffix) if kind != "daily" else _daily_index(items)
        body.append(f"<h2>{heading}</h2>{html_block}")
    return f"""<!DOCTYPE HTML>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zotero arXiv Daily</title>
  <style>{_PAGE_STYLE}</style>
</head>
<body>
<main class="index">
  <h1>Zotero arXiv Daily</h1>
  {''.join(body)}
</main>
</body>
</html>
"""


def _daily_index(entries: list[dict]) -> str:
    daily_items = []
    for item in sorted(entries, key=lambda entry: entry.get("run_date") or "", reverse=True):
        label = item.get("run_date") or item["file"]
        daily_items.append(f'<li><a href="{html.escape(item["file"])}">{html.escape(label)} Daily</a></li>')
    return "<ul>" + "".join(daily_items) + "</ul>" if daily_items else "<p class=\"lead\">No daily results yet.</p>"


def _grouped_index(entries: list[dict], suffix: str) -> str:
    if not entries:
        return f"<p class=\"lead\">No {html.escape(suffix)} yet.</p>"
    grouped: dict[str, dict[str, list[dict]]] = {}
    for item in entries:
        year = item.get("conference_year") or "Unknown"
        venue = item.get("venue") or suffix
        grouped.setdefault(year, {}).setdefault(venue, []).append(item)
    parts = []
    for year in sorted(grouped, reverse=True):
        parts.append(f'<h3 class="year">{html.escape(year)}</h3>')
        for venue in sorted(grouped[year]):
            parts.append(f'<h4 class="venue">{html.escape(venue)}</h4><ul>')
            papers = sorted(grouped[year][venue], key=lambda entry: entry.get("run_date") or "", reverse=True)
            for item in papers:
                label = f'{item.get("run_date") or item["file"]} {suffix}'
                parts.append(f'<li><a href="{html.escape(item["file"])}">{html.escape(label)}</a></li>')
            parts.append("</ul>")
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
        by_file.setdefault(archived_page.name, _infer_record(archived_page.name))
    manifest_path.write_text(json.dumps(list(by_file.values()), indent=2) + "\n")

    site = root / SITE_DIR
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    for item in by_file.values():
        source = archive / item["file"]
        if source.is_file():
            shutil.copyfile(source, site / item["file"])
    (site / "index.html").write_text(_index_html(list(by_file.values())))
    return site

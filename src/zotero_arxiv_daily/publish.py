import shutil
from pathlib import Path

from .openreview_state import PAGES_DIR, SITE_DIR, project_root


def _page_sort_key(path: Path) -> tuple[str, str]:
    stem = path.stem
    date = stem[:10] if len(stem) >= 10 and stem[4] == "-" and stem[7] == "-" else ""
    return date, stem


def _link_label(path: Path) -> str:
    stem = path.stem
    date = stem[:10]
    kind = stem[11:] if len(stem) > 11 else stem
    if kind == "daily":
        return f"{date} Daily"
    return f"{date} {kind} 精选"


def publish_results(
    pages: list[tuple[str, str]],
    root: Path | None = None,
) -> Path:
    """Archive this run's HTML and rebuild the GitHub Pages site.

    ``pages`` is a list of ``(filename, html)`` pairs. Filenames look like
    ``2026-11-09-daily.html`` or ``2026-11-09-ICLR.html``. Older archived
    pages stay in ``data/pages`` so the next deploy still lists them.
    """
    root = root or project_root()
    archive = root / PAGES_DIR
    archive.mkdir(parents=True, exist_ok=True)
    for name, html in pages:
        (archive / name).write_text(html)

    site = root / SITE_DIR
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    archived = sorted(archive.glob("*.html"), key=_page_sort_key, reverse=True)
    items = []
    for page in archived:
        target = site / page.name
        shutil.copyfile(page, target)
        items.append(f'<li><a href="{page.name}">{_link_label(page)}</a></li>')
    if not items:
        items.append("<li>No results yet.</li>")
    index = """<!DOCTYPE HTML>
<html>
<head>
  <meta charset="utf-8">
  <title>Zotero arXiv Daily</title>
</head>
<body style="font-family: Arial, sans-serif; color: #333; max-width: 720px; margin: 32px auto;">
  <h1>Zotero arXiv Daily</h1>
  <ul>
    __ITEMS__
  </ul>
</body>
</html>
"""
    (site / "index.html").write_text(index.replace("__ITEMS__", "\n    ".join(items)))
    return site

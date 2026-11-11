"""Tests for OpenReview state and the GitHub Pages archive."""

import json

from zotero_arxiv_daily.openreview_state import OpenReviewState
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.publish import PageRecord, publish_results, render_site_page


def test_openreview_state_roundtrip(tmp_path):
    state = OpenReviewState(seen_ids={"b", "a"}, bootstrapped={"ICLR"})
    state.save(tmp_path)
    loaded = OpenReviewState.load(tmp_path)
    assert loaded.seen_ids == {"a", "b"}
    assert loaded.bootstrapped == {"ICLR"}
    payload = json.loads((tmp_path / "data" / "openreview_state.json").read_text())
    assert payload["seen_ids"] == ["a", "b"]


def test_publish_results_keeps_older_pages(tmp_path):
    publish_results([
        PageRecord("2026-11-01-daily.html", "<p>first</p>", "daily", "2026-11-01"),
    ], tmp_path)
    publish_results([
        PageRecord("2026-11-09-ICLR.html", "<p>iclr</p>", "digest", "2026-11-09", "ICLR", "2026"),
    ], tmp_path)
    site = tmp_path / "site"
    assert (site / "2026-11-01-daily.html").read_text() == "<p>first</p>"
    assert (site / "2026-11-09-ICLR.html").read_text() == "<p>iclr</p>"
    index = (site / "index.html").read_text()
    assert index.index('href="daily.html"') < index.index('href="digest.html"') < index.index('href="under-review.html"')
    assert ">会议精选</strong>" in index
    digest = (site / "digest.html").read_text()
    daily = (site / "daily.html").read_text()
    assert ">ICLR</strong>" in digest
    assert ">2026-11-01</strong>" in daily


def test_publish_groups_digests_by_conference_year(tmp_path):
    publish_results([
        PageRecord("2026-11-09-ICLR.html", "<p>new</p>", "digest", "2026-11-09", "ICLR", "2026"),
        PageRecord("2026-01-02-ICLR.html", "<p>old</p>", "digest", "2026-01-02", "ICLR", "2025"),
        PageRecord("2026-11-09-daily.html", "<p>daily</p>", "daily", "2026-11-09"),
    ], tmp_path)
    digest = (tmp_path / "site" / "digest.html").read_text()
    assert '<details open><summary>2026</summary>' in digest
    assert "<details><summary>2025</summary>" in digest
    assert digest.index(">2026<") < digest.index(">2025<")
    assert ">ICLR</strong>" in digest


def test_publish_infers_year_for_old_digest_files(tmp_path):
    archive = tmp_path / "data" / "pages"
    archive.mkdir(parents=True)
    (archive / "2025-05-01-ACL.html").write_text("<p>old acl</p>")
    publish_results([], tmp_path)
    digest = (tmp_path / "site" / "digest.html").read_text()
    assert "<details open><summary>2025</summary>" in digest
    assert ">ACL</strong>" in digest
    assert "2025-05-01" in digest


def test_publish_puts_accepted_conference_pages_under_digest(tmp_path):
    archive = tmp_path / "data" / "pages"
    archive.mkdir(parents=True)
    (archive / "ICML-2026-digest.html").write_text("<p>accepted</p>")
    (archive / "ICML-2025-under_review.html").write_text("<p>review</p>")
    (archive / "2026-11-11-daily.html").write_text("<p>daily</p>")
    (archive / "manifest.json").write_text(json.dumps([{
        "file": "ICML-2026-digest.html",
        "kind": "daily",
        "run_date": "",
        "venue": None,
        "conference_year": None,
    }]))
    publish_results([], tmp_path)
    index = (tmp_path / "site" / "index.html").read_text()
    digest = (tmp_path / "site" / "digest.html").read_text()
    review = (tmp_path / "site" / "under-review.html").read_text()
    daily = (tmp_path / "site" / "daily.html").read_text()
    assert "ICML-2026-digest.html" not in index
    assert ">每日</strong>" in index
    assert ">会议精选</strong>" in index
    assert ">Under Review</strong>" in index
    assert ">ICML</strong>" in digest
    assert "<details open><summary>2026</summary>" in digest
    assert "<details open><summary>2025</summary>" in review
    assert ">2026-11-11</strong>" in daily


def test_daily_page_collapses_older_months(tmp_path):
    archive = tmp_path / "data" / "pages"
    archive.mkdir(parents=True)
    (archive / "2026-11-11-daily.html").write_text("<p>new</p>")
    (archive / "2026-10-02-daily.html").write_text("<p>old</p>")
    (archive / "2026-10-monthly.html").write_text("<p>month</p>")
    publish_results([], tmp_path)
    daily = (tmp_path / "site" / "daily.html").read_text()
    assert '<details open><summary>2026-11</summary>' in daily
    assert "<details><summary>2026-10</summary>" in daily
    assert "2026-10 月度精选" in daily


def test_month_archive_dedupes_titles(tmp_path):
    from zotero_arxiv_daily.publish import load_month_archive, save_daily_archive

    first = Paper(source="arxiv", title="Same", authors=["A"], abstract="one", url="https://arxiv.org/abs/1")
    second = Paper(source="arxiv", title="Same", authors=["A"], abstract="one", url="https://arxiv.org/abs/1")
    other = Paper(source="arxiv", title="Other", authors=["B"], abstract="two", url="https://arxiv.org/abs/2")
    save_daily_archive("2026-10-01", [first], tmp_path)
    save_daily_archive("2026-10-02", [second, other], tmp_path)
    loaded = load_month_archive("2026-10", tmp_path)
    assert [paper.title for paper in loaded] == ["Same", "Other"]


def test_site_page_shows_source_links_and_rating():
    html = render_site_page("ICLR 精选", [
        Paper(
            source="openreview",
            title="A <paper>",
            authors=["Ann"],
            abstract="abs",
            url="https://openreview.net/forum?id=abc",
            pdf_url="https://openreview.net/pdf?id=abc",
            tldr="idea",
            score=7.2,
            rating_summary="6.8 / 4 reviews",
        ),
        Paper(
            source="arxiv",
            title="Arxiv paper",
            authors=["Bob"],
            abstract="A short abstract.",
            url="https://arxiv.org/abs/2607.14846v1",
            pdf_url="https://arxiv.org/pdf/2607.14846v1",
        ),
    ])
    assert html.count('href="https://openreview.net/forum?id=abc"') == 2
    assert ">OpenReview</a>" in html
    assert "A short abstract." in html
    assert 'href="https://arxiv.org/abs/2607.14846v1"' in html
    assert ">arXiv</a>" in html
    assert "Rating 6.8 / 4 reviews" in html
    assert "A &lt;paper&gt;" in html

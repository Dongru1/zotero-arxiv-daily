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
    assert index.index("精选") < index.index("每日")
    assert "2026-11-09 精选" in index
    assert "2026-11-01 Daily" in index


def test_publish_groups_digests_by_conference_year(tmp_path):
    publish_results([
        PageRecord("2026-11-09-ICLR.html", "<p>new</p>", "digest", "2026-11-09", "ICLR", "2026"),
        PageRecord("2026-01-02-ICLR.html", "<p>old</p>", "digest", "2026-01-02", "ICLR", "2025"),
        PageRecord("2026-11-09-daily.html", "<p>daily</p>", "daily", "2026-11-09"),
    ], tmp_path)
    index = (tmp_path / "site" / "index.html").read_text()
    assert index.index(">2026<") < index.index(">2025<")
    assert index.index("2026-11-09 精选") < index.index(">2025<")


def test_publish_infers_year_for_old_digest_files(tmp_path):
    archive = tmp_path / "data" / "pages"
    archive.mkdir(parents=True)
    (archive / "2025-05-01-ACL.html").write_text("<p>old acl</p>")
    publish_results([], tmp_path)
    index = (tmp_path / "site" / "index.html").read_text()
    assert ">2025<" in index
    assert "ACL" in index
    assert "2025-05-01 精选" in index


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

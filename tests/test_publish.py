"""Tests for OpenReview state and the GitHub Pages archive."""

import json

from zotero_arxiv_daily.openreview_state import OpenReviewState
from zotero_arxiv_daily.publish import publish_results


def test_openreview_state_roundtrip(tmp_path):
    state = OpenReviewState(seen_ids={"b", "a"}, bootstrapped={"ICLR"})
    state.save(tmp_path)
    loaded = OpenReviewState.load(tmp_path)
    assert loaded.seen_ids == {"a", "b"}
    assert loaded.bootstrapped == {"ICLR"}
    payload = json.loads((tmp_path / "data" / "openreview_state.json").read_text())
    assert payload["seen_ids"] == ["a", "b"]


def test_publish_results_keeps_older_pages(tmp_path):
    publish_results([("2026-11-01-daily.html", "<p>first</p>")], tmp_path)
    publish_results([("2026-11-09-ICLR.html", "<p>iclr</p>")], tmp_path)
    site = tmp_path / "site"
    assert (site / "2026-11-01-daily.html").read_text() == "<p>first</p>"
    assert (site / "2026-11-09-ICLR.html").read_text() == "<p>iclr</p>"
    index = (site / "index.html").read_text()
    assert index.index("2026-11-09 ICLR 精选") < index.index("2026-11-01 Daily")

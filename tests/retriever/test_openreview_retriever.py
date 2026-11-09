"""Tests for OpenReviewRetriever."""

from datetime import datetime, timezone

import pytest
import requests
from omegaconf import open_dict

from zotero_arxiv_daily.retriever.base import get_retriever_cls
from zotero_arxiv_daily.retriever.openreview_retriever import OpenReviewRetriever


FIXED_NOW = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)
RECENT_MS = int(datetime(2026, 11, 9, 6, 0, tzinfo=timezone.utc).timestamp() * 1000)
OLD_MS = int(datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
OLDER_MS = int(datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def _group(group_id, submission_id=None):
    content = {}
    if submission_id:
        content = {
            "submission_id": {"value": submission_id},
            "submission_venue_id": {"value": f"{group_id}/Submission"},
            "withdrawn_venue_id": {"value": f"{group_id}/Withdrawn_Submission"},
            "desk_rejected_venue_id": {"value": f"{group_id}/Desk_Rejected_Submission"},
            "rejected_venue_id": {"value": f"{group_id}/Rejected_Submission"},
        }
    return {"groups": [{"id": group_id, "content": content}], "count": 1}


def _note(note_id, title, abstract, tmdate, venue_id, authors=None):
    return {
        "id": note_id,
        "tmdate": tmdate,
        "content": {
            "title": {"value": title},
            "abstract": {"value": abstract},
            "authors": {"value": authors if authors is not None else ["Ann Author"]},
            "venueid": {"value": venue_id},
        },
    }


@pytest.fixture()
def fixed_now(monkeypatch):
    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.openreview_retriever.datetime",
        _FixedDatetime,
    )


@pytest.fixture(autouse=True)
def no_live_program_lists(monkeypatch):
    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.openreview_retriever.load_conference_programs",
        lambda year: [],
    )
    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.openreview_retriever.datetime",
        _FixedDatetime,
    )


def _configure(config, venues=("iclr",), lookback_hours=48):
    with open_dict(config.source):
        config.source.openreview = {
            "username": "reviewer@example.com",
            "password": "secret",
            "venues": list(venues),
            "lookback_hours": lookback_hours,
        }
    return OpenReviewRetriever(config)


def _route(monkeypatch, groups, notes_by_invitation, prefixes=None, forbidden_invitations=()):
    calls = []

    def _request(method, url, **kwargs):
        params = kwargs.get("params") or {}
        body = kwargs.get("json") or {}
        calls.append((method, url, params, body))
        if url.endswith("/login"):
            assert body["id"] == "reviewer@example.com"
            assert body["password"] == "secret"
            assert "Authorization" not in kwargs["headers"]
            return _Response(200, {"token": "token-1"})
        assert kwargs["headers"]["Authorization"] == "Bearer token-1"
        if url.endswith("/groups") and "id" in params:
            group = groups.get(params["id"])
            if group is None:
                return _Response(404, {})
            return _Response(200, group)
        if url.endswith("/groups") and "prefix" in params:
            return _Response(200, {"groups": (prefixes or {}).get(params["prefix"], []), "count": 0})
        if url.endswith("/notes"):
            invitation = params["invitation"]
            if invitation in forbidden_invitations:
                return _Response(403, {})
            assert params["sort"] == "tmdate:desc"
            pages = notes_by_invitation.get(invitation, [[]])
            page_index = int(params["offset"]) // int(params["limit"])
            page = pages[page_index] if page_index < len(pages) else []
            return _Response(200, {"notes": page, "count": sum(len(p) for p in pages)})
        raise AssertionError(f"unexpected request {method} {url} {params}")

    monkeypatch.setattr(requests, "request", _request)
    return calls


def test_openreview_is_registered():
    assert get_retriever_cls("openreview") is OpenReviewRetriever


def test_openreview_requires_credentials(config):
    with open_dict(config.source):
        config.source.openreview = {"username": None, "password": None, "venues": ["iclr"]}
    with pytest.raises(ValueError, match="OPENREVIEW_USERNAME"):
        OpenReviewRetriever(config)


def test_openreview_requires_venues(config):
    with open_dict(config.source):
        config.source.openreview = {"username": "a@b.c", "password": "x", "venues": None}
    with pytest.raises(ValueError, match="venues must be specified"):
        OpenReviewRetriever(config)


def test_openreview_rejects_unknown_venue(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.openreview_retriever.sleep", lambda _: None)
    _route(monkeypatch, {}, {})
    retriever = _configure(config, venues=("not-a-conference",))
    with pytest.raises(ValueError, match="Unknown OpenReview venue"):
        retriever.retrieve_papers()


def test_openreview_keeps_recent_public_notes(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    venue_id = "ICLR.cc/2026/Conference/Submission"
    notes = [[
        _note("recent", "A recent paper", "This is the abstract.", RECENT_MS, venue_id),
        _note("withdrawn", "Withdrawn paper", "Hidden.", RECENT_MS, "ICLR.cc/2026/Conference/Withdrawn_Submission"),
        _note("blind", "Blind paper", "", RECENT_MS, venue_id),
        _note("old", "Old paper", "Too old.", OLD_MS, venue_id),
    ]]
    calls = _route(
        monkeypatch,
        {"ICLR.cc/2026/Conference": _group("ICLR.cc/2026/Conference", "ICLR.cc/2026/Conference/-/Submission")},
        {"ICLR.cc/2026/Conference/-/Submission": notes},
    )
    retriever = _configure(config)
    papers = retriever.retrieve_papers()

    assert [paper.title for paper in papers] == ["A recent paper", "Old paper"]
    assert {paper.status for paper in papers} == {"under_review"}
    paper = papers[0]
    assert paper.source == "openreview"
    assert paper.authors == ["Ann Author"]
    assert paper.abstract == "This is the abstract."
    assert paper.url == "https://openreview.net/forum?id=recent"
    assert paper.pdf_url == "https://openreview.net/pdf?id=recent"
    assert paper.full_text is None
    assert paper.venue == "ICLR"
    assert paper.external_id == "recent"
    assert paper.conference_year == "2026"
    note_calls = [params for method, url, params, _ in calls if url.endswith("/notes")]
    assert len(note_calls) == 1
    assert note_calls[0]["invitation"] == "ICLR.cc/2026/Conference/-/Submission"


def test_openreview_paginates_until_cutoff(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.openreview_retriever.sleep", lambda _: None)
    venue_id = "ICLR.cc/2026/Conference/Submission"
    pages = [
        [
            _note("p0", "Page zero", "Abstract zero.", RECENT_MS, venue_id),
            _note("p1", "Page one", "Abstract one.", RECENT_MS, venue_id),
        ],
        [_note("old", "Old paper", "Too old.", OLD_MS, venue_id)],
    ]
    calls = _route(
        monkeypatch,
        {"ICLR.cc/2026/Conference": _group("ICLR.cc/2026/Conference", "ICLR.cc/2026/Conference/-/Submission")},
        {"ICLR.cc/2026/Conference/-/Submission": pages},
    )
    retriever = _configure(config)
    retriever.page_size = 2
    papers = retriever.retrieve_papers()
    assert [paper.title for paper in papers] == ["Page zero", "Page one", "Old paper"]
    offsets = [params["offset"] for _, url, params, _ in calls if url.endswith("/notes")]
    assert offsets == [0, 2]


def test_openreview_expands_slt_tracks_and_skips_workshops(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.openreview_retriever.sleep", lambda _: None)
    parent = "IEEE.org/SLT/2026"
    challenge = f"{parent}/Challenge_Track"
    workshop = f"{parent}/Workshop"
    venue_id = f"{challenge}/Submission"
    prefixes = {
        parent: [
            {"id": parent},
            {"id": challenge},
            {"id": workshop},
            {"id": f"{challenge}/Publication_Chairs"},
        ]
    }
    _route(
        monkeypatch,
        {
            challenge: _group(challenge, f"{challenge}/-/Submission"),
            workshop: _group(workshop, f"{workshop}/-/Submission"),
            parent: _group(parent),
        },
        {f"{challenge}/-/Submission": [[_note("slt1", "SLT challenge", "Speech abstract.", RECENT_MS, venue_id)]]},
        prefixes=prefixes,
    )
    retriever = _configure(config, venues=("slt",))
    papers = retriever.retrieve_papers()
    assert [paper.title for paper in papers] == ["SLT challenge"]
    assert papers[0].venue == "SLT"


def test_openreview_missing_conference_returns_no_papers(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.openreview_retriever.sleep", lambda _: None)
    calls = _route(monkeypatch, {}, {})
    retriever = _configure(config, venues=("icassp", "interspeech"))
    assert retriever.retrieve_papers() == []
    assert not any(url.endswith("/notes") for _, url, _, _ in calls)


def test_openreview_skips_forbidden_venue(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.openreview_retriever.sleep", lambda _: None)
    iclr = "ICLR.cc/2026/Conference"
    icml = "ICML.cc/2026/Conference"
    _route(
        monkeypatch,
        {
            iclr: _group(iclr, f"{iclr}/-/Submission"),
            icml: _group(icml, f"{icml}/-/Submission"),
        },
        {f"{iclr}/-/Submission": [[_note("ok", "Visible paper", "Abstract.", RECENT_MS, f"{iclr}/Submission")]]},
        forbidden_invitations={f"{icml}/-/Submission"},
    )
    retriever = _configure(config, venues=("iclr", "icml"))
    papers = retriever.retrieve_papers()
    assert [paper.title for paper in papers] == ["Visible paper"]


def test_openreview_debug_keeps_the_release_batch(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.openreview_retriever.sleep", lambda _: None)
    venue_id = "ICLR.cc/2026/Conference/Submission"
    notes = [[
        _note(f"n{i}", f"Paper {i}", f"Abstract {i}.", RECENT_MS, venue_id)
        for i in range(12)
    ]]
    _route(
        monkeypatch,
        {"ICLR.cc/2026/Conference": _group("ICLR.cc/2026/Conference", "ICLR.cc/2026/Conference/-/Submission")},
        {"ICLR.cc/2026/Conference/-/Submission": notes},
    )
    retriever = _configure(config)
    retriever.config.executor.debug = True
    papers = retriever.retrieve_papers()
    assert len(papers) == 12
    assert {paper.venue for paper in papers} == {"ICLR"}


def test_openreview_convert_anonymous_authors(config):
    retriever = _configure(config)
    paper = retriever.convert_to_paper(
        {
            "id": "anon",
            "content": {
                "title": {"value": "  Anonymous   Title "},
                "abstract": {"value": "An abstract."},
                "authors": {"value": []},
            },
        }
    )
    assert paper.title == "Anonymous Title"
    assert paper.authors == ["Anonymous"]


def _iclr_route(monkeypatch, notes):
    return _route(
        monkeypatch,
        {"ICLR.cc/2026/Conference": _group("ICLR.cc/2026/Conference", "ICLR.cc/2026/Conference/-/Submission")},
        {"ICLR.cc/2026/Conference/-/Submission": [notes]},
    )


def test_openreview_skips_seen_ids(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.openreview_retriever.sleep", lambda _: None)
    venue_id = "ICLR.cc/2026/Conference/Submission"
    _iclr_route(monkeypatch, [
        _note("seen", "Already handled", "Old abstract.", RECENT_MS, venue_id),
        _note("fresh", "Fresh paper", "New abstract.", RECENT_MS, venue_id),
    ])
    retriever = _configure(config)
    retriever.state.seen_ids.add("seen")
    papers = retriever.retrieve_papers()
    assert [paper.title for paper in papers] == ["Fresh paper"]
    assert retriever.newly_bootstrapped == set()


def test_openreview_bootstraps_only_the_latest_public_day(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.openreview_retriever.sleep", lambda _: None)
    venue_id = "ICLR.cc/2026/Conference/Submission"
    calls = _iclr_route(monkeypatch, [
        _note("last", "Last public paper", "From the last public day.", OLD_MS, venue_id),
        _note("last-2", "Another last public paper", "Same day.", OLD_MS, venue_id),
        _note("ancient", "Ancient paper", "From an older day.", OLDER_MS, venue_id),
    ])
    retriever = _configure(config)
    papers = retriever.retrieve_papers()
    assert [paper.title for paper in papers] == [
        "Last public paper",
        "Another last public paper",
        "Ancient paper",
    ]
    assert {paper.status for paper in papers} == {"under_review"}


def test_attach_ratings_averages_public_reviews(config):
    from zotero_arxiv_daily.protocol import Paper

    retriever = _configure(config)
    retriever._token = "token"

    def fake_request(method, path, **kwargs):
        assert kwargs["params"]["forum"] == "note1"
        return {"notes": [
            {"id": "note1", "content": {"title": {"value": "Paper"}}},
            {"id": "r1", "content": {"rating": {"value": "8: Accept"}}},
            {"id": "r2", "content": {"overall_recommendation": {"value": "6: Weak Accept"}}},
            {"id": "c1", "content": {"comment": {"value": "nice"}}},
        ]}

    retriever._request = fake_request
    paper = Paper(
        source="openreview",
        title="Rated paper",
        authors=["Ann"],
        abstract="abs",
        url="https://openreview.net/forum?id=note1",
        external_id="note1",
    )
    hidden = Paper(
        source="openreview",
        title="Hidden",
        authors=["Ann"],
        abstract="abs",
        url="https://openreview.net/forum?id=note2",
        external_id="note2",
    )

    def flaky_request(method, path, **kwargs):
        if kwargs["params"]["forum"] == "note2":
            raise RuntimeError("reviews are private")
        return fake_request(method, path, **kwargs)

    retriever._request = flaky_request
    retriever.attach_ratings([paper, hidden])
    assert paper.rating_summary == "7.0 / 2 reviews"
    assert hidden.rating_summary is None

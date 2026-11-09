"""Tests for zotero_arxiv_daily.executor: normalize_path_patterns, filter_corpus, fetch_zotero_corpus, E2E."""

from datetime import datetime

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.executor import Executor, normalize_path_patterns
from zotero_arxiv_daily.openreview_state import OpenReviewState
from zotero_arxiv_daily.protocol import CorpusPaper, Paper


# ---------------------------------------------------------------------------
# normalize_path_patterns — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.zotero.include_path must be a list"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])
    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.zotero.ignore_path must be a list"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])
    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_normalize_path_patterns_accepts_none():
    assert normalize_path_patterns(None, "include_path") is None


# ---------------------------------------------------------------------------
# filter_corpus — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def _make_executor(include_patterns=None, ignore_patterns=None):
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(include_patterns, "include_path") if include_patterns else None
    executor.ignore_path_patterns = normalize_path_patterns(ignore_patterns, "ignore_path") if ignore_patterns else None
    return executor


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = _make_executor(include_patterns=["2026/survey/**", "2026/reading-group/**"])
    corpus = [
        CorpusPaper(title="Survey Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a", "archive/misc"]),
        CorpusPaper(title="Reading Group Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["notes/inbox", "2026/reading-group/week-1"]),
        CorpusPaper(title="Excluded Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = _make_executor(ignore_patterns=["archive/**", "2025/**"])
    corpus = [
        CorpusPaper(title="Active Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Archived Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["archive/misc"]),
        CorpusPaper(title="Old Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    executor = _make_executor(include_patterns=["2026/**"], ignore_patterns=["2026/ignore/**"])
    corpus = [
        CorpusPaper(title="Included Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Ignored Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["2026/ignore/topic-b"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = _make_executor()
    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert filtered == corpus


# ---------------------------------------------------------------------------
# fetch_zotero_corpus
# ---------------------------------------------------------------------------


def test_fetch_zotero_corpus(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 2
    assert corpus[0].title == "Stub Paper 1"
    assert "survey/topic-a" in corpus[0].paths[0]


def test_fetch_zotero_corpus_paper_with_zero_collections(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "data": {
                "title": "No Collection Paper",
                "abstractNote": "Abstract.",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": [],
            }
        }
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 1
    assert corpus[0].paths == []


# ---------------------------------------------------------------------------
# E2E: Executor.run()
# ---------------------------------------------------------------------------


def test_run_end_to_end(config, monkeypatch):
    """Full pipeline: Zotero fetch -> filter -> retrieve -> rerank -> TLDR -> email."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_corpus,
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_zotero_client,
    )

    # Config: source=["arxiv"], reranker="api", send_empty=false
    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    # 1. Stub pyzotero
    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    # 2. Stub OpenAI (for reranker + TLDR/affiliations)
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    retrieved = [
        make_sample_paper(title="E2E Paper 1", score=None),
        make_sample_paper(title="E2E Paper 2", score=None),
    ]

    # Import to register the arxiv retriever
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: retrieved,
    )

    # 4. Stub SMTP
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    # 5. Stub sleep (reranker/retriever)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # 6. Run
    executor = Executor(config)
    executor.run()

    # Assertions
    assert len(sent) == 1, "Email should have been sent"
    _, _, email_body = sent[0]
    assert "text/html" in email_body


def test_run_no_papers_send_empty_false(config, monkeypatch):
    """When no papers are found and send_empty=false, no email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 0, "No email should be sent when no papers and send_empty=false"


def test_run_no_papers_send_empty_true(config, monkeypatch):
    """When no papers are found and send_empty=true, empty email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1, "Email should be sent even with no papers when send_empty=true"
    _, _, body = sent[0]
    assert "text/html" in body


def _paper(title: str, source: str = "arxiv", venue: str | None = None) -> Paper:
    from tests.canned_responses import make_sample_paper

    return make_sample_paper(title=title, source=source, venue=venue)


def test_partition_openreview_digests_splits_a_large_release():
    papers = [
        _paper("arXiv paper"),
        _paper("NeurIPS one", source="openreview", venue="NeurIPS"),
        *[_paper(f"ICLR {i}", source="openreview", venue="ICLR") for i in range(3)],
    ]
    daily, digests = Executor.partition_openreview_digests(papers, min_papers=3)
    assert [paper.title for paper in daily] == ["arXiv paper", "NeurIPS one"]
    assert len(digests) == 1
    venue, venue_papers = digests[0]
    assert venue == "ICLR"
    assert len(venue_papers) == 3


def test_run_sends_conference_digest_separately(config, monkeypatch):
    import smtplib
    from email import message_from_string
    from email.header import decode_header

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv", "openreview"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        config.executor.max_paper_num = 5
        config.source.openreview.username = "reviewer@example.com"
        config.source.openreview.password = "secret"
        config.source.openreview.venues = ["iclr"]
        config.source.openreview.digest_min_papers = 3
        config.source.openreview.digest_paper_num = 2

    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    import zotero_arxiv_daily.retriever.openreview_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [_paper("Daily arXiv paper")])
    conference = []
    for i in range(4):
        paper = _paper(f"ICLR {i}", source="openreview", venue="ICLR")
        paper.status = "accepted"
        paper.conference_year = "2026"
        paper.external_id = f"iclr-{i}"
        paper.url = f"https://openreview.net/forum?id=iclr-{i}"
        conference.append(paper)
    monkeypatch.setattr(registered_retrievers["openreview"], "retrieve_papers", lambda self: conference)
    monkeypatch.setattr(registered_retrievers["openreview"], "attach_ratings", lambda self, papers: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.attach_arxiv_links", lambda papers, **kwargs: None)

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    Executor(config).run()

    assert len(sent) == 2

    def subject_of(raw: str) -> str:
        header = message_from_string(raw)["Subject"]
        chunks = []
        for text, encoding in decode_header(header):
            chunks.append(text.decode(encoding or "utf-8") if isinstance(text, bytes) else text)
        return "".join(chunks)

    subjects = [subject_of(body) for _, _, body in sent]
    assert any(subject.startswith("Daily arXiv") for subject in subjects)
    assert any(subject.startswith("ICLR 2026 精选") for subject in subjects)
    digest = next(body for _, _, body in sent if subject_of(body).startswith("ICLR 2026 精选"))
    html = message_from_string(digest).get_payload(decode=True).decode()
    assert "4 papers ranked by relevance" in html
    assert "first 4" in html


def test_run_persists_seen_ids_and_pages(config, monkeypatch, tmp_path):
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv", "openreview"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        config.executor.max_paper_num = 5
        config.source.openreview.username = "reviewer@example.com"
        config.source.openreview.password = "secret"
        config.source.openreview.venues = ["iclr"]
        config.source.openreview.digest_min_papers = 3
        config.source.openreview.digest_paper_num = 2
        config.source.openreview.persist_state = True

    monkeypatch.setattr("zotero_arxiv_daily.executor.project_root", lambda: tmp_path)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    import zotero_arxiv_daily.retriever.openreview_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    openreview_papers = []
    for i in range(4):
        paper = _paper(f"ICLR {i}", source="openreview", venue="ICLR")
        paper.external_id = f"note-{i}"
        paper.status = "accepted"
        paper.conference_year = "2026"
        paper.url = f"https://openreview.net/forum?id=note-{i}"
        openreview_papers.append(paper)
    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [_paper("Daily arXiv paper")])
    monkeypatch.setattr(registered_retrievers["openreview"], "retrieve_papers", lambda self: openreview_papers)
    monkeypatch.setattr(registered_retrievers["openreview"], "attach_ratings", lambda self, papers: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.attach_arxiv_links", lambda papers, **kwargs: None)
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp([]))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    Executor(config).run()

    saved = OpenReviewState.load(tmp_path)
    assert saved.seen_ids == {f"note-{i}" for i in range(4)}
    assert (tmp_path / "site" / "index.html").is_file()
    archived = list((tmp_path / "data" / "pages").glob("*.html"))
    assert any(path.name.endswith("-daily.html") for path in archived)
    assert any(path.name == "ICLR-2026-digest.html" for path in archived)


def test_daily_scope_does_not_retrieve_openreview(config, monkeypatch, tmp_path):
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv", "openreview"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        config.source.openreview.username = "reviewer@example.com"
        config.source.openreview.password = "secret"
        config.source.openreview.venues = ["iclr"]

    monkeypatch.setenv("RUN_SCOPE", "daily")
    monkeypatch.setattr("zotero_arxiv_daily.executor.project_root", lambda: tmp_path)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    import zotero_arxiv_daily.retriever.openreview_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    calls = {"openreview": 0}

    def retrieve_openreview(self):
        calls["openreview"] += 1
        return []

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [_paper("Daily arXiv paper")])
    monkeypatch.setattr(registered_retrievers["openreview"], "retrieve_papers", retrieve_openreview)
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp([]))

    Executor(config).run()

    assert calls["openreview"] == 0


def test_conference_scope_enriches_before_the_mailed_check(config, monkeypatch, tmp_path):
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv", "openreview"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        config.executor.max_paper_num = 5
        config.source.openreview.username = "reviewer@example.com"
        config.source.openreview.password = "secret"
        config.source.openreview.venues = ["iclr"]
        config.source.openreview.persist_state = True

    unfinished = _paper("Still missing", source="openreview", venue="ICLR")
    unfinished.abstract = ""
    unfinished.status = "accepted"
    unfinished.conference_year = "2026"
    unfinished.external_id = "missing"
    unfinished.url = "https://openreview.net/forum?id=missing"
    state = OpenReviewState(mailed={"ICLR-2026-digest": ["missing"]})
    state.save(tmp_path)

    monkeypatch.setenv("RUN_SCOPE", "conference")
    monkeypatch.setattr("zotero_arxiv_daily.executor.project_root", lambda: tmp_path)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    import zotero_arxiv_daily.retriever.openreview_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    seen = {"titles": []}

    def enrich(papers, **kwargs):
        seen["titles"] = [paper.title for paper in papers]
        for paper in papers:
            paper.abstract = "Filled in this run"

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [_paper("Daily arXiv paper")])
    monkeypatch.setattr(registered_retrievers["openreview"], "retrieve_papers", lambda self: [unfinished])
    monkeypatch.setattr(registered_retrievers["openreview"], "attach_ratings", lambda self, papers: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.enrich_papers", enrich)
    monkeypatch.setattr("zotero_arxiv_daily.executor.attach_arxiv_links", lambda papers, **kwargs: None)
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp([]))

    Executor(config).run()

    assert seen["titles"] == ["Still missing"]

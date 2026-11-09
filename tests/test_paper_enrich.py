"""Progress reporting while missing abstracts are filled in."""

from zotero_arxiv_daily.paper_enrich import ArxivRateLimit, enrich_papers, load_cache
from zotero_arxiv_daily.protocol import Paper


def _paper(title: str) -> Paper:
    return Paper(
        source="openreview",
        title=title,
        authors=["A"],
        abstract="Already known",
        url="https://example.com/paper",
        arxiv_url="https://arxiv.org/abs/1",
    )


def test_enrich_papers_logs_percentage(monkeypatch, tmp_path):
    messages = []
    monkeypatch.setattr(
        "zotero_arxiv_daily.paper_enrich.logger.info",
        lambda message: messages.append(message),
    )
    enrich_papers([_paper(f"Paper {i}") for i in range(4)], root=tmp_path, pause_seconds=0)
    assert messages[0] == "Enriching papers: 0% (0/4)"
    assert "Enriching papers: 25% (1/4) Paper 0" in messages
    assert "Enriching papers: 50% (2/4) Paper 1" in messages
    assert "Enriching papers: 100% (4/4) Paper 3" in messages


def test_enrich_papers_logs_a_heartbeat_inside_one_percent(monkeypatch, tmp_path):
    messages = []
    monkeypatch.setattr(
        "zotero_arxiv_daily.paper_enrich.logger.info",
        lambda message: messages.append(message),
    )
    enrich_papers([_paper(f"Paper {i}") for i in range(200)], root=tmp_path, pause_seconds=0)
    assert any(message.startswith("Enriching papers: 12% (25/200)") for message in messages)


def test_enrich_does_not_query_arxiv_when_the_abstract_exists(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("zotero_arxiv_daily.paper_enrich.lookup_arxiv", lambda title: calls.append(title))
    paper = _paper("Already abstracted")
    paper.arxiv_url = None
    enrich_papers([paper], root=tmp_path, pause_seconds=0)
    assert calls == []


def test_enrich_stops_after_the_lookup_budget(monkeypatch, tmp_path):
    calls = []

    def lookup(title):
        calls.append(title)
        return None

    monkeypatch.setattr("zotero_arxiv_daily.paper_enrich.lookup_arxiv", lookup)
    monkeypatch.setattr("zotero_arxiv_daily.paper_enrich._detail_abstract", lambda url: "")
    monkeypatch.setattr("zotero_arxiv_daily.paper_enrich.time.sleep", lambda delay: None)
    papers = []
    for index in range(5):
        paper = _paper(f"Needs abstract {index}")
        paper.abstract = ""
        paper.arxiv_url = None
        papers.append(paper)
    enrich_papers(papers, root=tmp_path, pause_seconds=0, max_lookups=2)
    assert calls == ["Needs abstract 0", "Needs abstract 1"]
    assert len(load_cache(tmp_path)) == 2


def test_repeated_rate_limits_are_not_cached(monkeypatch, tmp_path):
    calls = []

    def lookup(title):
        calls.append(title)
        raise ArxivRateLimit(title)

    monkeypatch.setattr("zotero_arxiv_daily.paper_enrich.lookup_arxiv", lookup)
    monkeypatch.setattr("zotero_arxiv_daily.paper_enrich._detail_abstract", lambda url: "")
    papers = []
    for index in range(5):
        paper = _paper(f"Limited {index}")
        paper.abstract = ""
        paper.arxiv_url = None
        papers.append(paper)
    enrich_papers(papers, root=tmp_path, pause_seconds=0)
    assert calls == ["Limited 0", "Limited 1", "Limited 2"]
    assert load_cache(tmp_path) == {}

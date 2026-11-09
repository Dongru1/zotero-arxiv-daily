"""Progress reporting while missing abstracts are filled in."""

from zotero_arxiv_daily.paper_enrich import enrich_papers
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

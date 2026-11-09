"""Parsers and title matching for official conference lists."""

import json

import requests

from zotero_arxiv_daily.conference_programs import (
    candidate_years,
    load_conference_programs,
    papers_for_venue_year,
    parse_dblp_hits,
    parse_icassp_accepted,
    parse_isca_index,
    select_newest,
)
from zotero_arxiv_daily.paper_enrich import titles_match


ICASSP_HTML = """
<table>
<tr><td>10540</td><td>A Real Paper Title</td></tr>
<tr><td>2</td><td>Another Paper</td></tr>
</table>
"""

ISCA_HTML = """
<h4>Keynote1 - John Hansen: A Talk</h4>
<p><a href="hansen26_interspeech.html">A Talk</a><br>John Hansen</p>
<h4>Speech Synthesis</h4>
<p><a href="liu26_interspeech.html">Controllable Speech</a><br>Min Liu, Jing Yin</p>
"""

DBLP = {
    "result": {
        "hits": {
            "hit": [{
                "info": {
                    "title": "Spoken Paper.",
                    "year": "2024",
                    "key": "conf/slt/Liu24",
                    "ee": "https://doi.org/10.example/slt",
                    "authors": {"author": [{"text": "Min Liu"}]},
                }
            }]
        }
    }
}


def test_icassp_parser_reads_titles():
    papers = parse_icassp_accepted(ICASSP_HTML, "2026")
    assert [paper.title for paper in papers] == ["A Real Paper Title", "Another Paper"]
    assert papers[0].external_id == "icassp-2026-10540"
    assert papers[0].status if False else papers[0].venue == "ICASSP"


def test_isca_parser_skips_keynotes():
    papers = parse_isca_index(ISCA_HTML, "2026")
    assert [paper.title for paper in papers] == ["Controllable Speech"]
    assert papers[0].authors == ["Min Liu", "Jing Yin"]
    assert papers[0].url.endswith("liu26_interspeech.html")


def test_dblp_parser_reads_title_and_link():
    papers = parse_dblp_hits(DBLP, "SLT", "2024")
    assert papers[0].title == "Spoken Paper"
    assert papers[0].url == "https://doi.org/10.example/slt"
    assert papers[0].year == "2024"


def test_slt_year_falls_back_two_years():
    years = candidate_years(2026, extra_back=2)
    assert select_newest(years, lambda year: year == 2024) == 2024


class _Response:
    def __init__(self, status: int, text: str = ""):
        self.status_code = status
        self.text = text
        self.headers = {"Retry-After": "2"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")


def test_dblp_retries_when_rate_limited(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return _Response(429)
        return _Response(200, json.dumps(DBLP))

    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.requests.get", fake_get)
    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.time.sleep", lambda delay: None)
    papers = papers_for_venue_year("SLT", "dblp", "https://dblp.org/search/publ/api?q=slt2024", 2024)
    assert len(calls) == 2
    assert papers[0].title == "Spoken Paper"


def test_slt_uses_older_year_when_newer_lists_are_missing(monkeypatch):
    def fake_get(url, **kwargs):
        if "slt2024" in url:
            return _Response(200, json.dumps(DBLP))
        return _Response(404)

    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.requests.get", fake_get)
    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.time.sleep", lambda delay: None)
    papers = [paper for paper in load_conference_programs(2026) if paper.venue == "SLT"]
    assert [paper.title for paper in papers] == ["Spoken Paper"]


def test_failed_newer_list_does_not_fall_back_to_an_older_year(monkeypatch):
    def fake_get(url, **kwargs):
        if "slt2026" in url:
            raise requests.ConnectionError("temporary failure")
        if "slt2024" in url:
            return _Response(200, json.dumps(DBLP))
        return _Response(404)

    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.requests.get", fake_get)
    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.time.sleep", lambda delay: None)
    papers = [paper for paper in load_conference_programs(2026) if paper.venue == "SLT"]
    assert papers == []


def test_arxiv_title_match_ignores_case_and_punctuation():
    assert titles_match("Hello, World!", "hello world")
    assert not titles_match("Hello World", "Hello There")

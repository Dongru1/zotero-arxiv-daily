"""Parsers and title matching for official conference lists."""

import json

import requests

from zotero_arxiv_daily.conference_programs import (
    biennial_years,
    load_conference_programs,
    papers_for_venue_year,
    parse_anthology_volume,
    parse_dblp_hits,
    parse_icassp_accepted,
    parse_isca_index,
    parse_slt_program,
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
<a href="../../index.html">ISCA</a>
<h4>Speech Synthesis</h4>
<p><a href="liu26_interspeech.html">Controllable Speech</a><br>Min Liu, Jing Yin</p>
<a class="w3-text" href="kano25_interspeech.html">
  <p>Pick and Summarize<br><span class="w3-text w3-text-theme">Takatomo Kano, Atsunori Ogawa</span></p>
</a>
"""

ANTHOLOGY_HTML = """
<a class=align-middle href=/2026.acl-long.0/><span>Proceedings of ACL</span></a>
<strong><a class=align-middle href=/2026.acl-long.56/><span class=acl-fixed-case>BIP</span>ro: Zero-shot Poem Generation</a></strong>
<div class="card abstract-collapse" id=abstract-2026--acl-long--56><div class="card-body p-3 small">A poem abstract.</div></div>
<a href="https://aclanthology.org/2026.acl-long.56.pdf">pdf</a>
"""

SLT_HTML = """
<tr><td>Keynote</td><td>Abeer Alwan</td></tr>
<tr><td style="width:56px;">#38</td><td>Attacking UTMOS: Probing Robustness</td></tr>
<tr><td>#46</td><td>Revisiting Lexicon Evaluation</td></tr>
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
    assert [paper.title for paper in papers] == ["Controllable Speech", "Pick and Summarize"]
    assert papers[0].authors == ["Min Liu", "Jing Yin"]
    assert papers[1].authors == ["Takatomo Kano", "Atsunori Ogawa"]
    assert papers[0].url.endswith("liu26_interspeech.html")
    assert all("ISCA" not in paper.title for paper in papers)


def test_anthology_parser_reads_unquoted_links_and_abstracts():
    papers = parse_anthology_volume(ANTHOLOGY_HTML, "ACL", "2026", "2026.acl-long")
    assert [paper.title for paper in papers] == ["BIPro: Zero-shot Poem Generation"]
    assert papers[0].abstract == "A poem abstract."
    assert papers[0].external_id == "2026.acl-long.56"
    assert papers[0].url == "https://aclanthology.org/2026.acl-long.56/"


def test_slt_program_reads_numbered_rows():
    papers = parse_slt_program(SLT_HTML, "2026")
    assert [paper.title for paper in papers] == [
        "Attacking UTMOS: Probing Robustness",
        "Revisiting Lexicon Evaluation",
    ]
    assert papers[0].external_id == "slt-2026-38"
    assert papers[0].url == "https://attend.ieee.org/slt-2026/program/"


def test_dblp_parser_reads_title_and_link():
    papers = parse_dblp_hits(DBLP, "SLT", "2024")
    assert papers[0].title == "Spoken Paper"
    assert papers[0].url == "https://doi.org/10.example/slt"
    assert papers[0].year == "2024"


def test_program_list_can_limit_itself_to_one_venue(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _Response(404)

    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.requests.get", fake_get)
    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.time.sleep", lambda delay: None)
    assert load_conference_programs(2026, only_venue="ICASSP") == []
    assert calls
    assert all("ICASSP" in url for url in calls)


def test_slt_checks_only_even_years():
    assert biennial_years(2026) == [2026, 2024]
    assert biennial_years(2025) == [2026, 2024]
    assert biennial_years(2027) == [2028, 2026]
    assert select_newest(biennial_years(2026), lambda year: year == 2024) == 2024


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
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "slt-2024" in url:
            return _Response(200, SLT_HTML)
        return _Response(404)

    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.requests.get", fake_get)
    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.time.sleep", lambda delay: None)
    papers = [paper for paper in load_conference_programs(2026) if paper.venue == "SLT"]
    slt_calls = [url for url in calls if "slt-" in url]
    assert slt_calls == [
        "https://attend.ieee.org/slt-2026/program/",
        "https://attend.ieee.org/slt-2024/program/",
    ]
    assert [paper.title for paper in papers] == [
        "Attacking UTMOS: Probing Robustness",
        "Revisiting Lexicon Evaluation",
    ]


def test_failed_newer_list_does_not_fall_back_to_an_older_year(monkeypatch):
    def fake_get(url, **kwargs):
        if "slt-2026" in url:
            raise requests.ConnectionError("temporary failure")
        if "slt-2024" in url:
            return _Response(200, SLT_HTML)
        return _Response(404)

    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.requests.get", fake_get)
    monkeypatch.setattr("zotero_arxiv_daily.conference_programs.time.sleep", lambda delay: None)
    papers = [paper for paper in load_conference_programs(2026) if paper.venue == "SLT"]
    assert papers == []


def test_arxiv_title_match_ignores_case_and_punctuation():
    assert titles_match("Hello, World!", "hello world")
    assert not titles_match("Hello World", "Hello There")

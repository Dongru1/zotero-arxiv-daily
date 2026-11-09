"""Parsers and title matching for official conference lists."""

from zotero_arxiv_daily.conference_programs import (
    candidate_years,
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


def test_arxiv_title_match_ignores_case_and_punctuation():
    assert titles_match("Hello, World!", "hello world")
    assert not titles_match("Hello World", "Hello There")

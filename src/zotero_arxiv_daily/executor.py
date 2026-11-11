import math
import os
from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match, _tokenize
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper
import random
from datetime import datetime
from .reranker import get_reranker_cls
import re
from collections import Counter
from .construct_email import render_email
from .deadline import clear_budget, start_budget
from .openreview_state import OpenReviewState, persist_enabled, project_root
from .paper_enrich import attach_arxiv_links, enrich_papers
from .publish import PageRecord, publish_results, render_site_page
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


UNDER_REVIEW_KEEP = 100
_MIN_TOKEN_LEN = 3


def _normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    def _openreview_digest_settings(self) -> tuple[int, int]:
        source = self.config.source if self.config else None
        openreview = source.get("openreview") if source is not None else None
        min_papers = 30
        paper_num = 20
        if openreview is not None:
            if openreview.get("digest_min_papers") is not None:
                min_papers = int(openreview.digest_min_papers)
            if openreview.get("digest_paper_num") is not None:
                paper_num = int(openreview.digest_paper_num)
        return min_papers, paper_num

    def _persist_openreview_state(self) -> bool:
        source = self.config.source if self.config else None
        openreview = source.get("openreview") if source is not None else None
        if openreview is None:
            return False
        return persist_enabled(openreview.get("persist_state"))

    @staticmethod
    def partition_openreview_digests(
        papers: list[Paper],
        min_papers: int,
    ) -> tuple[list[Paper], list[tuple[str, list[Paper]]]]:
        """Pull a same-day conference release out of the daily pool.

        A venue is a digest when at least ``min_papers`` of its papers arrived
        together. Smaller OpenReview updates stay in the daily email.
        """
        grouped: dict[str, list[Paper]] = {}
        daily: list[Paper] = []
        for paper in papers:
            if paper.source == "openreview" and paper.venue:
                grouped.setdefault(paper.venue, []).append(paper)
            else:
                daily.append(paper)
        digests: list[tuple[str, list[Paper]]] = []
        for venue, venue_papers in grouped.items():
            if len(venue_papers) >= min_papers:
                digests.append((venue, venue_papers))
            else:
                daily.extend(venue_papers)
        return daily, digests

    def _deliver(
        self,
        papers: list[Paper],
        corpus: list[CorpusPaper],
        limit: int,
        subject: str | None = None,
        intro: str | None = None,
        with_ratings: bool = False,
    ) -> tuple[str, list[Paper]]:
        ranked: list[Paper] = []
        if papers:
            logger.info(f"Reranking {len(papers)} papers for {subject or 'the daily email'}...")
            ranked = self.reranker.rerank(papers, corpus)[:limit]
            logger.info("Generating TLDR and affiliations...")
            for paper in tqdm(ranked):
                paper.generate_tldr(self.openai_client, self.config.llm)
                paper.generate_affiliations(self.openai_client, self.config.llm)
            if with_ratings:
                self._attach_openreview_ratings(ranked)
        email_content = render_email(ranked, intro=intro)
        send_email(self.config, email_content, subject=subject)
        logger.info(f"Email sent successfully: {subject or 'daily'}")
        return email_content, ranked

    def _attach_openreview_ratings(self, papers: list[Paper]) -> None:
        retriever = self.retrievers.get("openreview")
        attach = getattr(retriever, "attach_ratings", None)
        if attach is not None:
            attach(papers)

    @staticmethod
    def _conference_year(papers: list[Paper]) -> str | None:
        years = [paper.conference_year for paper in papers if paper.conference_year]
        if not years:
            return None
        return Counter(years).most_common(1)[0][0]

    def _run_scope(self) -> str:
        scope = os.environ.get("RUN_SCOPE", "all").strip().lower()
        if scope in {"daily", "conference"}:
            return scope
        return "all"

    def _sources_for_scope(self) -> list[str]:
        scope = self._run_scope()
        sources = list(self.retrievers)
        if scope == "daily":
            return [source for source in sources if source == "arxiv"]
        if scope == "conference":
            return [source for source in sources if source == "openreview"]
        return sources

    def run(self):
        scope = self._run_scope()
        if scope == "conference":
            start_budget()
            logger.info("Conference run will publish the pages it has within 4 hours.")
        try:
            self._run(scope)
        finally:
            if scope == "conference":
                clear_budget()

    def _run(self, scope: str):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source in self._sources_for_scope():
            retriever = self.retrievers[source]
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        if scope != "daily":
            missing = [paper for paper in all_papers if not paper.abstract]
            if missing:
                enrich_papers(missing)
            all_papers = [paper for paper in all_papers if paper.abstract or not self._is_conference_paper(paper)]
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        daily_papers, accepted, reviewing = self._split_by_status(all_papers)
        published: list[PageRecord] = []
        day = datetime.now().strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y/%m/%d")
        limit = int(self.config.executor.max_paper_num)
        if scope != "conference":
            if daily_papers:
                ranked = self._rank(daily_papers, corpus)[:limit]
                self._write_email(ranked, subject=None, intro=None, with_ratings=False)
                published.append(PageRecord(
                    filename=f"{day}-daily.html",
                    html=render_site_page(f"Daily {day}", ranked),
                    kind="daily",
                    run_date=day,
                ))
            elif not accepted and not reviewing and self.config.executor.send_empty:
                self._write_email([], subject=None, intro=None, with_ratings=False)
                published.append(PageRecord(
                    filename=f"{day}-daily.html",
                    html=render_site_page(f"Daily {day}", []),
                    kind="daily",
                    run_date=day,
                ))
            elif not accepted and not reviewing:
                logger.info("No new papers found. No email will be sent.")
                self._save_openreview_progress(all_papers, published)
                return
        if scope != "daily":
            for (venue, year), papers in accepted.items():
                self._publish_group(published, self._rank(papers, corpus), venue, year, day, today, "digest", "精选", limit)
            for (venue, year), papers in reviewing.items():
                shortlisted = self._shortlist_under_review(papers, corpus, UNDER_REVIEW_KEEP)
                self._publish_group(published, self._rank(shortlisted, corpus), venue, year, day, today, "under_review", "Under Review", limit)
        self._save_openreview_progress(all_papers, published)

    @staticmethod
    def _is_conference_paper(paper: Paper) -> bool:
        return bool(paper.venue and paper.status in {"accepted", "under_review"})

    def _split_by_status(self, papers: list[Paper]):
        accepted: dict[tuple[str, str], list[Paper]] = {}
        reviewing: dict[tuple[str, str], list[Paper]] = {}
        stored = OpenReviewState.load(project_root())
        conference_keys = set(stored.indexed_urls)
        conference_titles = set(stored.indexed_titles)
        for paper in papers:
            if paper.venue and paper.status in {"accepted", "under_review"}:
                if paper.arxiv_url:
                    conference_keys.add(paper.arxiv_url)
                if "arxiv.org/abs/" in (paper.url or ""):
                    conference_keys.add(paper.url)
                conference_titles.add(_normalized_title(paper.title))
        daily = []
        for paper in papers:
            if paper.venue and paper.status == "accepted":
                accepted.setdefault((paper.venue, paper.conference_year or ""), []).append(paper)
            elif paper.venue and paper.status == "under_review":
                reviewing.setdefault((paper.venue, paper.conference_year or ""), []).append(paper)
            elif paper.arxiv_url in conference_keys or paper.url in conference_keys or _normalized_title(paper.title) in conference_titles:
                continue
            else:
                daily.append(paper)
        return daily, accepted, reviewing

    @staticmethod
    def _library_idf(corpus: list[CorpusPaper]) -> dict[str, float]:
        """Weight a word by how specific it is inside the Zotero library.

        A word that appears in nearly every saved paper, such as "model" or
        "learning", gets almost no weight. A word that appears in only a few
        papers keeps a high weight. The same weights apply to every conference.
        """
        documents = []
        for item in corpus:
            tokens = {
                token for token in _tokenize(f"{item.title} {(item.abstract or '')[:2000]}")
                if len(token) >= _MIN_TOKEN_LEN
            }
            documents.append(tokens)
        total = len(documents)
        counts: Counter[str] = Counter()
        for tokens in documents:
            counts.update(tokens)
        return {
            token: math.log((total - count + 0.5) / (count + 0.5) + 1)
            for token, count in counts.items()
        }

    @staticmethod
    def _shortlist_under_review(papers: list[Paper], corpus: list[CorpusPaper], limit: int) -> list[Paper]:
        """Keep the submissions that use the library's distinctive words.

        Embedding every under-review submission takes many hours. Any venue
        uses this same reduction, then the embedding model orders only the
        papers that remain.
        """
        if len(papers) <= limit:
            return papers
        logger.info(f"Selecting {limit} of {len(papers)} under-review papers before embedding.")
        idf = Executor._library_idf(corpus)
        if not idf:
            return papers[:limit]
        lengths = []
        token_counts = []
        for paper in papers:
            tokens = [
                token for token in _tokenize(f"{paper.title} {(paper.abstract or '')[:2000]}")
                if len(token) >= _MIN_TOKEN_LEN
            ]
            lengths.append(len(tokens))
            token_counts.append(Counter(tokens))
        average_length = sum(lengths) / max(len(lengths), 1)
        ranked = []
        for paper, tokens, length in zip(papers, token_counts, lengths):
            score = 0.0
            for token, freq in tokens.items():
                weight = idf.get(token)
                if not weight:
                    continue
                score += weight * (freq * 2.5) / (freq + 1.5 * (0.25 + 0.75 * length / max(average_length, 1)))
            ranked.append((score, paper))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [paper for _, paper in ranked[:limit]]

    def _rank(self, papers: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        if not papers:
            return []
        logger.info(f"Reranking {len(papers)} papers...")
        return self.reranker.rerank(papers, corpus)

    def _write_email(self, papers: list[Paper], subject: str | None, intro: str | None, with_ratings: bool) -> None:
        for paper in papers:
            paper.generate_tldr(self.openai_client, self.config.llm)
            paper.generate_affiliations(self.openai_client, self.config.llm)
        if with_ratings:
            self._attach_openreview_ratings(papers)
        send_email(self.config, render_email(papers, intro=intro), subject=subject)

    def _publish_group(self, published, ranked, venue, year, day, today, kind, label, email_limit) -> None:
        filename = f"{venue}-{year}-{kind}.html"
        title = f"{venue} {year} {label}".strip()
        top = ranked[:email_limit]
        mailed_key = f"{venue}-{year}-{kind}"
        top_ids = [paper.external_id or paper.title for paper in top]
        state = OpenReviewState.load(project_root()) if self._persist_openreview_state() else OpenReviewState()
        send_mail = state.mailed.get(mailed_key) != top_ids
        if send_mail:
            attach_arxiv_links(top)
        published.append(PageRecord(
            filename=filename,
            html=render_site_page(title, ranked, intro=f"{len(ranked)} papers, ranked by relevance to the Zotero library."),
            kind=kind,
            run_date=day,
            venue=venue,
            conference_year=year or None,
        ))
        if not send_mail:
            logger.info(f"Skipping {title} email because the mailed set is unchanged.")
            return
        self._write_email(
            top,
            subject=f"会议精选 {title} {today}",
            intro=f"{len(ranked)} papers ranked by relevance. This email includes the first {len(top)}.",
            with_ratings=True,
        )
        if self._persist_openreview_state():
            state.mailed[mailed_key] = top_ids
            state.save(project_root())

    def _save_openreview_progress(self, papers: list[Paper], published: list[PageRecord]) -> None:
        if not self._persist_openreview_state():
            return
        root = project_root()
        state = OpenReviewState.load(root)
        for retriever in self.retrievers.values():
            state.bootstrapped.update(getattr(retriever, "newly_bootstrapped", set()))
        for paper in papers:
            if paper.source == "openreview" and paper.external_id:
                state.seen_ids.add(paper.external_id)
            if self._is_conference_paper(paper) and paper.abstract:
                state.indexed_titles.add(_normalized_title(paper.title))
                if paper.arxiv_url:
                    state.indexed_urls.add(paper.arxiv_url)
                if "arxiv.org/abs/" in (paper.url or ""):
                    state.indexed_urls.add(paper.url)
        state.save(root)
        publish_results(published, root)
        logger.info("Saved OpenReview state and published pages.")

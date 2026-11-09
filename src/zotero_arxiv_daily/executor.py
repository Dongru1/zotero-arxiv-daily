from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .openreview_state import OpenReviewState, persist_enabled, project_root
from .publish import publish_results
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


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

    def _deliver(self, papers: list[Paper], corpus: list[CorpusPaper], limit: int, subject: str | None = None, intro: str | None = None):
        ranked: list[Paper] = []
        if papers:
            logger.info(f"Reranking {len(papers)} papers for {subject or 'the daily email'}...")
            ranked = self.reranker.rerank(papers, corpus)[:limit]
            logger.info("Generating TLDR and affiliations...")
            for paper in tqdm(ranked):
                paper.generate_tldr(self.openai_client, self.config.llm)
                paper.generate_affiliations(self.openai_client, self.config.llm)
        email_content = render_email(ranked, intro=intro)
        send_email(self.config, email_content, subject=subject)
        logger.info(f"Email sent successfully: {subject or 'daily'}")
        return email_content

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        digest_min_papers, digest_paper_num = self._openreview_digest_settings()
        daily_papers, digests = self.partition_openreview_digests(all_papers, digest_min_papers)
        for venue, venue_papers in digests:
            logger.info(
                f"{venue} released {len(venue_papers)} papers; sending a separate digest of {digest_paper_num}"
            )

        published: list[tuple[str, str]] = []
        day = datetime.now().strftime("%Y-%m-%d")
        if daily_papers:
            html = self._deliver(daily_papers, corpus, self.config.executor.max_paper_num)
            published.append((f"{day}-daily.html", html))
        elif not digests and self.config.executor.send_empty:
            html = self._deliver([], corpus, self.config.executor.max_paper_num)
            published.append((f"{day}-daily.html", html))
        elif not digests:
            logger.info("No new papers found. No email will be sent.")
            self._save_openreview_progress(all_papers, published)
            return

        today = datetime.now().strftime("%Y/%m/%d")
        for venue, venue_papers in digests:
            selected = min(digest_paper_num, len(venue_papers))
            intro = f"{venue} 今日公开 {len(venue_papers)} 篇，按文献库相关度选出 {selected} 篇。"
            html = self._deliver(
                venue_papers,
                corpus,
                digest_paper_num,
                subject=f"{venue} 精选 {today}",
                intro=intro,
            )
            published.append((f"{day}-{venue}.html", html))
        self._save_openreview_progress(all_papers, published)

    def _save_openreview_progress(self, papers: list[Paper], published: list[tuple[str, str]]) -> None:
        if not self._persist_openreview_state():
            return
        root = project_root()
        state = OpenReviewState.load(root)
        for retriever in self.retrievers.values():
            state.bootstrapped.update(getattr(retriever, "newly_bootstrapped", set()))
        for paper in papers:
            if paper.source == "openreview" and paper.external_id:
                state.seen_ids.add(paper.external_id)
        state.save(root)
        publish_results(published, root)
        logger.info("Saved OpenReview state and published pages.")

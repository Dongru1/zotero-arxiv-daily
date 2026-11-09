from datetime import date, datetime, timedelta, timezone
from time import sleep
from typing import Any

import re

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..conference_programs import ListedPaper, load_conference_programs
from ..openreview_state import OpenReviewState, project_root
from ..paper_enrich import enrich_papers
from ..protocol import Paper


# Main-conference group ids. {year} is filled with the previous, current, and next
# calendar year so a submission cycle that spans New Year is still found.
VENUE_PATTERNS = {
    "iclr": ["ICLR.cc/{year}/Conference"],
    "neurips": ["NeurIPS.cc/{year}/Conference"],
    "icml": ["ICML.cc/{year}/Conference"],
    "emnlp": ["EMNLP/{year}/Conference"],
    "acl": ["aclweb.org/ACL/{year}/Conference"],
    # SLT 2026 has no single Conference group; submissions live on the tracks below it.
    "slt": ["IEEE.org/SLT/{year}/Conference", "IEEE.org/SLT/{year}"],
    "icassp": ["ICASSP/{year}/Conference"],
    "interspeech": ["Interspeech.org/{year}/Conference"],
}

VENUE_LABELS = {
    "iclr": "ICLR",
    "neurips": "NeurIPS",
    "icml": "ICML",
    "emnlp": "EMNLP",
    "acl": "ACL",
    "slt": "SLT",
    "icassp": "ICASSP",
    "interspeech": "Interspeech",
}

# Child groups that are not the conference's own paper tracks.
_SKIP_CHILD_PARTS = {
    "Workshop",
    "Tutorials",
    "Publication_Chairs",
    "Authors",
    "Reviewers",
    "Area_Chairs",
    "Senior_Area_Chairs",
}
_INACTIVE_VENUE_SUFFIXES = (
    "/Withdrawn_Submission",
    "/Desk_Rejected_Submission",
    "/Rejected_Submission",
)
_YEAR_IN_ID = re.compile(r"(?:19|20)\d{2}")
_RATING_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")
_RATING_FIELDS = ("rating", "overall_recommendation", "recommendation")


@register_retriever("openreview")
class OpenReviewRetriever(BaseRetriever):
    """Retrieve recent public OpenReview submissions for configured conferences.

    OpenReview's notes API rejects unauthenticated callers with a browser
    challenge, so this retriever logs in with an OpenReview account and reads
    API v2. It keeps notes modified within ``lookback_hours`` that still have a
    public abstract. Notes already recorded in ``data/openreview_state.json``
    are skipped. The first time a venue has nothing in that window, it is
    filled from the single newest public day and then marked bootstrapped.
    Withdrawn, desk-rejected, and rejected notes are skipped.

    ICASSP and Interspeech do not host their main conferences on OpenReview.
    Those names stay in the venue list so a future venue id is picked up
    automatically; until then they contribute no papers.
    """

    api_base = "https://api2.openreview.net"
    page_size = 50
    request_timeout = 60

    def __init__(self, config):
        super().__init__(config)
        venues = self.retriever_config.get("venues")
        if not venues:
            raise ValueError(
                "venues must be specified for openreview. "
                f"Known names: {', '.join(sorted(VENUE_PATTERNS))}."
            )
        self.venues = [str(venue) for venue in venues]
        self.username = self.retriever_config.get("username")
        self.password = self.retriever_config.get("password")
        if not self.username or not self.password:
            raise ValueError(
                "OpenReview username and password are required. "
                "Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD."
            )
        self.lookback_hours = int(self.retriever_config.get("lookback_hours") or 48)
        self._token: str | None = None
        self.state = OpenReviewState.load(project_root())
        self.newly_bootstrapped: set[str] = set()

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "zotero-arxiv-daily (https://github.com/TideDra/zotero-arxiv-daily)",
            "Accept": "application/json",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _login(self) -> None:
        payload = self._request(
            "POST",
            "/login",
            json={"id": self.username, "password": self.password},
            authenticated=False,
        )
        token = (payload or {}).get("token")
        if not token:
            raise RuntimeError("OpenReview login succeeded but no token was returned.")
        self._token = token

    def _request(
        self,
        method: str,
        path: str,
        *,
        authenticated: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        retry_num = 5
        delay_time = 5
        url = f"{self.api_base}{path}"
        for attempt in range(retry_num):
            headers = self._headers()
            if not authenticated:
                headers.pop("Authorization", None)
            try:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    timeout=self.request_timeout,
                    **kwargs,
                )
            except requests.RequestException as exc:
                if attempt == retry_num - 1:
                    raise
                logger.warning(f"OpenReview request failed: {exc}. Retry in {delay_time}s.")
                sleep(delay_time)
                continue

            if response.status_code == 404:
                return None
            if response.status_code == 401 and authenticated and attempt == 0:
                self._login()
                continue
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retry_num - 1:
                logger.warning(
                    f"OpenReview returned {response.status_code} for {path}. Retry in {delay_time}s."
                )
                sleep(delay_time)
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                body = response.text[:300]
                raise RuntimeError(f"OpenReview request failed for {path}: {exc}. {body}") from exc
            return response.json()
        raise RuntimeError(f"OpenReview request failed for {path} after {retry_num} attempts.")

    @staticmethod
    def _content_value(content: dict[str, Any] | None, key: str, default: Any = None) -> Any:
        if not content or key not in content:
            return default
        value = content.get(key)
        if isinstance(value, dict) and "value" in value:
            return value.get("value", default)
        return value

    def _patterns(self, venue: str) -> list[str]:
        key = venue.strip()
        patterns = VENUE_PATTERNS.get(key.lower())
        if patterns:
            return patterns
        if "/" in key:
            return [key]
        known = ", ".join(sorted(VENUE_PATTERNS))
        raise ValueError(f"Unknown OpenReview venue {venue!r}. Known names: {known}. Or pass a venue id.")

    def _get_group(self, group_id: str) -> dict[str, Any] | None:
        payload = self._request("GET", "/groups", params={"id": group_id})
        groups = (payload or {}).get("groups") or []
        return groups[0] if groups else None

    def _direct_children(self, parent_id: str) -> list[dict[str, Any]]:
        children = []
        offset = 0
        while True:
            payload = self._request(
                "GET",
                "/groups",
                params={"prefix": parent_id, "limit": 100, "offset": offset},
            )
            groups = (payload or {}).get("groups") or []
            for group in groups:
                group_id = group.get("id") or ""
                if not group_id.startswith(parent_id + "/"):
                    continue
                rest = group_id[len(parent_id) + 1 :]
                if "/" in rest or rest in _SKIP_CHILD_PARTS or "Workshop" in rest:
                    continue
                children.append(group)
            if len(groups) < 100:
                break
            offset += 100
            if offset >= 500:
                break
        return children

    def _digest_label(self, venue: str) -> str:
        key = venue.strip().lower()
        if key in VENUE_LABELS:
            return VENUE_LABELS[key]
        head = venue.split("/")[0].split(".")[0]
        return head.upper() if head.islower() else head

    def _submission_groups(self, venues: list[str], years: list[int]) -> list[tuple[str, dict[str, Any]]]:
        found: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for venue in venues:
            matched = False
            for year in years:
                for pattern in self._patterns(venue):
                    group_id = pattern.format(year=year)
                    group = self._get_group(group_id)
                    if group is None:
                        continue
                    candidates = [group]
                    if not self._content_value(group.get("content"), "submission_id"):
                        candidates = []
                        for child in self._direct_children(group_id):
                            full = self._get_group(child["id"])
                            if full is not None:
                                candidates.append(full)
                    for candidate in candidates:
                        submission_id = self._content_value(candidate.get("content"), "submission_id")
                        candidate_id = candidate.get("id")
                        if not submission_id or not candidate_id or candidate_id in seen:
                            continue
                        seen.add(candidate_id)
                        found.append((self._digest_label(venue), candidate))
                        matched = True
            if not matched:
                logger.warning(
                    f"No OpenReview submissions found for {venue} in {years[0]}-{years[-1]}. "
                    "The main conference may not be hosted on OpenReview."
                )
        return found

    @staticmethod
    def _note_time(note: dict[str, Any]) -> datetime | None:
        timestamp = note.get("tmdate") or note.get("mdate") or note.get("cdate")
        if not isinstance(timestamp, (int, float)):
            return None
        if timestamp > 10**12:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def _is_inactive(self, note: dict[str, Any], group: dict[str, Any]) -> bool:
        content = group.get("content") or {}
        inactive_ids = {
            self._content_value(content, "withdrawn_venue_id"),
            self._content_value(content, "desk_rejected_venue_id"),
            self._content_value(content, "rejected_venue_id"),
        }
        inactive_ids.discard(None)
        venue_id = self._content_value(note.get("content"), "venueid") or ""
        if venue_id in inactive_ids:
            return True
        return any(str(venue_id).endswith(suffix) for suffix in _INACTIVE_VENUE_SUFFIXES)

    def _usable_note(self, note: dict[str, Any], group: dict[str, Any]) -> bool:
        if self._is_inactive(note, group):
            return False
        if not self._content_value(note.get("content"), "abstract"):
            return False
        if not self._content_value(note.get("content"), "title"):
            return False
        note_id = note.get("id")
        if note_id and note_id in self.state.seen_ids:
            return False
        return True

    def _collect_notes(
        self,
        group: dict[str, Any],
        *,
        cutoff: datetime | None = None,
        latest_day: bool = False,
        unlimited: bool = False,
    ) -> tuple[list[dict[str, Any]], date | None]:
        """Collect notes newest-first.

        ``cutoff`` stops at the first note older than the lookback window.
        ``latest_day`` keeps only the UTC date of the newest note and does not
        read any earlier day.
        """
        submission_id = self._content_value(group.get("content"), "submission_id")
        collected: list[dict[str, Any]] = []
        anchor: date | None = None
        offset = 0
        while unlimited or len(collected) < 100000:
            payload = self._request(
                "GET",
                "/notes",
                params={
                    "invitation": submission_id,
                    "sort": "tmdate:desc",
                    "limit": self.page_size,
                    "offset": offset,
                },
            )
            notes = (payload or {}).get("notes") or []
            if not notes:
                break
            stop = False
            for note in notes:
                noted_at = self._note_time(note)
                if latest_day:
                    if noted_at is None:
                        continue
                    if anchor is None:
                        anchor = noted_at.date()
                    if noted_at.date() != anchor:
                        stop = True
                        break
                elif cutoff is not None and (noted_at is None or noted_at < cutoff):
                    stop = True
                    break
                if self._usable_note(note, group):
                    note["_conference_year"] = _conference_year(group.get("id") or "")
                    collected.append(note)
            if stop or len(notes) < self.page_size:
                break
            offset += self.page_size
        return collected, anchor

    def _paper_status(self, note: dict[str, Any], group_id: str) -> str | None:
        content = note.get("content") or {}
        venue = str(self._content_value(content, "venue") or "")
        venueid = str(self._content_value(content, "venueid") or "")
        blob = f"{venue} {venueid}".lower()
        if any(marker in blob for marker in ("withdrawn", "desk_rejected", "desk rejected", "/rejected")):
            return None
        if any(marker in blob for marker in ("oral", "spotlight", "poster")):
            return "accepted"
        if venueid and "submission" not in venueid.lower() and group_id and venueid.rstrip("/") == group_id.rstrip("/"):
            return "accepted"
        return "under_review"

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        self._login()
        now = datetime.now(timezone.utc)
        years = [now.year - 1, now.year, now.year + 1]
        groups = self._submission_groups(self.venues, years)

        notes = []
        seen_this_run: set[str] = set()
        for label, group in groups:
            try:
                group_notes, _ = self._collect_notes(group, unlimited=True)
            except (RuntimeError, requests.RequestException) as exc:
                logger.warning(f"Skipping {group.get('id')}: {exc}")
                continue
            group_id = group.get("id") or ""
            kept = 0
            for note in group_notes:
                status = self._paper_status(note, group_id)
                if status is None:
                    continue
                note_id = note.get("id")
                if note_id and note_id in seen_this_run:
                    continue
                if note_id:
                    seen_this_run.add(note_id)
                note["_digest_label"] = label
                note["_status"] = status
                notes.append(note)
                kept += 1
            logger.info(f"Retrieved {kept} notes for {group_id}")

        try:
            for paper in load_conference_programs(now.year):
                notes.append(_program_note(paper))
        except Exception as exc:
            logger.warning(f"Skipping official conference lists: {exc}")

        if not notes:
            logger.warning("No conference papers found.")
        return notes

    def retrieve_papers(self) -> list[Paper]:
        papers = []
        for raw_paper in self._retrieve_raw_papers():
            try:
                paper = self.convert_to_paper(raw_paper)
            except Exception as exc:
                logger.warning(f"Skipping paper {raw_paper.get('id', raw_paper)}: {exc}")
                continue
            if paper is not None:
                papers.append(paper)
        needs = [
            paper for paper in papers
            if not paper.abstract or (
                paper.status == "accepted" and not paper.arxiv_url and "openreview.net/forum" not in (paper.url or "")
            )
        ]
        if needs:
            enrich_papers(needs)
        return [paper for paper in papers if paper.abstract]

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        content = raw_paper.get("content") or {}
        title = self._content_value(content, "title") or ""
        title = " ".join(str(title).split())
        if not title:
            return None
        abstract = self._content_value(content, "abstract") or ""
        abstract = " ".join(str(abstract).split())
        authors = self._content_value(content, "authors") or []
        if isinstance(authors, str):
            authors = [authors]
        authors = [str(author).strip() for author in authors if str(author).strip()]
        if not authors:
            authors = ["Anonymous"]
        note_id = raw_paper.get("id") or ""
        url = raw_paper.get("_url") or (f"https://openreview.net/forum?id={note_id}" if note_id else "https://openreview.net")
        pdf_url = f"https://openreview.net/pdf?id={note_id}" if note_id and not raw_paper.get("_url") else None
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            full_text=None,
            venue=raw_paper.get("_digest_label"),
            external_id=raw_paper.get("id") or None,
            conference_year=raw_paper.get("_conference_year"),
            status=raw_paper.get("_status") or "under_review",
            arxiv_url=_arxiv_from_content(content),
        )

    def attach_ratings(self, papers: list[Paper]) -> None:
        """Fill public review scores for papers that will be shown.

        A forum with no visible reviews leaves ``rating_summary`` empty.
        A failed request for one paper does not stop the rest.
        """
        if not any(paper.source == "openreview" and paper.external_id for paper in papers):
            return
        if not self._token:
            try:
                self._login()
            except (RuntimeError, requests.RequestException) as exc:
                logger.warning(f"Skipping OpenReview ratings: {exc}")
                return
        for paper in papers:
            if paper.source != "openreview" or not paper.external_id:
                continue
            try:
                paper.rating_summary = self._rating_summary(paper.external_id)
            except (RuntimeError, requests.RequestException) as exc:
                logger.warning(f"Skipping rating for {paper.external_id}: {exc}")

    def _rating_summary(self, forum_id: str) -> str | None:
        payload = self._request("GET", "/notes", params={"forum": forum_id, "limit": 100})
        scores: list[float] = []
        for note in (payload or {}).get("notes") or []:
            if note.get("id") == forum_id:
                continue
            content = note.get("content") or {}
            for field in _RATING_FIELDS:
                score = _parse_rating(self._content_value(content, field))
                if score is not None:
                    scores.append(score)
                    break
        if not scores:
            return None
        average = sum(scores) / len(scores)
        noun = "review" if len(scores) == 1 else "reviews"
        return f"{average:.1f} / {len(scores)} {noun}"


def _conference_year(group_id: str) -> str | None:
    match = _YEAR_IN_ID.search(group_id)
    return match.group(0) if match else None


def _program_note(paper: ListedPaper) -> dict[str, Any]:
    return {
        "id": paper.external_id,
        "_digest_label": paper.venue,
        "_conference_year": paper.year,
        "_status": "accepted",
        "_url": paper.url,
        "content": {
            "title": {"value": paper.title},
            "abstract": {"value": paper.abstract},
            "authors": {"value": paper.authors},
        },
    }


def _arxiv_from_content(content: dict[str, Any]) -> str | None:
    for key in ("arxiv", "arxiv_id", "code", "pdf"):
        value = content.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, str) and "arxiv.org/abs/" in value:
            return value.split()[0]
    return None


def _parse_rating(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _RATING_NUMBER.search(value)
        if match:
            return float(match.group(1))
    return None

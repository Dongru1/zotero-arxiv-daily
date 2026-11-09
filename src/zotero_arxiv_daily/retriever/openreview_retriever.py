from datetime import date, datetime, timedelta, timezone
from time import sleep
from typing import Any

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..openreview_state import OpenReviewState, project_root
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
    max_notes_per_venue = 3000
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
        while len(collected) < self.max_notes_per_venue:
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
                elif noted_at is None or cutoff is None or noted_at < cutoff:
                    stop = True
                    break
                if self._usable_note(note, group):
                    collected.append(note)
            if stop or len(notes) < self.page_size:
                break
            offset += self.page_size
        if len(collected) >= self.max_notes_per_venue:
            logger.warning(
                f"Stopped after {self.max_notes_per_venue} recent notes for {submission_id}."
            )
        return collected, anchor

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        self._login()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.lookback_hours)
        years = [now.year - 1, now.year, now.year + 1]
        groups = self._submission_groups(self.venues, years)

        grouped: dict[str, list[dict[str, Any]]] = {}
        for label, group in groups:
            grouped.setdefault(label, []).append(group)

        notes = []
        seen_this_run: set[str] = set()
        for label, venue_groups in grouped.items():
            recent: list[dict[str, Any]] = []
            for group in venue_groups:
                try:
                    group_notes, _ = self._collect_notes(group, cutoff=cutoff)
                except (RuntimeError, requests.RequestException) as exc:
                    logger.warning(f"Skipping {group.get('id')}: {exc}")
                    continue
                recent.extend(group_notes)
            chosen = recent
            if not chosen and label not in self.state.bootstrapped:
                latest_day: date | None = None
                day_notes: list[dict[str, Any]] = []
                queried = False
                for group in venue_groups:
                    try:
                        batch, day = self._collect_notes(group, latest_day=True)
                    except (RuntimeError, requests.RequestException) as exc:
                        logger.warning(f"Skipping bootstrap for {group.get('id')}: {exc}")
                        continue
                    queried = True
                    if day is None:
                        continue
                    if latest_day is None or day > latest_day:
                        latest_day = day
                        day_notes = batch
                    elif day == latest_day:
                        day_notes.extend(batch)
                if queried:
                    chosen = day_notes
                    self.newly_bootstrapped.add(label)
                    logger.info(f"Bootstrapped {label} from the last public day {latest_day}.")
            logger.info(f"Retrieved {len(chosen)} notes for {label}")
            for note in chosen:
                note_id = note.get("id")
                if note_id and note_id in seen_this_run:
                    continue
                if note_id:
                    seen_this_run.add(note_id)
                note["_digest_label"] = label
                notes.append(note)

        if not notes:
            logger.warning(f"No OpenReview papers found in the last {self.lookback_hours} hours.")
        return notes

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        content = raw_paper.get("content") or {}
        title = self._content_value(content, "title") or ""
        title = " ".join(str(title).split())
        if not title:
            return None
        abstract = self._content_value(content, "abstract") or ""
        abstract = " ".join(str(abstract).split())
        if not abstract:
            return None
        authors = self._content_value(content, "authors") or []
        if isinstance(authors, str):
            authors = [authors]
        authors = [str(author).strip() for author in authors if str(author).strip()]
        if not authors:
            authors = ["Anonymous"]
        note_id = raw_paper.get("id") or ""
        url = f"https://openreview.net/forum?id={note_id}" if note_id else "https://openreview.net"
        pdf_url = f"https://openreview.net/pdf?id={note_id}" if note_id else None
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
        )

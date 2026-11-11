import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from hydra.utils import get_original_cwd


STATE_PATH = Path("data") / "openreview_state.json"
PAGES_DIR = Path("data") / "pages"
SITE_DIR = Path("site")


def project_root() -> Path:
    try:
        return Path(get_original_cwd())
    except ValueError:
        return Path.cwd()


@dataclass
class OpenReviewState:
    seen_ids: set[str] = field(default_factory=set)
    bootstrapped: set[str] = field(default_factory=set)
    mailed: dict[str, list[str]] = field(default_factory=dict)
    indexed_titles: set[str] = field(default_factory=set)
    indexed_urls: set[str] = field(default_factory=set)

    @classmethod
    def _from_payload(cls, payload: dict) -> "OpenReviewState":
        venues = payload.get("venues") or {}
        bootstrapped = {
            name
            for name, info in venues.items()
            if isinstance(info, dict) and info.get("bootstrapped")
        }
        return cls(
            seen_ids={str(note_id) for note_id in payload.get("seen_ids") or []},
            bootstrapped=bootstrapped,
            mailed={str(key): list(value) for key, value in (payload.get("mailed") or {}).items()},
            indexed_titles={str(title) for title in payload.get("indexed_titles") or []},
            indexed_urls={str(url) for url in payload.get("indexed_urls") or []},
        )

    def merge(self, other: "OpenReviewState") -> None:
        self.seen_ids.update(other.seen_ids)
        self.bootstrapped.update(other.bootstrapped)
        self.mailed.update(other.mailed)
        self.indexed_titles.update(other.indexed_titles)
        self.indexed_urls.update(other.indexed_urls)

    @classmethod
    def load(cls, root: Path | None = None) -> "OpenReviewState":
        root = root or project_root()
        state = cls()
        path = root / STATE_PATH
        if path.is_file():
            state.merge(cls._from_payload(json.loads(path.read_text())))
        folder = root / "data" / "openreview_state"
        if folder.is_dir():
            for shard in sorted(folder.glob("*.json")):
                state.merge(cls._from_payload(json.loads(shard.read_text())))
        return state

    def save(self, root: Path | None = None) -> Path:
        root = root or project_root()
        venue = os.environ.get("CONFERENCE_VENUE", "").strip().lower()
        if venue and venue != "all":
            path = root / "data" / "openreview_state" / f"{venue}.json"
        else:
            path = root / STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "venues": {
                name: {"bootstrapped": True}
                for name in sorted(self.bootstrapped)
            },
            "seen_ids": sorted(self.seen_ids),
            "mailed": {key: value for key, value in sorted(self.mailed.items())},
            "indexed_titles": sorted(self.indexed_titles),
            "indexed_urls": sorted(self.indexed_urls),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path


def persist_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}

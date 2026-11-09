import json
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

    @classmethod
    def load(cls, root: Path | None = None) -> "OpenReviewState":
        path = (root or project_root()) / STATE_PATH
        if not path.is_file():
            return cls()
        payload = json.loads(path.read_text())
        venues = payload.get("venues") or {}
        bootstrapped = {
            name
            for name, info in venues.items()
            if isinstance(info, dict) and info.get("bootstrapped")
        }
        return cls(
            seen_ids={str(note_id) for note_id in payload.get("seen_ids") or []},
            bootstrapped=bootstrapped,
        )

    def save(self, root: Path | None = None) -> Path:
        path = (root or project_root()) / STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "venues": {
                name: {"bootstrapped": True}
                for name in sorted(self.bootstrapped)
            },
            "seen_ids": sorted(self.seen_ids),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path


def persist_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}

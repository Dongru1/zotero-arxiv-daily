"""Stop a long conference run in time to publish the pages it already has."""

import time

CONFERENCE_BUDGET_SECONDS = 4 * 60 * 60
_deadline: float | None = None


class TimeBudgetExceeded(RuntimeError):
    """The conference run must stop and publish instead of waiting longer."""


def start_budget(seconds: float = CONFERENCE_BUDGET_SECONDS) -> None:
    global _deadline
    _deadline = time.monotonic() + seconds


def clear_budget() -> None:
    global _deadline
    _deadline = None


def remaining() -> float | None:
    if _deadline is None:
        return None
    return _deadline - time.monotonic()


def expired() -> bool:
    left = remaining()
    return left is not None and left <= 0

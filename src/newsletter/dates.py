from __future__ import annotations

from datetime import UTC, datetime


def current_hf_week(now: datetime | None = None) -> str:
    timestamp = now or datetime.now(tz=UTC)
    iso_year, iso_week, _ = timestamp.date().isocalendar()
    return f"{iso_year}-W{iso_week:02d}"

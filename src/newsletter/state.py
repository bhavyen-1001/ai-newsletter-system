from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from newsletter.models import NewsletterIssue


@dataclass
class SentPaperState:
    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "SentPaperState":
        if not path.exists():
            return cls(path=path, data={"sent_papers": []})
        return cls(path=path, data=json.loads(path.read_text(encoding="utf-8")))

    @property
    def sent_ids(self) -> set[str]:
        return {entry["arxiv_id"] for entry in self.data.get("sent_papers", [])}

    def is_sent(self, arxiv_id: str) -> bool:
        return arxiv_id in self.sent_ids

    def mark_issue_sent(self, issue: NewsletterIssue, *, campaign_id: str) -> None:
        sent_at = datetime.now(tz=UTC).isoformat()
        existing = self.sent_ids
        entries = self.data.setdefault("sent_papers", [])
        for paper in issue.papers:
            if paper.metadata.arxiv_id in existing:
                continue
            entries.append(
                {
                    "arxiv_id": paper.metadata.arxiv_id,
                    "title": paper.metadata.title,
                    "authors": paper.metadata.authors,
                    "week": issue.week,
                    "sent_at": sent_at,
                    "mailchimp_campaign_id": campaign_id,
                }
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(self.path)

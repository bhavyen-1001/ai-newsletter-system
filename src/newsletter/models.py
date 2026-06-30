from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PaperCandidate:
    arxiv_id: str
    hf_url: str


@dataclass(frozen=True)
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    arxiv_url: str
    pdf_url: str
    hf_url: str


@dataclass(frozen=True)
class PaperSummary:
    executive_summary: str
    problem: str
    method: str
    why_it_matters: str
    limitations: str


@dataclass(frozen=True)
class NewsletterPaper:
    metadata: PaperMetadata
    summary: PaperSummary
    source_week: str


@dataclass(frozen=True)
class NewsletterIssue:
    week: str
    generated_at: str
    papers: list[NewsletterPaper]


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)

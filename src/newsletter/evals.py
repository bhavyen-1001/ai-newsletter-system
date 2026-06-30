from __future__ import annotations

from newsletter.models import NewsletterIssue, NewsletterPaper


def validate_issue(issue: NewsletterIssue) -> list[str]:
    errors: list[str] = []
    if not issue.papers:
        errors.append("Newsletter has no papers.")

    for index, paper in enumerate(issue.papers, start=1):
        errors.extend(_validate_paper(paper, index))

    return errors


def raise_if_invalid(issue: NewsletterIssue) -> None:
    errors = validate_issue(issue)
    if errors:
        raise ValueError("Newsletter validation failed:\n- " + "\n- ".join(errors))


def _validate_paper(paper: NewsletterPaper, index: int) -> list[str]:
    errors: list[str] = []
    metadata = paper.metadata
    summary = paper.summary

    if not metadata.title.strip():
        errors.append(f"Paper {index} is missing a title.")
    if not metadata.authors:
        errors.append(f"Paper {index} is missing authors.")
    if not metadata.arxiv_url.startswith("https://arxiv.org/abs/"):
        errors.append(f"Paper {index} is missing a valid arXiv link.")

    for field_name in [
        "executive_summary",
        "problem",
        "method",
        "why_it_matters",
        "limitations",
    ]:
        if not getattr(summary, field_name).strip():
            errors.append(f"Paper {index} has an empty {field_name} summary.")

    return errors

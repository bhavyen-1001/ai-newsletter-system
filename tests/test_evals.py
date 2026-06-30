from newsletter.evals import validate_issue
from newsletter.models import NewsletterIssue, NewsletterPaper, PaperMetadata, PaperSummary


def _paper() -> NewsletterPaper:
    return NewsletterPaper(
        metadata=PaperMetadata(
            arxiv_id="2606.12345",
            title="A Useful AI Paper",
            authors=["Ada Lovelace", "Alan Turing"],
            arxiv_url="https://arxiv.org/abs/2606.12345",
            pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
            hf_url="https://huggingface.co/papers/2606.12345",
        ),
        summary=PaperSummary(
            executive_summary="A concise summary.",
            problem="A clear problem.",
            method="A clear method.",
            why_it_matters="A clear reason.",
            limitations="A clear limitation.",
        ),
        source_week="2026-W26",
    )


def test_validate_issue_accepts_complete_issue():
    issue = NewsletterIssue(week="2026-W26", generated_at="now", papers=[_paper()])

    assert validate_issue(issue) == []


def test_validate_issue_rejects_missing_authors():
    paper = _paper()
    paper = NewsletterPaper(
        metadata=PaperMetadata(
            arxiv_id=paper.metadata.arxiv_id,
            title=paper.metadata.title,
            authors=[],
            arxiv_url=paper.metadata.arxiv_url,
            pdf_url=paper.metadata.pdf_url,
            hf_url=paper.metadata.hf_url,
        ),
        summary=paper.summary,
        source_week=paper.source_week,
    )
    issue = NewsletterIssue(week="2026-W26", generated_at="now", papers=[paper])

    assert "Paper 1 is missing authors." in validate_issue(issue)

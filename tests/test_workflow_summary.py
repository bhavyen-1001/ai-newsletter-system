from newsletter.models import NewsletterIssue, NewsletterPaper, PaperMetadata, PaperSummary
from newsletter.workflow import _summary_markdown


def _issue() -> NewsletterIssue:
    paper = NewsletterPaper(
        metadata=PaperMetadata(
            arxiv_id="2606.12345",
            title="A Useful AI Paper",
            authors=["Ada Lovelace"],
            arxiv_url="https://arxiv.org/abs/2606.12345",
            pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
            hf_url="https://huggingface.co/papers/2606.12345",
        ),
        summary=PaperSummary(
            executive_summary="Summary",
            problem="Problem",
            method="Method",
            why_it_matters="Matters",
            limitations="Limits",
        ),
        source_week="2026-W26",
    )
    return NewsletterIssue(week="2026-W26", generated_at="now", papers=[paper])


def test_workflow_summary_includes_generated_paper_summary():
    summary = _summary_markdown(
        issue=_issue(),
        dry_run=True,
        campaign_id=None,
        skipped=[],
    )

    assert "### 1. A Useful AI Paper" in summary
    assert "**Executive summary:** Summary" in summary
    assert "**Method:** Method" in summary

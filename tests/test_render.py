from newsletter.models import NewsletterIssue, NewsletterPaper, PaperMetadata, PaperSummary
from newsletter.render import render_html, render_text


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


def test_render_html_contains_intro_anchor_and_unsubscribe():
    html = render_html(_issue(), subject_prefix="AI Research Weekly")

    assert 'href="#paper-2606-12345"' in html
    assert "*|UNSUB|*" in html


def test_render_text_contains_plain_text_fallback():
    text = render_text(_issue(), subject_prefix="AI Research Weekly")

    assert "Executive summary: Summary" in text
    assert "Unsubscribe: *|UNSUB|*" in text

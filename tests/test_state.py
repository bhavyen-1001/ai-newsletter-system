from newsletter.models import NewsletterIssue, NewsletterPaper, PaperMetadata, PaperSummary
from newsletter.state import SentPaperState


def test_state_marks_issue_sent(tmp_path):
    state_path = tmp_path / "sent_papers.json"
    state = SentPaperState.load(state_path)
    issue = NewsletterIssue(
        week="2026-W26",
        generated_at="now",
        papers=[
            NewsletterPaper(
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
        ],
    )

    state.mark_issue_sent(issue, campaign_id="abc123")
    state.save()

    loaded = SentPaperState.load(state_path)
    assert loaded.is_sent("2606.12345")
    assert loaded.data["sent_papers"][0]["mailchimp_campaign_id"] == "abc123"

import json

import pytest

import newsletter.workflow as workflow_module
from newsletter.config import Settings
from newsletter.models import PaperCandidate, PaperMetadata


def _settings(tmp_path) -> Settings:
    return Settings(
        hf_token=None,
        hf_model_id="test-model",
        hf_provider="auto",
        mailchimp_api_key="mailchimp-key",
        mailchimp_server_prefix="us1",
        mailchimp_audience_id="audience-id",
        mailchimp_from_name="AI Research Weekly",
        mailchimp_reply_to="reply@example.com",
        admin_email="admin@example.com",
        subject_prefix="AI Research Weekly",
        signup_url=None,
        state_path=tmp_path / "sent_papers.json",
        output_dir=tmp_path / "outputs",
        target_count=1,
        pdf_text_max_chars=1000,
        min_pdf_text_chars=10,
    )


def _patch_paper_fetching(monkeypatch) -> None:
    candidate = PaperCandidate(
        arxiv_id="2606.12345",
        hf_url="https://huggingface.co/papers/2606.12345",
    )
    metadata = PaperMetadata(
        arxiv_id="2606.12345",
        title="A Useful AI Paper",
        authors=["Ada Lovelace"],
        arxiv_url="https://arxiv.org/abs/2606.12345",
        pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
        hf_url="https://huggingface.co/papers/2606.12345",
    )

    monkeypatch.setattr(workflow_module, "fetch_weekly_trending_papers", lambda week: [candidate])
    monkeypatch.setattr(workflow_module, "fetch_arxiv_metadata", lambda candidate: metadata)
    monkeypatch.setattr(workflow_module, "download_pdf", lambda pdf_url: b"pdf")
    monkeypatch.setattr(workflow_module, "extract_pdf_text", lambda pdf_bytes, max_chars: "paper text " * 20)


def test_test_only_sends_mailchimp_test_without_live_send_or_state_update(tmp_path, monkeypatch):
    _patch_paper_fetching(monkeypatch)
    calls = []

    class FakeMailchimpProvider:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def create_campaign(self, *, title: str, subject: str) -> str:
            calls.append(("create_campaign", title, subject))
            return "campaign-123"

        def set_campaign_content(self, *, campaign_id: str, html: str, text: str) -> None:
            calls.append(("set_campaign_content", campaign_id, html, text))

        def send_test(self, *, campaign_id: str, test_email: str) -> None:
            calls.append(("send_test", campaign_id, test_email))

        def send_campaign(self, *, campaign_id: str) -> None:
            calls.append(("send_campaign", campaign_id))

    monkeypatch.setattr(workflow_module, "MailchimpProvider", FakeMailchimpProvider)

    result = workflow_module.NewsletterWorkflow(_settings(tmp_path)).run(
        week="2026-W26",
        dry_run=False,
        test_only=True,
        mock_llm=True,
    )

    call_names = [call[0] for call in calls]
    assert result.campaign_id == "campaign-123"
    assert result.test_only is True
    assert ("send_test", "campaign-123", "admin@example.com") in calls
    assert "send_campaign" not in call_names
    assert not (tmp_path / "sent_papers.json").exists()

    payload = json.loads((tmp_path / "outputs" / "2026-W26.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "test_only"
    assert payload["campaign_id"] == "campaign-123"
    assert payload["test_only"] is True


def test_send_mode_sends_campaign_and_updates_state(tmp_path, monkeypatch):
    _patch_paper_fetching(monkeypatch)
    calls = []

    class FakeMailchimpProvider:
        def __init__(self, **kwargs):
            pass

        def create_campaign(self, *, title: str, subject: str) -> str:
            return "campaign-456"

        def set_campaign_content(self, *, campaign_id: str, html: str, text: str) -> None:
            pass

        def send_test(self, *, campaign_id: str, test_email: str) -> None:
            calls.append(("send_test", campaign_id, test_email))

        def send_campaign(self, *, campaign_id: str) -> None:
            calls.append(("send_campaign", campaign_id))

    monkeypatch.setattr(workflow_module, "MailchimpProvider", FakeMailchimpProvider)

    result = workflow_module.NewsletterWorkflow(_settings(tmp_path)).run(
        week="2026-W26",
        dry_run=False,
        test_only=False,
        mock_llm=True,
    )

    assert result.campaign_id == "campaign-456"
    assert ("send_test", "campaign-456", "admin@example.com") in calls
    assert ("send_campaign", "campaign-456") in calls

    state = json.loads((tmp_path / "sent_papers.json").read_text(encoding="utf-8"))
    assert state["sent_papers"][0]["arxiv_id"] == "2606.12345"
    assert state["sent_papers"][0]["mailchimp_campaign_id"] == "campaign-456"


def test_workflow_reports_empty_candidate_week(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_module, "fetch_weekly_trending_papers", lambda week: [])

    with pytest.raises(ValueError) as exc_info:
        workflow_module.NewsletterWorkflow(_settings(tmp_path)).run(
            week="2026-W25",
            dry_run=True,
            mock_llm=True,
        )

    message = str(exc_info.value)
    assert "Newsletter selection failed: no papers were selected." in message
    assert "Week: 2026-W25" in message
    assert "Candidates fetched: 0" in message
    assert "https://huggingface.co/papers/week/2026-W25" in message


def test_workflow_reports_skip_reasons_when_all_candidates_fail(tmp_path, monkeypatch):
    candidates = [
        PaperCandidate(arxiv_id="2606.11111", hf_url="https://huggingface.co/papers/2606.11111"),
        PaperCandidate(arxiv_id="2606.22222", hf_url="https://huggingface.co/papers/2606.22222"),
    ]
    monkeypatch.setattr(workflow_module, "fetch_weekly_trending_papers", lambda week: candidates)

    def fail_metadata(candidate: PaperCandidate) -> PaperMetadata:
        raise RuntimeError(f"metadata failed for {candidate.arxiv_id}")

    monkeypatch.setattr(workflow_module, "fetch_arxiv_metadata", fail_metadata)

    with pytest.raises(ValueError) as exc_info:
        workflow_module.NewsletterWorkflow(_settings(tmp_path)).run(
            week="2026-W25",
            dry_run=True,
            mock_llm=True,
        )

    message = str(exc_info.value)
    assert "Candidates fetched: 2" in message
    assert "First skipped candidates:" in message
    assert "2606.11111: metadata failed for 2606.11111" in message
    assert "2606.22222: metadata failed for 2606.22222" in message

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from newsletter.arxiv import download_pdf, extract_pdf_text, fetch_arxiv_metadata
from newsletter.config import Settings
from newsletter.dates import current_hf_week
from newsletter.evals import raise_if_invalid
from newsletter.hf_papers import fetch_weekly_trending_papers
from newsletter.llm import HuggingFaceSummariser
from newsletter.models import NewsletterIssue, NewsletterPaper, dataclass_to_dict
from newsletter.providers.mailchimp import MailchimpProvider
from newsletter.render import render_html, render_text
from newsletter.state import SentPaperState


@dataclass(frozen=True)
class WorkflowResult:
    week: str
    dry_run: bool
    campaign_id: str | None
    selected_count: int
    skipped: list[str]
    output_dir: Path


class NewsletterWorkflow:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, *, week: str | None, dry_run: bool, mock_llm: bool) -> WorkflowResult:
        self.settings.require_llm_config(mock_llm=mock_llm)
        if not dry_run:
            self.settings.require_send_config()

        target_week = week or current_hf_week()
        state = SentPaperState.load(self.settings.state_path)
        summariser = HuggingFaceSummariser(
            token=self.settings.hf_token,
            model_id=self.settings.hf_model_id,
            provider=self.settings.hf_provider,
            max_input_chars=self.settings.pdf_text_max_chars,
            mock=mock_llm,
        )

        selected: list[NewsletterPaper] = []
        skipped: list[str] = []
        candidates = fetch_weekly_trending_papers(target_week)

        for candidate in candidates:
            if state.is_sent(candidate.arxiv_id):
                skipped.append(f"{candidate.arxiv_id}: already sent")
                continue

            try:
                metadata = fetch_arxiv_metadata(candidate)
                pdf_bytes = download_pdf(metadata.pdf_url)
                pdf_text = extract_pdf_text(pdf_bytes, max_chars=self.settings.pdf_text_max_chars)
                if len(pdf_text) < self.settings.min_pdf_text_chars:
                    raise RuntimeError("extracted PDF text was too short")
                summary = summariser.summarise(metadata, pdf_text)
                selected.append(NewsletterPaper(metadata=metadata, summary=summary, source_week=target_week))
            except Exception as exc:
                skipped.append(f"{candidate.arxiv_id}: {exc}")

            if len(selected) >= self.settings.target_count:
                break

        issue = NewsletterIssue(
            week=target_week,
            generated_at=datetime.now(tz=UTC).isoformat(),
            papers=selected,
        )
        raise_if_invalid(issue)

        html = render_html(
            issue,
            subject_prefix=self.settings.subject_prefix,
            signup_url=self.settings.signup_url,
        )
        text = render_text(
            issue,
            subject_prefix=self.settings.subject_prefix,
            signup_url=self.settings.signup_url,
        )

        campaign_id: str | None = None
        self._write_outputs(issue=issue, html=html, text=text, skipped=skipped, dry_run=dry_run, campaign_id=None)

        if not dry_run:
            provider = MailchimpProvider(
                api_key=self.settings.mailchimp_api_key or "",
                server_prefix=self.settings.mailchimp_server_prefix or "",
                audience_id=self.settings.mailchimp_audience_id or "",
                from_name=self.settings.mailchimp_from_name or "",
                reply_to=self.settings.mailchimp_reply_to or "",
            )
            campaign_title = f"{self.settings.subject_prefix} - {target_week}"
            campaign_id = provider.create_campaign(title=campaign_title, subject=campaign_title)
            provider.set_campaign_content(campaign_id=campaign_id, html=html, text=text)
            provider.send_test(campaign_id=campaign_id, test_email=self.settings.admin_email or "")
            provider.send_campaign(campaign_id=campaign_id)

            state.mark_issue_sent(issue, campaign_id=campaign_id)
            state.save()
            self._write_outputs(
                issue=issue,
                html=html,
                text=text,
                skipped=skipped,
                dry_run=dry_run,
                campaign_id=campaign_id,
            )

        return WorkflowResult(
            week=target_week,
            dry_run=dry_run,
            campaign_id=campaign_id,
            selected_count=len(selected),
            skipped=skipped,
            output_dir=self.settings.output_dir,
        )

    def _write_outputs(
        self,
        *,
        issue: NewsletterIssue,
        html: str,
        text: str,
        skipped: list[str],
        dry_run: bool,
        campaign_id: str | None,
    ) -> None:
        output_dir = self.settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = issue.week

        (output_dir / f"{stem}.html").write_text(html, encoding="utf-8")
        (output_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        payload: dict[str, Any] = {
            "week": issue.week,
            "generated_at": issue.generated_at,
            "dry_run": dry_run,
            "campaign_id": campaign_id,
            "papers": [dataclass_to_dict(paper) for paper in issue.papers],
            "skipped": skipped,
        }
        (output_dir / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "latest_summary.md").write_text(
            _summary_markdown(issue=issue, dry_run=dry_run, campaign_id=campaign_id, skipped=skipped),
            encoding="utf-8",
        )


def _summary_markdown(
    *,
    issue: NewsletterIssue,
    dry_run: bool,
    campaign_id: str | None,
    skipped: list[str],
) -> str:
    mode = "dry run" if dry_run else "sent"
    lines = [
        f"# AI Research Weekly - {issue.week}",
        "",
        f"Mode: {mode}",
        f"Papers selected: {len(issue.papers)}",
    ]
    if campaign_id:
        lines.append(f"Mailchimp campaign ID: {campaign_id}")
    lines.extend(["", "## Papers"])
    for paper in issue.papers:
        lines.append(f"- {paper.metadata.arxiv_id}: {paper.metadata.title}")
    if skipped:
        lines.extend(["", "## Skipped"])
        lines.extend(f"- {item}" for item in skipped[:20])
    lines.append("")
    return "\n".join(lines)

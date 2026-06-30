from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    hf_token: str | None
    hf_model_id: str
    hf_provider: str
    mailchimp_api_key: str | None
    mailchimp_server_prefix: str | None
    mailchimp_audience_id: str | None
    mailchimp_from_name: str | None
    mailchimp_reply_to: str | None
    admin_email: str | None
    subject_prefix: str
    signup_url: str | None
    state_path: Path
    output_dir: Path
    target_count: int
    pdf_text_max_chars: int
    min_pdf_text_chars: int

    @classmethod
    def from_env(
        cls,
        *,
        state_path: str | Path = "data/sent_papers.json",
        output_dir: str | Path = "outputs",
        target_count: int = 3,
    ) -> "Settings":
        return cls(
            hf_token=_blank_to_none(os.getenv("HF_TOKEN")),
            hf_model_id=_blank_to_none(os.getenv("HF_MODEL_ID")) or "Qwen/Qwen2.5-32B-Instruct",
            hf_provider=_blank_to_none(os.getenv("HF_PROVIDER")) or "auto",
            mailchimp_api_key=_blank_to_none(os.getenv("MAILCHIMP_API_KEY")),
            mailchimp_server_prefix=_blank_to_none(os.getenv("MAILCHIMP_SERVER_PREFIX")),
            mailchimp_audience_id=_blank_to_none(os.getenv("MAILCHIMP_AUDIENCE_ID")),
            mailchimp_from_name=_blank_to_none(os.getenv("MAILCHIMP_FROM_NAME")),
            mailchimp_reply_to=_blank_to_none(os.getenv("MAILCHIMP_REPLY_TO")),
            admin_email=_blank_to_none(os.getenv("ADMIN_EMAIL")),
            subject_prefix=_blank_to_none(os.getenv("NEWSLETTER_SUBJECT_PREFIX")) or "AI Research Weekly",
            signup_url=_blank_to_none(os.getenv("NEWSLETTER_SIGNUP_URL")),
            state_path=Path(state_path),
            output_dir=Path(output_dir),
            target_count=target_count,
            pdf_text_max_chars=_int_env("PDF_TEXT_MAX_CHARS", 55000),
            min_pdf_text_chars=_int_env("MIN_PDF_TEXT_CHARS", 2500),
        )

    def require_send_config(self) -> None:
        missing = [
            name
            for name, value in {
                "MAILCHIMP_API_KEY": self.mailchimp_api_key,
                "MAILCHIMP_SERVER_PREFIX": self.mailchimp_server_prefix,
                "MAILCHIMP_AUDIENCE_ID": self.mailchimp_audience_id,
                "MAILCHIMP_FROM_NAME": self.mailchimp_from_name,
                "MAILCHIMP_REPLY_TO": self.mailchimp_reply_to,
                "ADMIN_EMAIL": self.admin_email,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required send configuration: {', '.join(missing)}")

    def require_llm_config(self, *, mock_llm: bool) -> None:
        if not mock_llm and not self.hf_token:
            raise RuntimeError("HF_TOKEN is required unless --mock-llm is used.")


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int_env(name: str, default: int) -> int:
    value = _blank_to_none(os.getenv(name))
    return int(value) if value is not None else default

from __future__ import annotations

import json
import re

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

from newsletter.models import PaperMetadata, PaperSummary

SUMMARY_KEYS = [
    "executive_summary",
    "problem",
    "method",
    "why_it_matters",
    "limitations",
]


class HuggingFaceSummariser:
    def __init__(
        self,
        *,
        token: str | None,
        model_id: str,
        provider: str,
        max_input_chars: int,
        mock: bool = False,
    ) -> None:
        self.model_id = model_id
        self.max_input_chars = max_input_chars
        self.mock = mock
        self.client = None
        if not mock:
            self.client = InferenceClient(model=model_id, provider=provider, token=token, timeout=120)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=12))
    def summarise(self, metadata: PaperMetadata, pdf_text: str) -> PaperSummary:
        if self.mock:
            return PaperSummary(
                executive_summary=f"{metadata.title} is a mock summary generated for a dry run.",
                problem="The paper studies an AI research problem described in the extracted PDF text.",
                method="The method is summarised here by the mock summariser for local testing.",
                why_it_matters="It matters because the work may influence future AI systems or research practice.",
                limitations="Limitations are not assessed in mock mode.",
            )

        if self.client is None:
            raise RuntimeError("Hugging Face client is not configured.")

        prompt = _build_prompt(metadata, pdf_text[: self.max_input_chars])
        response = self.client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarise AI research for a mixed audience of senior engineers "
                        "and technical startup operators. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0.2,
        )
        content = _extract_content(response)
        return _parse_summary(content)


def _build_prompt(metadata: PaperMetadata, pdf_text: str) -> str:
    return f"""
Summarise the research paper below using only the extracted PDF text. Do not use outside knowledge.

Title: {metadata.title}
Authors: {", ".join(metadata.authors)}
arXiv ID: {metadata.arxiv_id}

Return JSON with exactly these string fields:
- executive_summary: 2-3 concise sentences.
- problem: the research problem in 1-2 sentences.
- method: the core method or approach in 2-3 sentences.
- why_it_matters: practical or research significance in 1-2 sentences.
- limitations: stated or likely limitations from the PDF text in 1-2 sentences.

Extracted PDF text:
{pdf_text}
""".strip()


def _extract_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        if isinstance(message, dict):
            return str(message.get("content", ""))
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
    if isinstance(response, dict):
        return str(response["choices"][0]["message"]["content"])
    return str(response)


def _parse_summary(raw: str) -> PaperSummary:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    data = json.loads(text)
    missing = [key for key in SUMMARY_KEYS if not str(data.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Summary is missing required fields: {', '.join(missing)}")

    return PaperSummary(**{key: str(data[key]).strip() for key in SUMMARY_KEYS})

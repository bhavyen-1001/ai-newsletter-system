from __future__ import annotations

import json
import re
from dataclasses import dataclass

from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from newsletter.authors import format_authors
from newsletter.models import PaperMetadata, PaperSummary

SUMMARY_KEYS = [
    "executive_summary",
    "problem",
    "method",
    "why_it_matters",
    "limitations",
]


@dataclass(frozen=True)
class ModelEndpoint:
    model_id: str
    provider: str


class HuggingFaceSummariser:
    def __init__(
        self,
        *,
        token: str | None,
        model_id: str,
        provider: str,
        max_input_chars: int,
        fallback_model_id: str | None = None,
        fallback_provider: str | None = None,
        mock: bool = False,
    ) -> None:
        self.model_id = model_id
        self.provider = provider
        self.max_input_chars = max_input_chars
        self.mock = mock
        self.clients: list[tuple[ModelEndpoint, InferenceClient]] = []
        if not mock:
            endpoints = _model_endpoints(
                model_id=model_id,
                provider=provider,
                fallback_model_id=fallback_model_id,
                fallback_provider=fallback_provider,
            )
            self.clients = [
                (endpoint, InferenceClient(model=endpoint.model_id, provider=endpoint.provider, token=token, timeout=120))
                for endpoint in endpoints
            ]

    def summarise(self, metadata: PaperMetadata, pdf_text: str) -> PaperSummary:
        if self.mock:
            return PaperSummary(
                executive_summary=f"{metadata.title} is a mock summary generated for a dry run.",
                problem="The paper studies an AI research problem described in the extracted PDF text.",
                method="The method is summarised here by the mock summariser for local testing.",
                why_it_matters="It matters because the work may influence future AI systems or research practice.",
                limitations="Limitations are not assessed in mock mode.",
            )

        if not self.clients:
            raise RuntimeError("Hugging Face client is not configured.")

        prompt = _build_prompt(metadata, pdf_text[: self.max_input_chars])
        messages = [
            {
                "role": "system",
                "content": (
                    "You summarise AI research for a mixed audience of senior engineers "
                    "and technical startup operators. Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        failures: list[str] = []

        for endpoint, client in self.clients:
            try:
                response = self._chat_completion(client=client, messages=messages)
                content = _extract_content(response)
                return _parse_summary(content)
            except Exception as exc:
                failures.append(_failure_message(endpoint, exc))

        raise RuntimeError("All Hugging Face summarisation models failed: " + " | ".join(failures))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=12),
        retry=retry_if_not_exception_type(BadRequestError),
        reraise=True,
    )
    def _chat_completion(self, *, client: InferenceClient, messages: list[dict[str, str]]) -> object:
        return client.chat_completion(
            messages=messages,
            max_tokens=1200,
            temperature=0.2,
        )


def _model_endpoints(
    *,
    model_id: str,
    provider: str,
    fallback_model_id: str | None,
    fallback_provider: str | None,
) -> list[ModelEndpoint]:
    endpoints = [ModelEndpoint(model_id=model_id, provider=provider)]
    if fallback_model_id:
        fallback = ModelEndpoint(model_id=fallback_model_id, provider=fallback_provider or provider)
        if fallback not in endpoints:
            endpoints.append(fallback)
    return endpoints


def _failure_message(endpoint: ModelEndpoint, exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"{endpoint.model_id} via {endpoint.provider}: {exc.__class__.__name__}: {message}"


def _build_prompt(metadata: PaperMetadata, pdf_text: str) -> str:
    return f"""
Summarise the research paper below using only the extracted PDF text. Do not use outside knowledge.

Title: {metadata.title}
Authors: {format_authors(metadata.authors)}
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

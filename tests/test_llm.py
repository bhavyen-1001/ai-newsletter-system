from __future__ import annotations

from httpx import Request, Response
from huggingface_hub.errors import BadRequestError

import newsletter.llm as llm_module
from newsletter.llm import HuggingFaceSummariser, _build_prompt
from newsletter.models import PaperMetadata


def _metadata() -> PaperMetadata:
    return PaperMetadata(
        arxiv_id="2606.12345",
        title="A Useful AI Paper",
        authors=["Ada Lovelace", "Alan Turing", "Grace Hopper", "Katherine Johnson"],
        arxiv_url="https://arxiv.org/abs/2606.12345",
        pdf_url="https://arxiv.org/pdf/2606.12345.pdf",
        hf_url="https://huggingface.co/papers/2606.12345",
    )


def _summary_json(label: str) -> str:
    return f"""{{
        "executive_summary": "{label} summary.",
        "problem": "{label} problem.",
        "method": "{label} method.",
        "why_it_matters": "{label} reason.",
        "limitations": "{label} limitation."
    }}"""


def _bad_request() -> BadRequestError:
    request = Request("POST", "https://router.huggingface.co/v1/chat/completions")
    response = Response(400, request=request)
    return BadRequestError("bad request", response=response)


def test_summariser_falls_back_after_primary_bad_request(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakeInferenceClient:
        def __init__(self, *, model: str, provider: str, token: str | None, timeout: int) -> None:
            self.model = model
            self.provider = provider

        def chat_completion(self, **kwargs):
            calls.append((self.model, self.provider))
            if self.model == "primary-model":
                raise _bad_request()
            return {"choices": [{"message": {"content": _summary_json("fallback")}}]}

    monkeypatch.setattr(llm_module, "InferenceClient", FakeInferenceClient)

    summary = HuggingFaceSummariser(
        token="token",
        model_id="primary-model",
        provider="primary-provider",
        fallback_model_id="fallback-model",
        fallback_provider="fallback-provider",
        max_input_chars=1000,
    ).summarise(_metadata(), "paper text")

    assert summary.executive_summary == "fallback summary."
    assert calls == [
        ("primary-model", "primary-provider"),
        ("fallback-model", "fallback-provider"),
    ]


def test_summariser_falls_back_after_primary_invalid_json(monkeypatch):
    calls: list[str] = []

    class FakeInferenceClient:
        def __init__(self, *, model: str, provider: str, token: str | None, timeout: int) -> None:
            self.model = model

        def chat_completion(self, **kwargs):
            calls.append(self.model)
            if self.model == "primary-model":
                return {"choices": [{"message": {"content": "not-json"}}]}
            return {"choices": [{"message": {"content": _summary_json("fallback")}}]}

    monkeypatch.setattr(llm_module, "InferenceClient", FakeInferenceClient)

    summary = HuggingFaceSummariser(
        token="token",
        model_id="primary-model",
        provider="primary-provider",
        fallback_model_id="fallback-model",
        fallback_provider="fallback-provider",
        max_input_chars=1000,
    ).summarise(_metadata(), "paper text")

    assert summary.method == "fallback method."
    assert calls == ["primary-model", "fallback-model"]


def test_build_prompt_uses_compact_authors():
    prompt = _build_prompt(_metadata(), "paper text")

    assert "Authors: Ada Lovelace et al." in prompt
    assert "Alan Turing" not in prompt

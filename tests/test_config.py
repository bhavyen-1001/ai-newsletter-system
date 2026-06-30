from newsletter.config import Settings


def test_blank_optional_env_uses_defaults(monkeypatch):
    monkeypatch.setenv("HF_MODEL_ID", "")
    monkeypatch.setenv("HF_PROVIDER", "")
    monkeypatch.setenv("NEWSLETTER_SUBJECT_PREFIX", "")
    monkeypatch.setenv("PDF_TEXT_MAX_CHARS", "")

    settings = Settings.from_env()

    assert settings.hf_model_id == "Qwen/Qwen2.5-32B-Instruct"
    assert settings.hf_provider == "auto"
    assert settings.subject_prefix == "AI Research Weekly"
    assert settings.pdf_text_max_chars == 55000

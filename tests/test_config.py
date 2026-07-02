from newsletter.config import Settings


def test_blank_optional_env_uses_defaults(monkeypatch):
    monkeypatch.setenv("HF_MODEL_ID", "")
    monkeypatch.setenv("HF_PROVIDER", "")
    monkeypatch.setenv("HF_FALLBACK_MODEL_ID", "")
    monkeypatch.setenv("HF_FALLBACK_PROVIDER", "")
    monkeypatch.setenv("NEWSLETTER_SUBJECT_PREFIX", "")
    monkeypatch.setenv("PDF_TEXT_MAX_CHARS", "")

    settings = Settings.from_env()

    assert settings.hf_model_id == "google/gemma-3-27b-it"
    assert settings.hf_provider == "auto"
    assert settings.hf_fallback_model_id == "CohereLabs/aya-expanse-32b"
    assert settings.hf_fallback_provider == "auto"
    assert settings.subject_prefix == "AI Research Weekly"
    assert settings.pdf_text_max_chars == 55000


def test_hf_model_alias_is_supported(monkeypatch):
    monkeypatch.delenv("HF_MODEL_ID", raising=False)
    monkeypatch.setenv("HF_MODEL", "CohereLabs/aya-expanse-32b")

    settings = Settings.from_env()

    assert settings.hf_model_id == "CohereLabs/aya-expanse-32b"

import pytest

import newsletter.arxiv as arxiv_module
from newsletter.arxiv import extract_pdf_text


class FakeTools:
    def __init__(self, *, errors: bool = True, warnings: bool = True) -> None:
        self.errors = errors
        self.warnings = warnings
        self.calls: list[tuple[str, object]] = []

    def mupdf_display_errors(self, on=None):
        if on is None:
            return self.errors
        self.calls.append(("errors", on))
        self.errors = on
        return on

    def mupdf_display_warnings(self, on=None):
        if on is None:
            return self.warnings
        self.calls.append(("warnings", on))
        self.warnings = on
        return on


class FakeDocument:
    def __init__(self, pages) -> None:
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(self.pages)


class FakePage:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_text(self, mode: str) -> str:
        assert mode == "text"
        return self.text


class FailingPage:
    def get_text(self, mode: str) -> str:
        raise RuntimeError("extract failed")


def test_extract_pdf_text_suppresses_and_restores_mupdf_diagnostics(monkeypatch):
    tools = FakeTools(errors=True, warnings=True)
    monkeypatch.setattr(arxiv_module.fitz, "TOOLS", tools)
    monkeypatch.setattr(
        arxiv_module.fitz,
        "open",
        lambda **kwargs: FakeDocument([FakePage(" first line \n"), FakePage("second line")]),
    )

    text = extract_pdf_text(b"%PDF", max_chars=100)

    assert text == "first line\nsecond line"
    assert tools.errors is True
    assert tools.warnings is True
    assert tools.calls == [
        ("errors", False),
        ("warnings", False),
        ("warnings", True),
        ("errors", True),
    ]


def test_extract_pdf_text_restores_mupdf_diagnostics_after_failure(monkeypatch):
    tools = FakeTools(errors=True, warnings=False)
    monkeypatch.setattr(arxiv_module.fitz, "TOOLS", tools)
    monkeypatch.setattr(arxiv_module.fitz, "open", lambda **kwargs: FakeDocument([FailingPage()]))

    with pytest.raises(RuntimeError, match="extract failed"):
        extract_pdf_text(b"%PDF", max_chars=100)

    assert tools.errors is True
    assert tools.warnings is False

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import fitz
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from newsletter.hf_papers import normalise_arxiv_id
from newsletter.models import PaperCandidate, PaperMetadata

ARXIV_API_URL = "https://export.arxiv.org/api/query"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_arxiv_metadata(candidate: PaperCandidate) -> PaperMetadata:
    arxiv_id = normalise_arxiv_id(candidate.arxiv_id)
    response = requests.get(
        ARXIV_API_URL,
        params={"id_list": arxiv_id},
        headers={"User-Agent": "ai-research-newsletter/0.1"},
        timeout=30,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise RuntimeError(f"No arXiv metadata found for {arxiv_id}")

    title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
    authors = [
        _clean_text(author.findtext("atom:name", default="", namespaces=ns))
        for author in entry.findall("atom:author", ns)
    ]
    authors = [author for author in authors if author]

    return PaperMetadata(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        hf_url=candidate.hf_url,
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
def download_pdf(pdf_url: str) -> bytes:
    response = requests.get(
        pdf_url,
        headers={"User-Agent": "ai-research-newsletter/0.1"},
        timeout=60,
    )
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError(f"Downloaded content is not a PDF: {pdf_url}")
    return response.content


def extract_pdf_text(pdf_bytes: bytes, *, max_chars: int) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        chunks: list[str] = []
        total = 0
        for page in document:
            text = page.get_text("text")
            if not text:
                continue
            chunks.append(text)
            total += len(text)
            if total >= max_chars:
                break

    cleaned = _clean_pdf_text("\n".join(chunks))
    return cleaned[:max_chars]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_pdf_text(value: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)

from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from newsletter.models import PaperCandidate

HF_BASE_URL = "https://huggingface.co"
ARXIV_ID_PATTERN = re.compile(r"/papers/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)")


def normalise_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v[0-9]+$", "", arxiv_id.strip())


def parse_trending_paper_ids(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    ids: list[str] = []

    for anchor in soup.find_all("a", href=True):
        match = ARXIV_ID_PATTERN.search(anchor["href"])
        if not match:
            continue
        arxiv_id = normalise_arxiv_id(match.group(1))
        if arxiv_id not in seen:
            seen.add(arxiv_id)
            ids.append(arxiv_id)

    return ids


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_weekly_trending_papers(week: str) -> list[PaperCandidate]:
    url = f"{HF_BASE_URL}/papers/week/{week}"
    response = requests.get(url, headers={"User-Agent": "ai-research-newsletter/0.1"}, timeout=30)
    response.raise_for_status()

    paper_ids = parse_trending_paper_ids(response.text)
    return [
        PaperCandidate(arxiv_id=paper_id, hf_url=f"{HF_BASE_URL}/papers/{paper_id}")
        for paper_id in paper_ids
    ]

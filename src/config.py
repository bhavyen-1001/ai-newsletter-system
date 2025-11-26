"""Configuration settings for the AI newsletter system."""

import os
from datetime import datetime


# Base URLs
HUGGINGFACE_BASE_URL = "https://huggingface.co/papers/week"
ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf"

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")

# Scraping settings
TOP_N_PAPERS = 5
REQUEST_TIMEOUT = 30  # seconds


def get_current_week_url():
    """Generates the HuggingFace URL for the current week.

    Returns:
        str: The HuggingFace weekly papers URL for current week.

    Example:
        For week 47 of 2025, returns:
        'https://huggingface.co/papers/week/2025-W47'
    """
    now = datetime.now()
    year = now.year
    # ISO week format: %V gives week number (01-53)
    week = now.strftime("%V")
    return f"{HUGGINGFACE_BASE_URL}/{year}-W{week}"


def get_week_folder_name():
    """Generates the folder name for storing current week's papers.

    Returns:
        str: Folder name in format 'YYYY-WW'.

    Example:
        For week 47 of 2025, returns: '2025-47'
    """
    now = datetime.now()
    year = now.year
    week = now.strftime("%V")
    return f"{year}-{week}"


def ensure_directories_exist():
    """Creates necessary directories if they don't exist."""
    os.makedirs(PAPERS_DIR, exist_ok=True)


# LLM settings
VERTEX_AI_LOCATION = "europe-west2"  # Change if needed based on model availability
CHUNK_SIZE = 20000  # tokens
CHUNK_OVERLAP = 500  # tokens overlap between chunks

# Model names for Vertex AI Model Garden
# GEMMA_MODEL = "google/gemma-3-4b-it"
# GPT_OSS_MODEL = "openai/gpt-oss-20b"

GEMMA_MODEL = "gemini-1.5-flash-002"
GPT_OSS_MODEL = "gemini-1.5-flash-002"

# Prompts
# -----------------------------------------------------------------------------
# MAP PROMPT: Content-Agnostic Extraction
# Designed to be robust against varying section names (e.g. "Results" vs "Analysis").
# We ask the model to categorize the segment itself before extracting.
MAP_PROMPT = """You are an expert AI Researcher analyzing a specific segment of a research paper. 
Your goal is to extract high-value technical information for a deep-dive newsletter.

<instructions>
1. Analyze Content Type: Determine if this segment discusses the Problem, Methodology/Architecture, Experimental Results, or Related Work. (Note: Papers use varied section headers; rely on the actual text content).
2. Extract Technical Specs: Look for architecture dimensions (parameter counts, layers), training details (batch size, hardware), and algorithms.
3. Extract Hard Metrics: If this segment contains results, extract specific numbers (e.g., "85.2% on MMLU", "latency reduction of 40ms"). Do not summarize vaguely; copy the exact numbers.
4. Ignore Fluff: Disregard general introductory text or broad claims of impact unless backed by data in this segment.
5. Output: A structured list of distinct facts found ONLY in this segment.
</instructions>

<paper_segment>
{text}
</paper_segment>

**Extracted Technical Notes:**
"""

# REDUCE PROMPT: Deep Technical Synthesis
# Enforces a strict Markdown schema and a high-density technical tone.
REDUCE_PROMPT = """You are the Lead Technical Editor for a specialized AI research newsletter read by senior engineers and researchers.
Your goal is to synthesize the extracted notes from a full paper into a single, dense technical summary.

<audience_profile>
- The readers are deep technical experts.
- They value specific architectural details, hyperparameters, and exact benchmark scores.
- They dislike buzzwords ("revolutionary", "unprecedented") and vague summaries ("performed well").
</audience_profile>

<output_schema>
Produce a summary in strict Markdown using exactly these headers:

# [Paper Title]

**TL;DR:** (1 bold sentence explaining the core technical innovation)

### The Problem
(2 sentences on the specific technical bottleneck or gap this paper addresses)

### Methodology & Architecture
(Deep dive into *how* it works. Mention specific mechanism names, loss functions, architectural changes, or training recipes. Use bullet points if describing a multi-step process.)

### Key Results & Metrics
(List the most critical quantitative results. Compare against SOTA if mentioned. Format as bullet points with bold metrics.)
* **Metric Name**: Value (Context/Comparison)

### Core Takeaway
(1 sentence on the immediate utility of this method for engineers.)
</output_schema>

<input_notes>
{text}
</input_notes>

**Final Technical Report:**
"""

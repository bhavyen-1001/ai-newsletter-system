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

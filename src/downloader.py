"""Module for downloading papers from arXiv and extracting text."""

import logging
import os
import time
from typing import Optional

import fitz  # PyMuPDF - pylint: disable=import-error
import requests

from src import config


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PaperDownloader:
    """Downloads papers from arXiv and extracts text content.

    Attributes:
        paper_id: The arXiv paper ID (e.g., '2511.11793').
        save_dir: Directory to save downloaded PDFs.
    """

    def __init__(self, paper_id: str, save_dir: str):
        """Initializes the PaperDownloader.

        Args:
            paper_id: arXiv paper ID to download.
            save_dir: Directory path where PDF should be saved.
        """
        self.paper_id = paper_id
        self.save_dir = save_dir
        self.pdf_path = os.path.join(save_dir, f"{paper_id}.pdf")

    def download_pdf(self) -> bool:
        """Downloads the PDF from arXiv.

        Returns:
            True if download successful, False otherwise.
        """
        url = f"{config.ARXIV_PDF_BASE_URL}/{self.paper_id}.pdf"

        try:
            logger.info("Downloading paper %s from %s", self.paper_id, url)
            response = requests.get(
                url,
                timeout=config.REQUEST_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            # Save the PDF
            with open(self.pdf_path, "wb") as file:
                file.write(response.content)

            logger.info("Downloaded PDF to %s", self.pdf_path)
            return True

        except requests.RequestException as error:
            logger.error("Error downloading %s: %s", self.paper_id, error)
            return False

    def extract_text(self) -> Optional[str]:
        """Extracts text content from the downloaded PDF.

        Returns:
            The extracted text as a string, or None if extraction fails.
        """
        if not os.path.exists(self.pdf_path):
            logger.error("PDF not found at %s", self.pdf_path)
            return None

        try:
            logger.info("Extracting text from %s", self.pdf_path)
            document = fitz.open(self.pdf_path)  # pylint: disable=no-member
            text = ""

            # Extract text from each page
            for page_num, page in enumerate(document):
                text += page.get_text()
                logger.debug("Extracted text from page %d", page_num)

            document.close()

            logger.info("Extracted %d characters from %s", len(text), self.paper_id)
            return text

        except (IOError, RuntimeError) as error:
            logger.error("Error extracting text from %s: %s", self.paper_id, error)
            return None

    def download_and_extract(self) -> Optional[str]:
        """Downloads PDF and extracts text in one step.

        Returns:
            The extracted text, or None if any step fails.
        """
        if self.download_pdf():
            return self.extract_text()
        return None


def download_papers(paper_ids: list[str]) -> dict[str, str]:
    """Downloads multiple papers and extracts their text.

    Args:
        paper_ids: List of arXiv paper IDs to download.

    Returns:
        A dictionary mapping paper IDs to their extracted text.
        Papers that failed to download will not be in the dictionary.
    """
    # Create save directory for this week
    week_folder = config.get_week_folder_name()
    save_dir = os.path.join(config.PAPERS_DIR, week_folder)
    os.makedirs(save_dir, exist_ok=True)

    papers_text = {}

    for paper_id in paper_ids:
        logger.info("Processing paper %s", paper_id)
        downloader = PaperDownloader(paper_id, save_dir)
        text = downloader.download_and_extract()

        if text:
            papers_text[paper_id] = text

        # Be polite to arXiv - wait a bit between downloads
        time.sleep(3)

    logger.info(
        "Successfully downloaded %d out of %d papers", len(papers_text), len(paper_ids)
    )
    return papers_text

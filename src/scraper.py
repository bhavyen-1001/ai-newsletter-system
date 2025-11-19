"""Module for scraping paper links from HuggingFace weekly papers page."""

import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src import config


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PaperScraper:
    """Scrapes paper information from HuggingFace weekly papers page.

    This class handles fetching the HuggingFace weekly papers page and
    extracting arXiv paper IDs from the top papers listed.

    Attributes:
        url: The HuggingFace URL to scrape.
        top_n: Number of top papers to extract.
    """

    def __init__(self, url: str, top_n: int = 5):
        """Initializes the PaperScraper.

        Args:
            url: HuggingFace weekly papers URL to scrape.
            top_n: Number of top papers to extract. Defaults to 5.
        """
        self.url = url
        self.top_n = top_n

    def fetch_page(self) -> Optional[str]:
        """Fetches the HTML content of the HuggingFace page.

        Returns:
            The HTML content as a string, or None if fetch fails.

        Raises:
            requests.RequestException: If there's an error fetching the page.
        """
        try:
            logger.info(f"Fetching page: {self.url}")
            response = requests.get(
                self.url,
                timeout=config.REQUEST_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            logger.info("Page fetched successfully")
            return response.text

        except requests.RequestException as error:
            logger.error(f"Error fetching page: {error}")
            return None

    def extract_paper_ids(self, html_content: str) -> list[str]:
        """Extracts arXiv paper IDs from the HTML content.

        Args:
            html_content: The HTML content to parse.

        Returns:
            A list of arXiv paper IDs (e.g., ['2511.11793', '2511.12345']).
        """
        soup = BeautifulSoup(html_content, "html.parser")
        paper_ids = []

        # Find all links that match the pattern /papers/XXXXX.XXXXX
        links = soup.find_all("a", href=True)

        for link in links:
            href = link["href"]
            # Check if href matches pattern /papers/XXXX.XXXXX
            if href.startswith("/papers/") and "." in href:
                # Extract the paper ID (e.g., '2511.11793' from '/papers/2511.11793')
                paper_id = href.replace("/papers/", "")

                # Skip community versions - only get the main paper
                if "#community" in paper_id:
                    continue

                # Avoid duplicates
                if paper_id not in paper_ids:
                    paper_ids.append(paper_id)
                    logger.info("Found paper: %s", paper_id)

                # Stop when we have enough papers
                if len(paper_ids) >= self.top_n:
                    break

        logger.info("Extracted %d paper IDs", len(paper_ids))
        return paper_ids

    def scrape(self) -> list[str]:
        """Main method to scrape paper IDs from HuggingFace.

        Returns:
            A list of arXiv paper IDs.
        """
        html_content = self.fetch_page()

        if html_content is None:
            logger.error("Failed to fetch page content")
            return []

        paper_ids = self.extract_paper_ids(html_content)

        if len(paper_ids) < self.top_n:
            logger.warning(f"Only found {len(paper_ids)} papers, expected {self.top_n}")

        return paper_ids


def scrape_weekly_papers() -> list[str]:
    """Convenience function to scrape current week's top papers.

    Returns:
        A list of arXiv paper IDs for the current week's top papers.
    """
    url = config.get_current_week_url()
    scraper = PaperScraper(url, top_n=config.TOP_N_PAPERS)
    return scraper.scrape()

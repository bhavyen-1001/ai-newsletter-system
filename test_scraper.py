"""Test script for scraping and downloading papers."""

import logging

from src import config
from src.scraper import scrape_weekly_papers
from src.downloader import download_papers


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to test the scraping and downloading pipeline."""
    logger.info("Starting paper scraping and download test")

    # Ensure directories exist
    config.ensure_directories_exist()

    # Step 1: Scrape paper IDs
    logger.info("Step 1: Scraping paper IDs from HuggingFace")
    paper_ids = scrape_weekly_papers()

    if not paper_ids:
        logger.error("No papers found! Check the HuggingFace URL.")
        return

    logger.info(f"Found {len(paper_ids)} papers: {paper_ids}")

    # Step 2: Download papers
    logger.info("Step 2: Downloading papers from arXiv")
    papers_text = download_papers(paper_ids)

    # Step 3: Display results
    logger.info("\n" + "=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)

    for paper_id, text in papers_text.items():
        logger.info(f"\nPaper: {paper_id}")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"First 200 characters: {text[:200]}...")

    logger.info("\n" + "=" * 50)
    logger.info(f"Successfully processed {len(papers_text)} papers")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

"""Test script for paper summarization."""

import logging

from src import config
from src.scraper import scrape_weekly_papers
from src.downloader import download_papers
from src.summarizer import summarize_with_both_models


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to test the full pipeline including summarization."""
    logger.info("Starting full pipeline test")

    # Ensure directories exist
    config.ensure_directories_exist()

    # Step 1: Scrape papers (or use existing)
    logger.info("Step 1: Scraping papers")
    paper_ids = scrape_weekly_papers()

    if not paper_ids:
        logger.error("No papers found")
        return

    # For testing, just use the first paper
    test_paper_id = paper_ids[0]
    logger.info("Testing with paper: %s", test_paper_id)

    # Step 2: Download paper
    logger.info("Step 2: Downloading paper")
    papers_text = download_papers([test_paper_id])

    if test_paper_id not in papers_text:
        logger.error("Failed to download test paper")
        return

    paper_text = papers_text[test_paper_id]
    logger.info("Paper text length: %d characters", len(paper_text))

    # Step 3: Summarize with both models
    logger.info("Step 3: Summarizing paper")
    summaries = summarize_with_both_models(paper_text)

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARIZATION RESULTS")
    logger.info("=" * 60)
    logger.info("\nPaper ID: %s", test_paper_id)

    for model_name, summary in summaries.items():
        logger.info("\n" + "-" * 60)
        logger.info("Model: %s", model_name)
        logger.info("-" * 60)
        if summary:
            logger.info("\n%s\n", summary)
        else:
            logger.info("Failed to generate summary")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

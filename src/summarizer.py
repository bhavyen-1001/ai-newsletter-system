"""Module for summarizing papers using LLMs via Vertex AI."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from google.cloud import aiplatform
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import HumanMessage

from src import config
from src.text_chunker import chunk_paper_text


# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PaperSummarizer:
    """Summarizes research papers using map-reduce with LLMs.

    Attributes:
        project_id: Google Cloud project ID.
        location: Vertex AI location.
        model_name: Name of the model to use.
    """

    def __init__(
        self,
        model_name: str,
        project_id: Optional[str] = None,
        location: str = "us-central1",
    ):
        """Initializes the PaperSummarizer.

        Args:
            model_name: Vertex AI model name to use.
            project_id: Google Cloud project ID. If None, reads from env.
            location: Vertex AI location.
        """
        self.model_name = model_name
        self.location = location

        # Get project ID
        if project_id is None:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT not set in environment")

        self.project_id = project_id

        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)
        logger.info(
            "Initialized Vertex AI for project %s in %s", self.project_id, self.location
        )

        # Initialize LLM
        self.llm = ChatVertexAI(
            model_name=self.model_name, temperature=0.3, max_output_tokens=2048
        )

    def summarize_chunk(self, chunk: str) -> Optional[str]:
        """Summarizes a single text chunk.

        Args:
            chunk: Text chunk to summarize.

        Returns:
            Summary of the chunk, or None if summarization fails.
        """
        try:
            prompt = config.MAP_PROMPT.format(text=chunk)
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            summary = response.content.strip()

            logger.info("Summarized chunk (length: %d -> %d)", len(chunk), len(summary))
            return summary

        except Exception as error:
            logger.error("Error summarizing chunk: %s", error)
            return None

    def combine_summaries(self, summaries: list[str]) -> Optional[str]:
        """Combines multiple chunk summaries into one final summary.

        Args:
            summaries: List of chunk summaries to combine.

        Returns:
            Combined final summary, or None if combination fails.
        """
        try:
            # Join summaries with separators
            combined = "\n\n---\n\n".join(summaries)
            prompt = config.REDUCE_PROMPT.format(summaries=combined)
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            final_summary = response.content.strip()

            logger.info("Combined %d summaries into final summary", len(summaries))
            return final_summary

        except Exception as error:
            logger.error("Error combining summaries: %s", error)
            return None

    def summarize_paper(self, text: str) -> Optional[str]:
        """Performs map-reduce summarization on a paper.

        Args:
            text: Full paper text to summarize.

        Returns:
            Final summary of the paper, or None if summarization fails.
        """
        logger.info("Starting summarization with model: %s", self.model_name)

        # Step 1: Chunk the text
        chunks = chunk_paper_text(text, chunk_size=config.CHUNK_SIZE)

        if not chunks:
            logger.error("No chunks created from text")
            return None

        logger.info("Created %d chunks", len(chunks))

        # Step 2: Summarize each chunk (MAP phase)
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            logger.info("Summarizing chunk %d/%d", i, len(chunks))
            summary = self.summarize_chunk(chunk)

            if summary:
                chunk_summaries.append(summary)
            else:
                logger.warning("Failed to summarize chunk %d", i)

        if not chunk_summaries:
            logger.error("No chunks were successfully summarized")
            return None

        logger.info("Successfully summarized %d chunks", len(chunk_summaries))

        # Step 3: If only one chunk, return its summary
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Step 4: Combine summaries (REDUCE phase)
        logger.info("Combining chunk summaries")
        final_summary = self.combine_summaries(chunk_summaries)

        return final_summary


def summarize_with_both_models(text: str) -> dict[str, Optional[str]]:
    """Summarizes text using both models.

    Args:
        text: Paper text to summarize.

    Returns:
        Dictionary with model names as keys and summaries as values.
    """
    results = {}

    # Summarize with Gemma (or Gemini for testing)
    logger.info("Summarizing with first model")
    gemma_summarizer = PaperSummarizer(config.GEMMA_MODEL)
    results["gemma"] = gemma_summarizer.summarize_paper(text)

    # For now, we're using the same model twice for testing
    # In production, you would use different models
    logger.info("Summarizing with second model")
    gpt_summarizer = PaperSummarizer(config.GPT_OSS_MODEL)
    results["gpt-oss"] = gpt_summarizer.summarize_paper(text)

    return results

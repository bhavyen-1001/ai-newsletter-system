"""Module for splitting text into chunks for processing."""

import logging
from typing import Optional

import tiktoken


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into chunks based on token count.

    Attributes:
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
        encoding: Tiktoken encoding to use for tokenization.
    """

    def __init__(
        self,
        chunk_size: int = 20000,
        chunk_overlap: int = 500,
        encoding_name: str = "cl100k_base",
    ):
        """Initializes the TextChunker.

        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens to overlap between chunks.
            encoding_name: Name of tiktoken encoding to use.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as error:
            logger.error("Error loading encoding: %s", error)
            raise

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in text.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str) -> list[str]:
        """Splits text into chunks based on token count.

        Args:
            text: The text to split into chunks.

        Returns:
            A list of text chunks.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # Encode the entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        logger.info("Total tokens in text: %d", total_tokens)

        # If text fits in one chunk, return as is
        if total_tokens <= self.chunk_size:
            logger.info("Text fits in single chunk")
            return [text]

        # Split into chunks
        chunks = []
        start = 0

        while start < total_tokens:
            # Get chunk tokens
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            logger.info("Created chunk %d: tokens %d-%d", len(chunks), start, end)

            # Move start position (with overlap)
            start = end - self.chunk_overlap

            # Avoid infinite loop
            if start >= total_tokens:
                break

        logger.info("Created %d chunks from text", len(chunks))
        return chunks


def chunk_paper_text(text: str, chunk_size: int = 20000) -> list[str]:
    """Convenience function to chunk paper text.

    Args:
        text: The paper text to chunk.
        chunk_size: Maximum tokens per chunk.

    Returns:
        List of text chunks.
    """
    chunker = TextChunker(chunk_size=chunk_size)
    return chunker.chunk_text(text)

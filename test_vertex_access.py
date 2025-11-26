"""Test script to verify Vertex AI access and list available models."""

import logging
import os

from dotenv import load_dotenv
from google.cloud import aiplatform


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test Vertex AI connection and list available models."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT not set in .env file")
        return

    location = "europe-west2"  # Change if needed

    logger.info("Initializing Vertex AI")
    logger.info("Project: %s", project_id)
    logger.info("Location: %s", location)

    try:
        aiplatform.init(project=project_id, location=location)
        logger.info("Successfully connected to Vertex AI!")

        # Try listing models (may be empty if you haven't deployed any)
        logger.info("\nAttempting to list models...")
        models = aiplatform.Model.list()

        if models:
            logger.info("Found %d models:", len(models))
            for model in models:
                logger.info("  - %s", model.display_name)
        else:
            logger.info("No deployed models found (this is normal)")
            logger.info("Models in Model Garden need to be accessed differently")

    except Exception as error:
        logger.error("Error connecting to Vertex AI: %s", error)
        logger.error("Check your service account key and permissions")


if __name__ == "__main__":
    main()

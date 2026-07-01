# Architecture

## Purpose

The system builds AI Research Weekly, a weekly email that finds new Hugging Face trending AI papers, summarises the arXiv PDFs, and sends the result through Mailchimp.

## Workflow Diagram

Add the exported draw.io workflow image here.

Suggested path: `docs/workflow-flowchart.png`

## Architecture Choices

- **Scheduled command line workflow:** The project is a Python CLI so it can run locally, in dry run mode, or on a GitHub Actions schedule without a long-running service.
- **Hugging Face as the discovery source:** The workflow reads the weekly Hugging Face Trending Papers page because it provides a ranked shortlist and avoids building a separate discovery system.
- **arXiv as the paper source:** arXiv provides stable IDs, metadata, links, and PDFs. The summariser uses extracted PDF text only, so summaries stay grounded in the paper.
- **Small persistent state:** `data/sent_papers.json` stores sent arXiv IDs and Mailchimp campaign IDs. This prevents duplicate sends while keeping generated files out of the repo.
- **Provider boundary for email:** Email delivery is behind `EmailProvider`, with Mailchimp as the first implementation. This keeps the workflow separate from Mailchimp API details and leaves room for another provider later.
- **Explicit run modes:** `dry_run`, `test_only`, and `send` separate generation, review, and live delivery. Sent state is updated only after a full campaign send succeeds.
- **Validation before delivery:** The issue must include a title, authors, arXiv link, and non-empty summary fields before it can be sent.
- **Review artefacts:** HTML, plain text, JSON, and the workflow summary are written to `outputs/` and uploaded by GitHub Actions instead of being committed.
- **Retries and skips:** External fetches and summarisation are retried. A paper that cannot be fetched, parsed, or summarised is skipped, so one failed paper does not stop the whole issue.

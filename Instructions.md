# AI Research Weekly Newsletter

This repo contains a simple weekly workflow that turns the top new Hugging Face Trending Papers into an email newsletter and sends it through Mailchimp.

## Architecture

- GitHub Actions runs every Sunday at `12:05 UTC`.
- The workflow reads `https://huggingface.co/papers/week/YYYY-Www`.
- It walks the trending list in order and skips arXiv IDs already present in `data/sent_papers.json`.
- For each new paper, it fetches arXiv metadata and downloads the arXiv PDF.
- PDF text is extracted with PyMuPDF.
- Hugging Face Inference summarises the extracted PDF text only, using a primary model and one fallback model.
- The renderer creates HTML and plain text email bodies.
- Long author lists are abbreviated as `First Author et al.` in rendered outputs.
- Automated checks require title, authors, arXiv link and non-empty summary fields.
- In test-only mode, Mailchimp creates a regular campaign, sets the content, and sends a test campaign to `ADMIN_EMAIL` only.
- In send mode, Mailchimp creates a regular campaign, sends a test campaign to `ADMIN_EMAIL`, then sends to the full audience.
- Only after the full campaign send succeeds, `data/sent_papers.json` is updated.

## Storage Decision

The best v1 storage option is:

- Commit only `data/sent_papers.json` to the repo.
- Store generated HTML, text and JSON issues as GitHub Actions artifacts.
- Use Mailchimp campaign history as the sent-email archive.

This keeps the repo clean while preserving the one persistent state file needed for duplicate prevention.

## Email Signup and Unsubscribe

Use a Mailchimp-hosted signup form. It is the easiest route because Mailchimp handles:

- double opt-in;
- subscriber storage;
- unsubscribe links;
- list hygiene.

Add the hosted signup URL as `NEWSLETTER_SIGNUP_URL`. The email template also includes Mailchimp's `*|UNSUB|*` merge tag.

## Required GitHub Secrets

Add these in GitHub under `Settings -> Secrets and variables -> Actions -> Secrets`:

- `HF_TOKEN`
- `MAILCHIMP_API_KEY`
- `MAILCHIMP_SERVER_PREFIX`
- `MAILCHIMP_AUDIENCE_ID`
- `MAILCHIMP_REPLY_TO`
- `ADMIN_EMAIL`

No secret values should be committed to the repo.

## Recommended GitHub Variables

Add these under `Settings -> Secrets and variables -> Actions -> Variables`:

- `HF_MODEL_ID`: `google/gemma-3-27b-it`
- `HF_PROVIDER`: `featherless-ai`
- `HF_FALLBACK_MODEL_ID`: `CohereLabs/aya-expanse-32b`
- `HF_FALLBACK_PROVIDER`: `cohere`
- `MAILCHIMP_FROM_NAME`
- `NEWSLETTER_SUBJECT_PREFIX`: `AI Research Weekly`
- `NEWSLETTER_SIGNUP_URL`

## Mailchimp Setup

1. Create a Mailchimp account.
2. Create an audience/list.
3. Enable double opt-in for that audience.
4. Verify the sender email or domain.
5. Create an API key.
6. Find the server prefix from the API key suffix or account URL, for example `us6`.
7. Copy the audience/list ID into `MAILCHIMP_AUDIENCE_ID`.
8. Copy the hosted signup form URL into `NEWSLETTER_SIGNUP_URL`.

## Local Commands

Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run a local dry run without an LLM call:

```bash
python -m newsletter run --dry-run --mock-llm --week 2026-W26
```

Run a dry run with Hugging Face Inference:

```bash
python -m newsletter run --dry-run --week 2026-W26
```

Send for real:

```bash
python -m newsletter run --send
```

Send a Mailchimp-rendered test email to `ADMIN_EMAIL` without sending to subscribers:

```bash
python -m newsletter run --test-only
```

By default, `run` is a dry run unless `--test-only` or `--send` is passed.

## GitHub Actions Manual Modes

When manually running the workflow, choose one of:

- `dry_run`: generate artifacts and the workflow summary only.
- `test_only`: create a Mailchimp campaign and send a test email to `ADMIN_EMAIL`, but do not send to subscribers or update `data/sent_papers.json`.
- `send`: send to subscribers and update `data/sent_papers.json`.

## Failover Behaviour

- Hugging Face weekly page fetch retries 3 times.
- arXiv metadata fetch retries 3 times.
- PDF download retries 2 times.
- LLM summarisation retries transient failures 3 times for each model endpoint.
- If the primary LLM endpoint fails or returns unusable JSON, the workflow tries the fallback model endpoint.
- If a paper cannot be downloaded, extracted or summarised, the workflow skips it and tries the next trending paper.
- If fewer than 3 usable new papers are available, it sends fewer.
- If zero papers are usable, validation fails and no email is sent.

## Duplicate Prevention

`data/sent_papers.json` stores:

- arXiv ID;
- title;
- authors;
- source week;
- sent timestamp;
- Mailchimp campaign ID.

A paper is marked as sent only after Mailchimp successfully sends the full campaign.

## Manual Recovery

If Mailchimp sends successfully but the GitHub commit of `data/sent_papers.json` fails, the next run may try to send the same papers again.

To recover:

1. Open the successful workflow run summary or Mailchimp campaign.
2. Copy the arXiv IDs, titles, authors, week and campaign ID.
3. Manually add those entries to `data/sent_papers.json`.
4. Commit the file to the repo before the next scheduled run.

## Provider Boundary

Email delivery is isolated behind `newsletter.providers.base.EmailProvider`.

Mailchimp is the first implementation. To switch later, add a new provider with the same methods:

- `create_campaign`;
- `set_campaign_content`;
- `send_test`;
- `send_campaign`.

The rest of the workflow should not need to change.

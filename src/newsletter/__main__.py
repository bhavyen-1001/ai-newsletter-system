from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from newsletter.config import Settings
from newsletter.workflow import NewsletterWorkflow


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m newsletter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Build and optionally send the weekly newsletter.")
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Generate outputs without sending.")
    mode.add_argument("--send", action="store_true", help="Send via Mailchimp and update sent state.")
    run_parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock summaries.")
    run_parser.add_argument("--week", help="Hugging Face week, for example 2026-W26.")
    run_parser.add_argument("--count", type=int, default=3, help="Number of papers to include.")
    run_parser.add_argument("--state-path", default="data/sent_papers.json")
    run_parser.add_argument("--output-dir", default="outputs")

    args = parser.parse_args(argv)
    if args.command == "run":
        dry_run = not args.send
        settings = Settings.from_env(
            state_path=Path(args.state_path),
            output_dir=Path(args.output_dir),
            target_count=args.count,
        )
        result = NewsletterWorkflow(settings).run(
            week=args.week,
            dry_run=dry_run,
            mock_llm=args.mock_llm,
        )
        print(f"Week: {result.week}")
        print(f"Mode: {'dry-run' if result.dry_run else 'send'}")
        print(f"Papers selected: {result.selected_count}")
        if result.campaign_id:
            print(f"Mailchimp campaign ID: {result.campaign_id}")
        print(f"Outputs: {result.output_dir}")
        if result.skipped:
            print("Skipped:")
            for item in result.skipped[:20]:
                print(f"- {item}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

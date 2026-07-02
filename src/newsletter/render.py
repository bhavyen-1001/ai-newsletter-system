from __future__ import annotations

import html

from newsletter.authors import format_authors
from newsletter.models import NewsletterIssue


def render_html(issue: NewsletterIssue, *, subject_prefix: str, signup_url: str | None = None) -> str:
    title = html.escape(f"{subject_prefix} - {issue.week}")
    intro_links = ", ".join(
        f'<a href="#{_anchor(paper.metadata.arxiv_id)}">{html.escape(paper.metadata.title)}</a>'
        for paper in issue.papers
    )
    issue_count = len(issue.papers)
    paper_word = "paper" if issue_count == 1 else "papers"
    signup = (
        f'<p>Share the signup page: <a href="{html.escape(signup_url)}">{html.escape(signup_url)}</a></p>'
        if signup_url
        else ""
    )

    sections = "\n".join(_render_paper_html(index, paper) for index, paper in enumerate(issue.papers, start=1))

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>{title}</title>
  </head>
  <body style="margin:0;background:#f5f7fa;color:#111827;font-family:Arial,Helvetica,sans-serif;line-height:1.55;">
    <main style="max-width:760px;margin:0 auto;padding:32px 20px;background:#ffffff;">
      <h1 style="font-size:28px;margin:0 0 12px;">{title}</h1>
      <p style="margin:0 0 20px;">This week's Hugging Face Trending Papers shortlist includes {issue_count} new {paper_word}: {intro_links}.</p>
      {signup}
      {sections}
      <hr style="border:none;border-top:1px solid #e5e7eb;margin:32px 0 16px;">
      <p style="font-size:12px;color:#4b5563;">You are receiving this because you subscribed to the AI Research Weekly mailing list. <a href="*|UNSUB|*">Unsubscribe</a>.</p>
    </main>
  </body>
</html>
"""


def render_text(issue: NewsletterIssue, *, subject_prefix: str, signup_url: str | None = None) -> str:
    lines = [
        f"{subject_prefix} - {issue.week}",
        "",
        "This week's Hugging Face Trending Papers shortlist:",
    ]
    for index, paper in enumerate(issue.papers, start=1):
        lines.append(f"{index}. {paper.metadata.title} [{paper.metadata.arxiv_url}]")
    if signup_url:
        lines.extend(["", f"Signup page: {signup_url}"])
    lines.append("")

    for index, paper in enumerate(issue.papers, start=1):
        metadata = paper.metadata
        summary = paper.summary
        lines.extend(
            [
                f"{index}. {metadata.title}",
                f"Authors: {format_authors(metadata.authors)}",
                f"Link: {metadata.arxiv_url}",
                "",
                f"Executive summary: {summary.executive_summary}",
                f"Problem: {summary.problem}",
                f"Method: {summary.method}",
                f"Why it matters: {summary.why_it_matters}",
                f"Limitations: {summary.limitations}",
                "",
            ]
        )

    lines.extend(
        [
            "You are receiving this because you subscribed to the AI Research Weekly mailing list.",
            "Unsubscribe: *|UNSUB|*",
        ]
    )
    return "\n".join(lines)


def _render_paper_html(index: int, paper) -> str:
    metadata = paper.metadata
    summary = paper.summary
    authors = html.escape(format_authors(metadata.authors))
    return f"""
      <section id="{_anchor(metadata.arxiv_id)}" style="margin-top:30px;">
        <h2 style="font-size:22px;margin:0 0 8px;">{index}. {html.escape(metadata.title)}</h2>
        <p style="margin:0 0 8px;color:#374151;"><strong>Authors:</strong> {authors}</p>
        <p style="margin:0 0 16px;"><a href="{html.escape(metadata.arxiv_url)}">Read on arXiv</a></p>
        {_summary_block("Executive Summary", summary.executive_summary)}
        {_summary_block("Problem", summary.problem)}
        {_summary_block("Method", summary.method)}
        {_summary_block("Why It Matters", summary.why_it_matters)}
        {_summary_block("Limitations", summary.limitations)}
      </section>
"""


def _summary_block(label: str, value: str) -> str:
    return f'<p style="margin:0 0 12px;"><strong>{html.escape(label)}:</strong> {html.escape(value)}</p>'


def _anchor(arxiv_id: str) -> str:
    return "paper-" + arxiv_id.replace(".", "-").replace("/", "-")

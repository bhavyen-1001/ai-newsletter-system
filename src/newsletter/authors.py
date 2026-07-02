from __future__ import annotations


def format_authors(authors: list[str], *, max_full_authors: int = 3) -> str:
    if not authors:
        return "Unknown authors"
    if len(authors) <= max_full_authors:
        return ", ".join(authors)
    return f"{authors[0]} et al."

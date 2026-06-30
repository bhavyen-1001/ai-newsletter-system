from newsletter.hf_papers import parse_trending_paper_ids


def test_parse_trending_paper_ids_deduplicates_in_order():
    html = """
    <a href="/papers/2606.00002">Second</a>
    <a href="/papers/2606.00001">First</a>
    <a href="/papers/2606.00002">Second duplicate</a>
    <a href="/papers/week/2026-W26">Week</a>
    """

    assert parse_trending_paper_ids(html) == ["2606.00002", "2606.00001"]

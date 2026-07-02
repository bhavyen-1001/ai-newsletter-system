from newsletter.authors import format_authors


def test_format_authors_keeps_short_lists():
    assert format_authors(["Ada Lovelace", "Alan Turing"]) == "Ada Lovelace, Alan Turing"


def test_format_authors_abbreviates_long_lists():
    assert (
        format_authors(["Ada Lovelace", "Alan Turing", "Grace Hopper", "Katherine Johnson"])
        == "Ada Lovelace et al."
    )

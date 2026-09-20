"""Tests for `nagcsu.deck` against the real UGH-1 deck."""

import pytest

from nagcsu.deck import Deck
from nagcsu.exceptions import DeckPatchError


def test_replace_once_requires_exactly_one_match(sample_deck: Deck) -> None:
    with pytest.raises(DeckPatchError):
        sample_deck.replace_once(r"THIS_PATTERN_DOES_NOT_EXIST", "x")


def test_replace_once_rejects_ambiguous_pattern(sample_deck: Deck) -> None:
    # 'PERMX' appears many times in the deck; a bare match is ambiguous.
    with pytest.raises(DeckPatchError):
        sample_deck.replace_once(r"'PERMX'", "'PERMY'")


def test_transform_each_respects_expected_count(sample_deck: Deck) -> None:
    with pytest.raises(DeckPatchError):
        sample_deck.transform_each(r"'PORO'", lambda match: match.group(), expected_count=999)


def test_read_relperm_table_parses_sgof(sample_deck: Deck) -> None:
    from nagcsu.deck import read_relperm_table

    rows = read_relperm_table(sample_deck, "SGOF")
    assert len(rows) == 16
    assert rows[0] == (0.0, 0.0, 0.8, 0.0)
    assert rows[-1][0] == pytest.approx(0.72)


def test_read_relperm_table_parses_swof(sample_deck: Deck) -> None:
    from nagcsu.deck import read_relperm_table

    rows = read_relperm_table(sample_deck, "SWOF")
    assert len(rows) == 16
    assert rows[0][0] == pytest.approx(0.13)


def test_patch_relperm_table_rewrites_only_kr_columns(sample_deck: Deck) -> None:
    from nagcsu.deck import patch_relperm_table, read_relperm_table

    def double_kr(
        rows: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        return [(sg, krg * 2, krog, pc) for sg, krg, krog, pc in rows]

    patched = patch_relperm_table(sample_deck, "SGOF", double_kr)
    original_rows = read_relperm_table(sample_deck, "SGOF")
    patched_rows = read_relperm_table(patched, "SGOF")

    for original, patched_row in zip(original_rows, patched_rows, strict=True):
        assert patched_row[0] == pytest.approx(original[0])  # Sg unchanged
        assert patched_row[3] == pytest.approx(original[3])  # Pcog unchanged
        assert patched_row[1] == pytest.approx(original[1] * 2, abs=1e-4)  # Krg doubled


def test_find_block_span_covers_keyword_through_closing_slash(sample_deck: Deck) -> None:
    from nagcsu.deck import find_block_span

    start, end = find_block_span(sample_deck, "ROCK")
    block_text = sample_deck.text[start:end]
    assert block_text.startswith("ROCK")
    assert block_text.rstrip().endswith("/")
    assert "3.577E-06" in block_text

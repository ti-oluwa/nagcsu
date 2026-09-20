"""Reading and safely patching an OPM Flow `.DATA` deck as text.

The UGH-1 deck is a single monolithic file rather than a base deck plus
`INCLUDE` files, so every patch here operates directly on the deck text
and returns a new `Deck`. Every patch is refused rather than
guessed at if its target pattern does not match exactly once, since a
silent wrong-occurrence match (the exact failure mode Stage D.5 of the
Execution Plan warns about for hand-edited `INCLUDE` files) is far worse
than a loud error.
"""

import dataclasses
import pathlib
import re
import typing

from nagcsu.exceptions import DeckPatchError

RowTransform = typing.Callable[
    [list[tuple[float, float, float, float]]],
    list[tuple[float, float, float, float]],
]
"""A function that takes the parsed rows of a 4-column relative
permeability table and returns the replacement rows, same length and
same row order.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class Deck:
    """The full text of an OPM Flow `.DATA` deck, plus its source path."""

    text: str
    """Complete contents of the deck file."""

    path: pathlib.Path
    """Path the deck was loaded from. Used as the default save target."""

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "Deck":
        """Read a deck from disk.

        :raises FileNotFoundError: if `path` does not exist.
        """
        path = pathlib.Path(path)
        return cls(text=path.read_text(encoding="utf-8"), path=path)

    def save(self, path: pathlib.Path | str | None = None) -> pathlib.Path:
        """Write this deck's text out to disk.

        :param path: Destination path. Defaults to the path the deck was
            loaded from, overwriting it.
        :returns: The path written to.
        """
        destination = pathlib.Path(path) if path is not None else self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.text, encoding="utf-8")
        return destination

    def replace_once(self, pattern: str, replacement: str, *, flags: int = 0) -> "Deck":
        """Return a new `Deck` with `pattern` substituted by `replacement`.

        `pattern` must match `text` in exactly one place; this is what
        makes every edit auditable instead of a hopeful regex that might
        have hit the wrong AQUANCON record or the wrong well's line.

        :raises DeckPatchError: if `pattern` matches zero or more than
            one place in the deck text.
        """
        compiled = re.compile(pattern, flags)
        matches = list(compiled.finditer(self.text))
        if len(matches) != 1:
            raise DeckPatchError(
                f"Expected exactly one match for pattern {pattern!r} in "
                f"{self.path}, found {len(matches)}"
            )
        new_text = compiled.sub(replacement, self.text, count=1)
        return dataclasses.replace(self, text=new_text)

    def transform_each(
        self,
        pattern: str,
        transform: typing.Callable[[re.Match], str],
        *,
        expected_count: int | None = None,
        flags: int = 0,
    ) -> "Deck":
        """Return a new `Deck` with every match of `pattern` rewritten by `transform`.

        Unlike `replace_once`, this supports patching several
        occurrences at once (for example, every layer's `PORO` value in
        an `EQUALS` block), but only after confirming the match count is
        exactly what the caller expects.

        :param transform: Called once per match with the `re.Match`
            object; its return value replaces that match in the text.
        :param expected_count: If given, the number of matches that must
            be found. Leave unset only when the count genuinely varies
            (for example, tables that may gain or lose rows), since
            checking the count is what catches an ambiguous pattern.
        :raises DeckPatchError: if `expected_count` is given and does
            not equal the number of matches found.
        """
        compiled = re.compile(pattern, flags)
        matches = list(compiled.finditer(self.text))
        if expected_count is not None and len(matches) != expected_count:
            raise DeckPatchError(
                f"Expected {expected_count} matches for pattern {pattern!r} in "
                f"{self.path}, found {len(matches)}"
            )
        new_text = compiled.sub(lambda match: transform(match), self.text)
        return dataclasses.replace(self, text=new_text)

    def find_once(self, pattern: str, *, flags: int = 0) -> re.Match:
        """Return the single match of `pattern` against this deck's text.

        :raises DeckPatchError: if `pattern` matches zero or more than
            one place in the deck text.
        """
        compiled = re.compile(pattern, flags)
        matches = list(compiled.finditer(self.text))
        if len(matches) != 1:
            raise DeckPatchError(
                f"Expected exactly one match for pattern {pattern!r} in "
                f"{self.path}, found {len(matches)}"
            )
        return matches[0]


NUMBER_PATTERN: typing.Final[str] = r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?"
"""Regex fragment matching an OPM-style numeric literal (plain decimal
or exponential notation using e, E, d or D). Exported so other modules
that build their own deck patch patterns, such as
`nagcsu.parameters`, do not need to redefine it.
"""


def read_relperm_table(deck: Deck, keyword: str) -> list[tuple[float, float, float, float]]:
    """Parse the 4-column data rows of a `SWOF` or `SGOF` table.

    :param keyword: "SWOF" or "SGOF".
    :returns: One `(saturation, kr1, kr2, capillary_pressure)` tuple per
        data row, in file order.
    :raises DeckPatchError: if the keyword's block cannot be found.
    """
    block = find_table_block(deck, keyword)
    rows: list[tuple[float, float, float, float]] = []
    row_pattern = re.compile(
        rf"^\s*({NUMBER_PATTERN})\s+({NUMBER_PATTERN})\s+({NUMBER_PATTERN})\s+({NUMBER_PATTERN})\s*$"
    )
    for line in block.body_lines:
        match = row_pattern.match(line)
        if match:
            rows.append(tuple(float(value) for value in match.groups()))  # type: ignore[misc]
    return rows


def patch_relperm_table(deck: Deck, keyword: str, transform: RowTransform) -> Deck:
    """Recompute a `SWOF` or `SGOF` table's data rows and patch it into the deck.

    :param keyword: "SWOF" or "SGOF".
    :param transform: Called with the table's current rows, must return
        the replacement rows (same length and order). Typically built
        from `nagcsu.corey` by keeping the saturation and
        capillary-pressure columns and recomputing the two relative
        permeability columns.
    :raises DeckPatchError: if the keyword's block cannot be found, or
        if `transform` returns a different number of rows than it was given.
    """
    block = find_table_block(deck, keyword)
    current_rows = read_relperm_table(deck, keyword)
    new_rows = transform(current_rows)
    if len(new_rows) != len(current_rows):
        raise DeckPatchError(
            f"Row transform for {keyword} returned {len(new_rows)} rows, "
            f"expected {len(current_rows)}"
        )
    new_body = "\n".join(format_relperm_row(row) for row in new_rows)
    new_block_text = f"{block.header}\n{new_body}\n/"
    new_text = deck.text[: block.start] + new_block_text + deck.text[block.end :]
    return dataclasses.replace(deck, text=new_text)


def format_relperm_row(row: tuple[float, float, float, float]) -> str:
    """Format one relative permeability table row in the deck's own style."""
    saturation, kr1, kr2, capillary_pressure = row
    return f"{saturation:.4f}   {kr1:.5f}   {kr2:.5f}   {capillary_pressure:.3f}"


def find_block_span(deck: Deck, keyword: str) -> tuple[int, int]:
    """Return the `(start, end)` character offsets of a `KEYWORD ... /` block.

    `start` is the beginning of the keyword's own line; `end` is just
    past the first lone `/` line found after it, which OPM decks use to
    close a keyword's data. Useful for scoping a patch pattern to one
    keyword's record so it cannot accidentally match a similarly shaped
    line elsewhere in the deck.

    :raises DeckPatchError: if `keyword` does not appear exactly once as
        a standalone keyword line, or if no closing `/` line follows it.
    """
    keyword_pattern = re.compile(rf"^{re.escape(keyword)}\s*$", re.MULTILINE)
    matches = list(keyword_pattern.finditer(deck.text))
    if len(matches) != 1:
        raise DeckPatchError(
            f"Expected exactly one {keyword} keyword line in {deck.path}, found {len(matches)}"
        )
    keyword_match = matches[0]
    closing_pattern = re.compile(r"^\s*/\s*$", re.MULTILINE)
    closing_match = closing_pattern.search(deck.text, pos=keyword_match.end())
    if closing_match is None:
        raise DeckPatchError(f"No closing '/' found for {keyword} block in {deck.path}")
    return keyword_match.start(), closing_match.end()


def transform_within_block(
    deck: Deck,
    keyword: str,
    pattern: str,
    transform: typing.Callable[[re.Match], str],
    *,
    expected_count: int | None = None,
    flags: int = 0,
) -> "Deck":
    """Apply `Deck.transform_each`, restricted to one keyword's block.

    Scoping the search to `keyword`'s own `(start, end)` span (see
    `find_block_span`) is what keeps a positional pattern such as
    AQUCT's field list from also matching an unrelated numeric line
    elsewhere in the deck.
    """
    start, end = find_block_span(deck, keyword)
    block_text = deck.text[start:end]
    block_deck = dataclasses.replace(deck, text=block_text)
    patched_block = block_deck.transform_each(
        pattern, transform, expected_count=expected_count, flags=flags
    )
    new_text = deck.text[:start] + patched_block.text + deck.text[end:]
    return dataclasses.replace(deck, text=new_text)


@dataclasses.dataclass(frozen=True, slots=True)
class TableBlock:
    """Span and content of one `KEYWORD ... /` table block in a deck."""

    header: str
    """Text from the keyword line up to (not including) the first data row."""

    body_lines: list[str]
    """Lines between the header and the closing `/`, including any
    comment lines, which are dropped when the table is rewritten.
    """

    start: int
    """Character offset of the keyword line in the deck text."""

    end: int
    """Character offset just past the closing `/` of the block."""


def find_table_block(deck: Deck, keyword: str) -> TableBlock:
    """Locate a `KEYWORD ... /` block and split it into header and body.

    :raises DeckPatchError: if `keyword` does not appear exactly once as
        a standalone keyword line, or if no closing `/` line is found
        after it.
    """
    start, end = find_block_span(deck, keyword)
    block_text = deck.text[start:end]
    lines = block_text.splitlines()
    data_row_pattern = re.compile(
        rf"^\s*{NUMBER_PATTERN}\s+{NUMBER_PATTERN}\s+{NUMBER_PATTERN}\s+{NUMBER_PATTERN}\s*$"
    )
    first_data_line_index = next(
        (index for index, line in enumerate(lines) if data_row_pattern.match(line)),
        len(lines),
    )
    header = "\n".join(lines[:first_data_line_index])
    body_lines = lines[first_data_line_index:-1] if len(lines) > 1 else []
    return TableBlock(header=header, body_lines=body_lines, start=start, end=end)

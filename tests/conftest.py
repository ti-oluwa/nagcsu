"""Shared fixtures for the nagcsu test suite.

Tests are written against the real sample deck and `.PRT` file already
checked into `Data/` from the project's first OPM Flow run, rather than
synthetic fixtures, so a passing test suite means the code actually
works against this project's real output, not just an idealized shape
of it.
"""

import pathlib

import pytest

from nagcsu.deck import Deck


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_DECK_PATH = REPO_ROOT / "Data" / "NigerDelta UGH1 Composite Field.DATA"
SAMPLE_PRT_PATH = REPO_ROOT / "Data" / "NIGERDELTA UGH1 COMPOSITE FIELD.PRT"


@pytest.fixture
def sample_deck() -> Deck:
    """The real UGH-1 base deck, loaded fresh for each test."""
    return Deck.load(SAMPLE_DECK_PATH)


@pytest.fixture
def sample_prt_path() -> pathlib.Path:
    """Path to the real `.PRT` file from the project's first OPM Flow run."""
    return SAMPLE_PRT_PATH

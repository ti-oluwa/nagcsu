"""Tests for `nagcsu.cli.commands.match`."""

import click
import pytest

from nagcsu.cli.commands import match


def test_parse_weight_overrides_parses_comma_separated_pairs() -> None:
    overrides = match.parse_weight_overrides("pressure=1,watercut=0,gor=0")

    assert overrides == {"pressure": 1.0, "watercut": 0.0, "gor": 0.0}


def test_parse_weight_overrides_allows_a_single_pair() -> None:
    assert match.parse_weight_overrides("gor=0") == {"gor": 0.0}


def test_parse_weight_overrides_raises_on_missing_equals() -> None:
    with pytest.raises(click.BadParameter):
        match.parse_weight_overrides("pressure")


def test_parse_weight_overrides_raises_on_non_numeric_value() -> None:
    with pytest.raises(click.BadParameter):
        match.parse_weight_overrides("gor=not-a-number")

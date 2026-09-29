"""Tests for `nagcsu.cli.commands.match`."""

import click
import numpy as np
import pytest
import yaml

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


def test_write_parameters_snapshot_coerces_numpy_floats(tmp_path) -> None:
    # Regression test: scipy.optimize returns numpy.float64, which
    # PyYAML's SafeDumper cannot serialize on its own.
    numpy_state = {"aquifer.radius": np.float64(55.94746325289693)}
    output_path = tmp_path / "snapshot.yaml"

    match.write_parameters_snapshot(numpy_state, output_path)

    loaded = yaml.safe_load(output_path.read_text())
    assert loaded == {"aquifer.radius": pytest.approx(55.94746325289693)}
    assert isinstance(loaded["aquifer.radius"], float)

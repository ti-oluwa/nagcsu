"""Tests for `nagcsu.parameters` against the real UGH-1 deck."""

import re

import pytest

from nagcsu import parameters
from nagcsu.deck import Deck, read_relperm_table


def test_default_state_round_trips_to_an_unchanged_deck(sample_deck: Deck) -> None:
    patched = parameters.apply_state(sample_deck, parameters.default_state())
    assert "16137.20" in patched.text
    assert "566.00" in patched.text
    assert "77.00" in patched.text


def test_apply_state_is_reproducible_from_the_pristine_base(sample_deck: Deck) -> None:
    state = parameters.default_state()
    state["aquifer.radius"] = 21000.0
    state["porosity.multiplier"] = 1.1

    first = parameters.apply_state(sample_deck, state)
    second = parameters.apply_state(sample_deck, state)
    assert first.text == second.text


def test_apply_aquifer_only_changes_the_four_tunable_fields(sample_deck: Deck) -> None:
    state = parameters.default_state()
    state["aquifer.radius"] = 25000.0
    state["aquifer.permeability"] = 900.0
    state["aquifer.thickness"] = 100.0
    state["aquifer.encroachment_angle"] = 120.0

    patched = parameters.apply_state(sample_deck, state)
    match = re.search(r"^\s*1\s+8700.*$", patched.text, re.MULTILINE)
    assert match is not None
    line = match.group()
    assert "25000.00" in line
    assert "900.00" in line
    assert "100.00" in line
    assert "120.00" in line
    # Untouched fields: depth (8700), Pi (2751.0), Poro (0.20), Ctotal (6.9E-06)
    assert "8700" in line
    assert "2751.0" in line


def test_apply_permeability_multiplier_is_symmetric(sample_deck: Deck) -> None:
    state = parameters.default_state()
    state["permeability.areal_contrast"] = 0.30

    patched = parameters.apply_state(sample_deck, state)
    high_values = re.findall(r"'PERMX'\s+([\d.]+)\s+1\s+10", patched.text)
    low_values = re.findall(r"'PERMX'\s+([\d.]+)\s+21\s+30", patched.text)
    assert float(high_values[0]) == pytest.approx(1.30)
    assert float(low_values[0]) == pytest.approx(0.70)


def test_apply_sgof_table_preserves_saturation_and_capillary_pressure(sample_deck: Deck) -> None:
    state = parameters.default_state()
    state["sgof.sorg"] = 0.22

    patched = parameters.apply_state(sample_deck, state)
    original_rows = read_relperm_table(sample_deck, "SGOF")
    patched_rows = read_relperm_table(patched, "SGOF")

    for original, new in zip(original_rows, patched_rows, strict=True):
        assert new[0] == original[0]  # Sg
        assert new[3] == original[3]  # Pcog


def test_apply_swof_table_preserves_saturation_and_capillary_pressure(sample_deck: Deck) -> None:
    state = parameters.default_state()
    state["swof.residual_oil_saturation"] = 0.18

    patched = parameters.apply_state(sample_deck, state)
    original_rows = read_relperm_table(sample_deck, "SWOF")
    patched_rows = read_relperm_table(patched, "SWOF")

    for original, new in zip(original_rows, patched_rows, strict=True):
        assert new[0] == original[0]  # Sw
        assert new[3] == original[3]  # Pcow


def test_swof_oil_exponent_is_tunable_and_actually_changes_krow(sample_deck: Deck) -> None:
    # swof.oil_exponent used to be hardcoded to 4.0 in apply_swof_table
    # regardless of state; this pins it as a real, effective parameter.
    assert "swof.oil_exponent" in parameters.PARAMETERS
    assert parameters.PARAMETERS["swof.oil_exponent"].group == "swof_endpoints"

    default_deck = parameters.apply_state(sample_deck, parameters.default_state())

    state = parameters.default_state()
    state["swof.oil_exponent"] = 2.0
    changed_deck = parameters.apply_state(sample_deck, state)

    default_rows = read_relperm_table(default_deck, "SWOF")
    changed_rows = read_relperm_table(changed_deck, "SWOF")

    # Krow at an interior Sw row must differ once the exponent changes;
    # Sw and Pcow (columns 0 and 3) must not.
    interior_index = 5
    assert default_rows[interior_index][0] == changed_rows[interior_index][0]
    assert default_rows[interior_index][3] == changed_rows[interior_index][3]
    assert default_rows[interior_index][2] != pytest.approx(changed_rows[interior_index][2])


def test_apply_rock_and_porosity_scales_every_layer(sample_deck: Deck) -> None:
    state = parameters.default_state()
    state["porosity.multiplier"] = 1.5
    state["rock.compressibility"] = 4.2e-06

    patched = parameters.apply_state(sample_deck, state)
    poro_values = [
        float(value)
        for value in re.findall(r"'PORO'\s+([\d.]+)\s+1 30  1 30  \d \d", patched.text)
    ]
    expected = [0.24, 0.22, 0.20, 0.23, 0.19]
    for actual, base in zip(poro_values, expected, strict=True):
        assert actual == pytest.approx(base * 1.5, abs=1e-3)
    assert "4.200E-06" in patched.text

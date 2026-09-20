"""Tunable history-matching parameters and how each one patches the deck.

Every function here rebuilds the affected part of the deck from a
pristine base `nagcsu.deck.Deck` rather than editing an
already-edited one, so a run's exact deck is always reproducible from
its logged parameter state alone (see `apply_state`). This
sidesteps the drift risk of applying the same multiplicative edit twice
by accident.

Parameter groups and their tuning order follow Stage D.1 of the Phase 2
Execution Plan: change one group, rerun, rescore, and only move to the
next group once the current one stops helping (Stage D.3).
"""

import dataclasses
import re
import typing

import numpy as np

from nagcsu import corey
from nagcsu.deck import NUMBER_PATTERN, Deck, patch_relperm_table, transform_within_block

CONNATE_WATER_SATURATION: typing.Final[float] = 0.13
"""Swc, the first saturation row of the SWOF table. Taken directly from
the EK6 anchor's relative-permeability table and not exposed as a
tunable parameter, since Stage D.1 only names Krw_max, Sorw and nw as
SWOF parameters worth tuning.
"""

MAX_OIL_RELATIVE_PERMEABILITY: typing.Final[float] = 0.80
"""kro_max, the oil relative permeability endpoint shared by the SWOF
and SGOF tables. Anchor-derived, not exposed as a tunable parameter.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Metadata for one tunable history-matching parameter."""

    name: str
    """Dotted key used in a parameter state dict and on the CLI, for
    example "aquifer.radius" or "sgof.sorg".
    """

    group: str
    """Tuning priority group this parameter belongs to. Must be one of
    `nagcsu.constants.TUNING_PRIORITY_ORDER`.
    """

    bounds: tuple[float, float]
    """Inclusive `(low, high)` range a search algorithm should stay within."""

    default: float
    """Value matching the deck as shipped (the EK6 anchor or its
    literature-assumed companion), used whenever a state dict omits
    this parameter.
    """

    description: str
    """One line explaining what this parameter controls, shown by
    `nagcsu match auto --list-parameters` and similar CLI help.
    """


PARAMETERS: typing.Final[dict[str, ParameterSpec]] = {
    spec.name: spec
    for spec in (
        ParameterSpec(
            name="aquifer.radius",
            group="aquifer",
            bounds=(8_000.0, 30_000.0),
            default=16_137.2,
            description="AQUCT outer aquifer radius (Rres, ft). The single biggest lever on pressure support.",
        ),
        ParameterSpec(
            name="aquifer.permeability",
            group="aquifer",
            bounds=(100.0, 1_500.0),
            default=566.0,
            description="AQUCT aquifer permeability (md).",
        ),
        ParameterSpec(
            name="aquifer.thickness",
            group="aquifer",
            bounds=(20.0, 200.0),
            default=77.0,
            description="AQUCT aquifer thickness (ft).",
        ),
        ParameterSpec(
            name="aquifer.encroachment_angle",
            group="aquifer",
            bounds=(10.0, 360.0),
            default=56.0,
            description="AQUCT encroachment angle (degrees).",
        ),
        ParameterSpec(
            name="permeability.areal_contrast",
            group="permeability_multiplier",
            bounds=(0.0, 0.5),
            default=0.15,
            description=(
                "Areal PERMX/PERMY multiplier contrast: west block gets 1+delta, "
                "east block gets 1-delta."
            ),
        ),
        ParameterSpec(
            name="sgof.sorg",
            group="sgof_shape",
            bounds=(0.05, 0.30),
            default=0.15,
            description="Residual oil saturation in the gas-oil system (SGOF), a literature assumption.",
        ),
        ParameterSpec(
            name="sgof.oil_exponent",
            group="sgof_shape",
            bounds=(1.0, 4.0),
            default=2.0,
            description="Oil-in-gas Corey exponent (SGOF), a literature assumption.",
        ),
        ParameterSpec(
            name="sgof.critical_gas_saturation",
            group="sgof_endpoints",
            bounds=(0.01, 0.10),
            default=0.04,
            description="Sgc (SGOF), taken directly from the EK6 anchor. Touch last.",
        ),
        ParameterSpec(
            name="sgof.max_gas_relative_permeability",
            group="sgof_endpoints",
            bounds=(0.5, 1.0),
            default=0.77,
            description="krg_max (SGOF), taken directly from the EK6 anchor. Touch last.",
        ),
        ParameterSpec(
            name="sgof.gas_exponent",
            group="sgof_endpoints",
            bounds=(1.0, 5.0),
            default=3.0,
            description="Gas Corey exponent ng (SGOF), taken directly from the EK6 anchor. Touch last.",
        ),
        ParameterSpec(
            name="swof.max_water_relative_permeability",
            group="swof_endpoints",
            bounds=(0.5, 1.0),
            default=0.78,
            description="krw_max (SWOF), from the EK6 anchor. Touch only if aquifer/perm/SGOF don't close the water-cut gap.",
        ),
        ParameterSpec(
            name="swof.residual_oil_saturation",
            group="swof_endpoints",
            bounds=(0.05, 0.30),
            default=0.12,
            description="Sorw (SWOF), from the EK6 anchor. Touch only if aquifer/perm/SGOF don't close the water-cut gap.",
        ),
        ParameterSpec(
            name="swof.water_exponent",
            group="swof_endpoints",
            bounds=(1.0, 6.0),
            default=3.8,
            description="Water Corey exponent nw (SWOF), from the EK6 anchor. Touch only if aquifer/perm/SGOF don't close the water-cut gap.",
        ),
        ParameterSpec(
            name="swof.oil_exponent",
            group="swof_endpoints",
            bounds=(1.0, 6.0),
            default=4.0,
            description=(
                "Water-oil Corey exponent no (SWOF), from the EK6 anchor. Not named in the "
                "Execution Plan's Stage D.1 table alongside Krw_max/Sorw/nw, but exposed here "
                "for consistency with sgof.oil_exponent rather than left permanently fixed."
            ),
        ),
        ParameterSpec(
            name="rock.compressibility",
            group="rock_and_porosity",
            bounds=(1.0e-6, 1.0e-5),
            default=3.577e-06,
            description="Rock compressibility (1/psi) at the ROCK keyword's reference pressure. Last resort.",
        ),
        ParameterSpec(
            name="porosity.multiplier",
            group="rock_and_porosity",
            bounds=(0.7, 1.3),
            default=1.0,
            description="Uniform scale factor applied to every layer's PORO value from the base deck. Last resort.",
        ),
    )
}
"""
Every tunable parameter, keyed by its dotted name. See `nagcsu.constants.TUNING_PRIORITY_ORDER` for the 
groups' tuning priority.
"""


def parameters_in_group(group: str) -> list[ParameterSpec]:
    """Return the `ParameterSpec` entries belonging to `group`, in a stable order."""
    return [spec for spec in PARAMETERS.values() if spec.group == group]


def default_state() -> dict[str, float]:
    """Return the parameter state matching the deck exactly as shipped."""
    return {name: spec.default for name, spec in PARAMETERS.items()}


def resolve_state(state: dict[str, float]) -> dict[str, float]:
    """Fill in any parameter missing from `state` with its default value."""
    return {name: state.get(name, spec.default) for name, spec in PARAMETERS.items()}


def apply_state(base_deck: Deck, state: dict[str, float]) -> Deck:
    """Return the deck that `state` produces, starting from `base_deck`.

    `base_deck` should always be the pristine, as-shipped deck rather
    than a previously patched one; every group's patch below reads the
    values it needs straight out of `state` (falling back to each
    parameter's default), so applying the same state twice always
    produces the same deck.
    """
    resolved = resolve_state(state)
    deck = apply_aquifer(base_deck, resolved)
    deck = apply_permeability_multiplier(deck, resolved)
    deck = apply_sgof_table(deck, resolved)
    deck = apply_swof_table(deck, resolved)
    deck = apply_rock_and_porosity(deck, resolved)
    return deck


def apply_aquifer(deck: Deck, state: dict[str, float]) -> Deck:
    """Patch the AQUCT record's permeability, radius, thickness and angle."""
    pattern = (
        r"^(?P<lead>\s*)(?P<aqid>" + NUMBER_PATTERN + r")(?P<s1>\s+)"
        r"(?P<depth>" + NUMBER_PATTERN + r")(?P<s2>\s+)"
        r"(?P<pi>" + NUMBER_PATTERN + r")(?P<s3>\s+)"
        r"(?P<perm>" + NUMBER_PATTERN + r")(?P<s4>\s+)"
        r"(?P<poro>" + NUMBER_PATTERN + r")(?P<s5>\s+)"
        r"(?P<ctotal>" + NUMBER_PATTERN + r")(?P<s6>\s+)"
        r"(?P<rres>" + NUMBER_PATTERN + r")(?P<s7>\s+)"
        r"(?P<thick>" + NUMBER_PATTERN + r")(?P<s8>\s+)"
        r"(?P<angle>" + NUMBER_PATTERN + r")(?P<tail>\s+\S.*/\s*)$"
    )

    def transform(match: re.Match) -> str:
        return (
            f"{match['lead']}{match['aqid']}{match['s1']}"
            f"{match['depth']}{match['s2']}"
            f"{match['pi']}{match['s3']}"
            f"{state['aquifer.permeability']:.2f}{match['s4']}"
            f"{match['poro']}{match['s5']}"
            f"{match['ctotal']}{match['s6']}"
            f"{state['aquifer.radius']:.2f}{match['s7']}"
            f"{state['aquifer.thickness']:.2f}{match['s8']}"
            f"{state['aquifer.encroachment_angle']:.2f}{match['tail']}"
        )

    return transform_within_block(
        deck, "AQUCT", pattern, transform, expected_count=1, flags=re.MULTILINE
    )


def apply_permeability_multiplier(deck: Deck, state: dict[str, float]) -> Deck:
    """Patch the MULTIPLY block's areal PERMX/PERMY contrast.

    The west block (columns 1-10) is set to `1 + delta`, the east block
    (columns 21-30) to `1 - delta`, matching how the base deck's
    "1.15 / 0.85" contrast was built. `delta` is
    `state["permeability.areal_contrast"]`.
    """
    delta = state["permeability.areal_contrast"]
    high, low = 1.0 + delta, 1.0 - delta

    def patch_side(deck_in: Deck, property_name: str, box: str, value: float) -> Deck:
        pattern = rf"('{property_name}'\s+){NUMBER_PATTERN}(\s+{box}\s+1\s+30\s+1\s+5\s*/)"
        return deck_in.replace_once(pattern, rf"\g<1>{value:.4f}\g<2>")

    deck = patch_side(deck, "PERMX", r"1\s+10", high)
    deck = patch_side(deck, "PERMX", r"21\s+30", low)
    deck = patch_side(deck, "PERMY", r"1\s+10", high)
    deck = patch_side(deck, "PERMY", r"21\s+30", low)
    return deck


def apply_sgof_table(deck: Deck, state: dict[str, float]) -> Deck:
    """Regenerate the SGOF table's Krg/Krog columns from the Corey model."""

    def transform(
        rows: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        gas_saturation = np.array([row[0] for row in rows])
        capillary_pressure = [row[3] for row in rows]
        krg = corey.gas_relative_permeability(
            gas_saturation,
            critical_gas_saturation=state["sgof.critical_gas_saturation"],
            connate_water_saturation=CONNATE_WATER_SATURATION,
            residual_oil_saturation=state["sgof.sorg"],
            max_gas_relative_permeability=state["sgof.max_gas_relative_permeability"],
            gas_corey_exponent=state["sgof.gas_exponent"],
        )
        krog = corey.oil_relative_permeability_in_gas(
            gas_saturation,
            critical_gas_saturation=state["sgof.critical_gas_saturation"],
            connate_water_saturation=CONNATE_WATER_SATURATION,
            residual_oil_saturation=state["sgof.sorg"],
            max_oil_relative_permeability=MAX_OIL_RELATIVE_PERMEABILITY,
            oil_corey_exponent=state["sgof.oil_exponent"],
        )
        return list(
            zip(
                gas_saturation.tolist(),
                krg.tolist(),
                krog.tolist(),
                capillary_pressure,
                strict=True,
            )
        )

    return patch_relperm_table(deck, "SGOF", transform)


def apply_swof_table(deck: Deck, state: dict[str, float]) -> Deck:
    """Regenerate the SWOF table's Krw/Krow columns from the Corey model."""

    def transform(
        rows: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        water_saturation = np.array([row[0] for row in rows])
        capillary_pressure = [row[3] for row in rows]
        krw = corey.water_relative_permeability(
            water_saturation,
            connate_water_saturation=CONNATE_WATER_SATURATION,
            residual_oil_saturation=state["swof.residual_oil_saturation"],
            max_water_relative_permeability=state["swof.max_water_relative_permeability"],
            water_corey_exponent=state["swof.water_exponent"],
        )
        krow = corey.oil_relative_permeability_in_water(
            water_saturation,
            connate_water_saturation=CONNATE_WATER_SATURATION,
            residual_oil_saturation=state["swof.residual_oil_saturation"],
            max_oil_relative_permeability=MAX_OIL_RELATIVE_PERMEABILITY,
            oil_corey_exponent=state["swof.oil_exponent"],
        )
        return list(
            zip(
                water_saturation.tolist(),
                krw.tolist(),
                krow.tolist(),
                capillary_pressure,
                strict=True,
            )
        )

    return patch_relperm_table(deck, "SWOF", transform)


def apply_rock_and_porosity(deck: Deck, state: dict[str, float]) -> Deck:
    """Patch rock compressibility and scale every layer's PORO value."""
    deck = deck.replace_once(
        r"(ROCK\s*\n\s*" + NUMBER_PATTERN + r"\s+)(" + NUMBER_PATTERN + r")(\s*/)",
        rf"\g<1>{state['rock.compressibility']:.3E}\g<3>",
    )
    multiplier = state["porosity.multiplier"]

    def transform(match: re.Match) -> str:
        original = float(match["value"])
        return f"{match['lead']}{original * multiplier:.4f}{match['tail']}"

    pattern = (
        r"(?P<lead>'PORO'\s+)(?P<value>" + NUMBER_PATTERN + r")"
        r"(?P<tail>\s+1\s+30\s+1\s+30\s+\d\s+\d\s*/)"
    )
    return deck.transform_each(pattern, transform, expected_count=5)

"""Shared constants for the UGH-1 composite sector model.

These are defaults, not hard requirements: every value here can be
overridden from a project config file (see `nagcsu.config`) or from
the CLI, since a future deck revision may add or rename wells, or the
team may want to test a different objective weighting.
"""

import typing

PRODUCER_WELLS: typing.Final[tuple[str, ...]] = (
    "AFIESERE",
    "ERIEMU",
    "EVWRENI",
    "OWEH",
    "OLOMORO",
    "KOKORI",
    "ORONI",
    "UZERE",
)
"""Well names as spelled in WELSPECS in the UGH-1 composite deck."""

INJECTOR_WELL: typing.Final[str] = "INJ-1"
"""Name of the storage/EOR injector pre-wired (shut) in the deck."""

DEFAULT_OBJECTIVE_WEIGHTS: typing.Final[dict[str, float]] = {
    "pressure": 0.50,
    "watercut": 0.35,
    "gor": 0.15,
}
"""Starting NRMSE weights for the combined objective J, per Stage C.2 of
the Phase 2 Execution Plan. Pressure is weighted highest because it is
the most direct read on volumetric support, the thing the calibration is
least sure of.
"""

DEFAULT_TARGET_J: typing.Final[float] = 0.125
"""Midpoint of the 0.10 to 0.15 target range for J from Stage C.4. Below
this, further tuning is more likely to overfit the single synthetic
anchor than to genuinely improve the match.
"""

DEFAULT_CUSHION_GAS_FRACTION: typing.Final[float] = 0.60
"""Starting cushion gas fraction for a depleted-reservoir storage scheme,
from the 50 to 70 percent range given in Stage F.2. The remaining
fraction is working gas, cycled in and out each period.
"""

TUNING_PRIORITY_ORDER: typing.Final[tuple[str, ...]] = (
    "aquifer",
    "permeability_multiplier",
    "sgof_shape",
    "sgof_endpoints",
    "swof_endpoints",
    "rock_and_porosity",
)
"""Parameter group tuning order from Stage D.1. Auto mode changes one
group at a time in this order and only moves on to a later group once
earlier groups stop reducing the objective, so a run that improves the
fit can always be attributed to a single change.
"""

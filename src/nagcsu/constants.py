"""Shared constants for the UGH-1 composite sector model.

These are defaults, not hard requirements as every value here can be
overridden from a project config file (see `nagcsu.config`) or from
the CLI.
"""

import pathlib
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
"""Starting NRMSE weights for the combined objective J. Pressure is
weighted highest because it is the most direct read on volumetric
support, the thing the calibration is least sure of.
"""

DEFAULT_TARGET_J: typing.Final[float] = 0.125
"""Midpoint of a 0.10 to 0.15 target range for J. Below this, further
tuning is more likely to overfit the single synthetic anchor than to
genuinely improve the match.
"""

DEFAULT_CUSHION_GAS_FRACTION: typing.Final[float] = 0.60
"""Starting cushion gas fraction for a depleted-reservoir storage scheme,
within the usual 50 to 70 percent range. The remaining fraction is
working gas, cycled in and out each period.
"""

TUNING_PRIORITY_ORDER: typing.Final[tuple[str, ...]] = (
    "aquifer",
    "permeability_multiplier",
    "sgof_shape",
    "sgof_endpoints",
    "swof_endpoints",
    "rock_and_porosity",
)
"""Parameter group tuning order. Auto mode changes one
group at a time in this order and only moves on to a later group once
earlier groups stop reducing the objective, so a run that improves the
fit can always be attributed to a single change.
"""

DEFAULT_DECK_PATH = pathlib.Path("Data/NigerDelta UGH1 Composite Field.DATA")
"""Reservoir deck file for the UGH-1 composite model used by default when
no explicit deck path is configured.
"""

DEFAULT_HISTORY_PATH = pathlib.Path("Data/Monthly Production Data.xlsx")
"""Production-history workbook used to compare model output against observed
field performance when a history file is not supplied.
"""

DEFAULT_OUTPUT_DIR = pathlib.Path("runs")
"""Default directory under which run artifacts, reports, and generated
outputs are written.
"""

DEFAULT_LEDGER_PATH = DEFAULT_OUTPUT_DIR / "ledger.json"
"""Ledger file recording run metadata and provenance for generated model
runs.
"""

DEFAULT_CONFIG_FILE: typing.Final[str] = "nagcsu.yaml"
"""Filename a bare `nagcsu <command>` looks for in the current directory."""

DEFAULT_ROOT_DIR = pathlib.Path(".")
"""Default project root used when resolving relative paths for config and
input files.
"""

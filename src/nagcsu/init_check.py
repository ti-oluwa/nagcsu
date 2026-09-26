"""Checking a run's initial fluid saturation against the static model's own expectations.

This module reads the very first `SWAT` array written to a run's
`.UNRST` file, the pre-production, EQUIL-derived state before any
`DATES` step, and compares it against the deck's own connate water
saturation (Swc, the first row of its SWOF table), so a bad EQUIL
contact depth, PVT/density input, or capillary-pressure table is caught
in seconds instead of after dozens of expensive history-match trials.
"""

import dataclasses
import pathlib

import numpy as np
import numpy.typing as npt
import resfo


@dataclasses.dataclass(frozen=True, slots=True)
class InitialSaturationReport:
    """Water saturation at a run's first (pre-production) `SWAT` array."""

    case_basename: pathlib.Path
    """Path (without extension) the `.UNRST` file was read from."""

    cell_count: int
    """Number of active grid cells the initial `SWAT` array covers."""

    min_water_saturation: float
    """Minimum initial water saturation over every active cell."""

    mean_water_saturation: float
    """Mean initial water saturation over every active cell."""

    max_water_saturation: float
    """Maximum initial water saturation over every active cell."""

    expected_water_saturation: float
    """The static model's own connate water saturation (Swc), which
    every active cell above the water-oil contact should initialize at
    or near, normally `nagcsu.parameters.CONNATE_WATER_SATURATION`.
    """

    tolerance: float
    """How far `mean_water_saturation` is allowed to sit above
    `expected_water_saturation` before `is_plausible` reports `False`.
    """

    @property
    def is_plausible(self) -> bool:
        """Whether the initial state looks like a fresh reservoir.

        `False` means the model has already started at or near residual
        oil saturation before a single day of production, something no
        dynamic history-match parameter (aquifer, relperm shape, and so
        on) can fix; the static model itself (EQUIL contact depths, the
        density/PVT inputs behind the capillary-gravity balance, or the
        SWOF `Pcow` column) needs to change instead.
        """
        return self.mean_water_saturation <= self.expected_water_saturation + self.tolerance


def read_initial_water_saturation(case_basename: pathlib.Path | str) -> npt.NDArray[np.float64]:
    """Return the first `SWAT` array written to a run's `.UNRST` file.

    A run's `.UNRST` always writes the EQUIL-derived initial state, the
    pre-production condition at report step 0, before any other `SWAT`
    array, so the first one found is always this initial state.

    :raises FileNotFoundError: if the `.UNRST` file does not exist.
    :raises ValueError: if the file has no `SWAT` array at all, for
        example an oil/water-only setup that never wrote one; check the
        deck's `RUNSPEC` phases first.
    """
    unrst_path = pathlib.Path(case_basename).with_suffix(".UNRST")
    if not unrst_path.exists():
        raise FileNotFoundError(unrst_path)

    for entry in resfo.lazy_read(unrst_path):
        if entry.read_keyword().strip() == "SWAT":
            return np.asarray(entry.read_array(), dtype=np.float64)

    raise ValueError(f"No SWAT array found in {unrst_path}")


def check_initial_water_saturation(
    case_basename: pathlib.Path | str,
    *,
    expected_water_saturation: float,
    tolerance: float = 0.05,
) -> InitialSaturationReport:
    """Build an `InitialSaturationReport` for one run.

    :param expected_water_saturation: The deck's connate water
        saturation (Swc), normally
        `nagcsu.parameters.CONNATE_WATER_SATURATION`.
    :param tolerance: See `InitialSaturationReport.tolerance`.
    :raises FileNotFoundError: if the run's `.UNRST` file does not exist.
    :raises ValueError: if the run's `.UNRST` file has no `SWAT` array.
    """
    case_basename = pathlib.Path(case_basename)
    swat = read_initial_water_saturation(case_basename)
    return InitialSaturationReport(
        case_basename=case_basename,
        cell_count=swat.size,
        min_water_saturation=float(swat.min()),
        mean_water_saturation=float(swat.mean()),
        max_water_saturation=float(swat.max()),
        expected_water_saturation=expected_water_saturation,
        tolerance=tolerance,
    )

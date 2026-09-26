"""Scoring how well a simulated summary matches the observed history.

Implements Stage C of the Phase 2 Execution Plan: normalized RMSE per
scored vector, combined into one weighted objective J. Only pressure,
water cut and GOR are ever scored, never the rate vectors, since the
rates are a prescribed input to the deck (`WCONPROD`) rather than
something OPM Flow predicts; a "perfect" rate match proves nothing.
"""

import dataclasses

import numpy as np
import numpy.typing as npt
import pandas

from nagcsu.exceptions import HistoryAlignmentError

SCORED_FIELD_VECTORS: dict[str, str] = {
    "pressure": "FPR",
    "watercut": "FWCT",
    "gor": "FGOR",
}
"""Mapping from a scored quantity's short name to its field-total
res2df summary vector name.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class VectorScore:
    """NRMSE for one scored vector, plus how many points it was computed over."""

    name: str
    """Short name of the scored quantity, for example "pressure"."""

    nrmse: float
    """Range-normalized RMSE: `RMSE / (obs_max - obs_min)`. Dimensionless,
    so it can be combined with other vectors' NRMSE in a weighted sum.
    """

    point_count: int
    """Number of aligned (date-matched) points the score was computed over."""


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """Combined mismatch score J and its per-vector components."""

    j: float
    """Weighted sum of the per-vector NRMSE values. Lower is a better match."""

    vector_scores: dict[str, VectorScore]
    """Per-vector `VectorScore`, keyed the same way as `SCORED_FIELD_VECTORS`."""

    weights: dict[str, float]
    """Weights actually used for this result, echoed back for the run record."""


def nrmse(
    simulated: npt.NDArray[np.float64] | pandas.Series,
    observed: npt.NDArray[np.float64] | pandas.Series,
) -> float:
    """Range-normalized root-mean-square error between two aligned series.

    :returns: `RMSE / (observed.max() - observed.min())`. Falls back to
        raw RMSE if the observed series has zero range (a constant
        history), which avoids a division by zero for a degenerate case
        rather than raising.
    """
    simulated_array = np.asarray(simulated, dtype=np.float64)
    observed_array = np.asarray(observed, dtype=np.float64)
    rmse = np.sqrt(np.mean((simulated_array - observed_array) ** 2))
    span = observed_array.max() - observed_array.min()
    return float(rmse / span) if span > 0 else float(rmse)


def score(
    simulated: pandas.DataFrame,
    observed: pandas.DataFrame,
    *,
    weights: dict[str, float],
    date_column: str = "DATE",
) -> ObjectiveResult:
    """Compute the combined objective J between a simulated and observed frame.

    Both frames must have a `date_column` plus one column per entry in
    `SCORED_FIELD_VECTORS`; they are inner-joined on date before scoring,
    so only dates present in both contribute.

    :param weights: NRMSE weight per entry in `SCORED_FIELD_VECTORS`,
        typically `nagcsu.config.ObjectiveConfig.weights`.
    :raises nagcsu.exceptions.HistoryAlignmentError: if the two frames
        share no common dates, or if a required column is missing from
        either frame.
    """
    print(simulated.head())
    print(observed.head())
    missing_columns = [
        column
        for column in (date_column, *SCORED_FIELD_VECTORS.values())
        if column not in simulated.columns or column not in observed.columns
    ]
    if missing_columns:
        raise HistoryAlignmentError(
            f"Simulated and observed frames must both have columns "
            f"{[date_column, *SCORED_FIELD_VECTORS.values()]}; missing or "
            f"mismatched: {missing_columns}"
        )

    merged = simulated.merge(observed, on=date_column, suffixes=("_sim", "_obs"))
    if merged.empty:
        raise HistoryAlignmentError(
            f"No overlapping {date_column!r} values between simulated and observed frames"
        )

    vector_scores: dict[str, VectorScore] = {}
    weighted_total = 0.0
    for name, vector in SCORED_FIELD_VECTORS.items():
        vector_score = nrmse(merged[f"{vector}_sim"], merged[f"{vector}_obs"])
        vector_scores[name] = VectorScore(name=name, nrmse=vector_score, point_count=len(merged))
        weighted_total += weights.get(name, 0.0) * vector_score

    return ObjectiveResult(j=weighted_total, vector_scores=vector_scores, weights=dict(weights))

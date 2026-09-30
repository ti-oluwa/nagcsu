"""Scoring how well a simulated summary matches the observed history.

Normalized RMSE per scored vector, combined into one weighted objective J. Only pressure,
water cut and GOR are ever scored, never the rate vectors, since the
rates are a prescribed input to the deck (`WCONPROD`) rather than
something OPM Flow predicts; a "perfect" rate match proves nothing.
"""

import dataclasses
import typing

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


WELL_AGGREGATE_VECTORS: dict[str, str] = {
    "wells_watercut": "WWCT",
    "wells_gor": "WGOR",
}
"""Weight names for per-well scoring, and the summary vector each one
averages over the selected wells. `wells_watercut` is the mean of the
selected wells' water-cut NRMSE, so adding wells does not inflate its
influence on J: one weight covers the whole set."""


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """Combined mismatch score J and its per-vector components."""

    j: float
    """Weighted sum of the per-vector NRMSE values. Lower is a better match."""

    vector_scores: dict[str, VectorScore]
    """Per-vector `VectorScore`, keyed the same way as `SCORED_FIELD_VECTORS`."""

    weights: dict[str, float]
    """Weights actually used for this result, echoed back for the run record."""

    well_scores: dict[str, VectorScore] = dataclasses.field(default_factory=dict)
    """Per-well NRMSE keyed like `WWCT:AFIESERE`, for every well whose
    column exists in both frames, whether or not that well is part of `J`.
    Diagnostic for reports and plots; only wells in `wells` feed `J`.
    """

    scored_wells: tuple[str, ...] = ()
    """Wells that fed the `wells_watercut` and `wells_gor` terms of `J`."""


def compute_nrmse(
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
    nrmse_ceiling: float | None = None,
    wells: typing.Sequence[str] = (),
) -> ObjectiveResult:
    """Compute the combined objective `J` between a simulated and observed frame.

    Both frames must have a date column matching `date_column` without
    regard to letter case, plus one column per entry in
    `SCORED_FIELD_VECTORS`; they are inner-joined on date before scoring,
    so only dates present in both contribute.

    :param weights: NRMSE weight per entry in `SCORED_FIELD_VECTORS`,
        typically `nagcsu.config.ObjectiveConfig.weights`.
    :param nrmse_ceiling: If given, each vector's NRMSE is clipped to
        this value before being weighted into `J`. A vector whose
        history sits in a narrow range (GOR is the usual case) can swing
        to an enormous NRMSE the moment the simulator's own output
        diverges even a little, for example a solution-gas reservoir
        producing free gas once pressure drops below bubble point. Left
        unclipped, that one vector silently drowns out real signal from
        the other two in every search strategy, since they only ever see
        the combined `J`. `None` (the default) preserves the old,
        unclipped behavior; a run's own `VectorScore.nrmse` is always the
        raw value either way, so `nagcsu sanity check-init` and manual
        inspection still see the true, unclipped mismatch even when a
        ceiling is set for the search itself.
    :param wells: Wells whose water cut and GOR feed the
        `wells_watercut` and `wells_gor` weights. Empty means field
        totals only. Per-well NRMSE is also computed, as a diagnostic
        in `ObjectiveResult.well_scores`, for every other well whose
        columns are present in both frames.
    :raises nagcsu.exceptions.HistoryAlignmentError: if the two frames
        share no common dates, if a required column is missing from
        either frame, or if a selected well has no usable data.
    """

    def matching_columns(frame: pandas.DataFrame, name: str) -> list[str]:
        return [
            column
            for column in frame.columns
            if isinstance(column, str) and column.strip().casefold() == name.strip().casefold()
        ]

    simulated_date_columns = matching_columns(simulated, date_column)
    observed_date_columns = matching_columns(observed, date_column)
    missing_columns = [
        column
        for column in SCORED_FIELD_VECTORS.values()
        if column not in simulated.columns or column not in observed.columns
    ]
    if len(simulated_date_columns) != 1 or len(observed_date_columns) != 1:
        missing_columns.insert(0, date_column)
    if missing_columns:
        raise HistoryAlignmentError(
            f"Simulated and observed frames must both have a date column matching "
            f"{date_column!r} (case-insensitively) and columns "
            f"{list(SCORED_FIELD_VECTORS.values())}; missing or "
            f"mismatched: {missing_columns}"
        )

    merged = simulated.merge(
        observed,
        left_on=simulated_date_columns[0],
        right_on=observed_date_columns[0],
        suffixes=("_sim", "_obs"),
    )
    if merged.empty:
        raise HistoryAlignmentError(
            f"No overlapping {date_column!r} values between simulated and observed frames"
        )

    vector_scores: dict[str, VectorScore] = {}
    weighted_total = 0.0
    for name, vector in SCORED_FIELD_VECTORS.items():
        raw_nrmse = compute_nrmse(merged[f"{vector}_sim"], merged[f"{vector}_obs"])
        vector_scores[name] = VectorScore(name=name, nrmse=raw_nrmse, point_count=len(merged))
        clamped_nrmse = min(raw_nrmse, nrmse_ceiling) if nrmse_ceiling is not None else raw_nrmse
        weighted_total += weights.get(name, 0.0) * clamped_nrmse

    well_scores: dict[str, VectorScore] = {}
    for column in simulated.columns:
        if not isinstance(column, str) or not column.startswith(("WWCT:", "WGOR:")):
            continue
        if f"{column}_sim" not in merged.columns or f"{column}_obs" not in merged.columns:
            continue
        pair = merged[[f"{column}_sim", f"{column}_obs"]].dropna()
        if pair.empty:
            continue
        well_scores[column] = VectorScore(
            name=column,
            nrmse=compute_nrmse(pair[f"{column}_sim"], pair[f"{column}_obs"]),
            point_count=len(pair),
        )

    selected = tuple(dict.fromkeys(well.strip().upper() for well in wells))
    aggregate_weight_used = any(weights.get(name, 0.0) > 0 for name in WELL_AGGREGATE_VECTORS)
    if aggregate_weight_used and not selected:
        raise HistoryAlignmentError(
            "Weights for per-well vectors (wells_watercut / wells_gor) are set but no wells "
            "are selected; set `objective.wells` in the config or pass --wells."
        )
    if selected:
        for aggregate_name, prefix in WELL_AGGREGATE_VECTORS.items():
            missing = [well for well in selected if f"{prefix}:{well}" not in well_scores]
            if weights.get(aggregate_name, 0.0) > 0 and missing:
                raise HistoryAlignmentError(
                    f"No usable {prefix} history or simulation for well(s) {missing}. Check the "
                    f"well names against the deck's WELSPECS and the history file's well column."
                )
            members = [
                well_scores[f"{prefix}:{well}"]
                for well in selected
                if f"{prefix}:{well}" in well_scores
            ]
            if not members:
                continue
            mean_nrmse = float(np.mean([member.nrmse for member in members]))
            vector_scores[aggregate_name] = VectorScore(
                name=aggregate_name,
                nrmse=mean_nrmse,
                point_count=int(sum(member.point_count for member in members)),
            )
            clamped = min(mean_nrmse, nrmse_ceiling) if nrmse_ceiling is not None else mean_nrmse
            weighted_total += weights.get(aggregate_name, 0.0) * clamped

    return ObjectiveResult(
        j=weighted_total,
        vector_scores=vector_scores,
        weights=dict(weights),
        well_scores=well_scores,
        scored_wells=selected,
    )

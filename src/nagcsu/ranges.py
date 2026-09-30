"""Suggesting a search range for each parameter from what tuning already tried.

The ledger holds every single-parameter trial with its J. Around a
parameter's best value there is usually a "basin" of values that all score
close to the best; the useful next search range covers that basin with some
margin, not the full registered bounds and not a blind widening of one side.
The registered bounds are a recommendation: a suggestion may go past them when
the trials say the best value lies there, and is limited only by the hard
physical limits in `nagcsu.parameters.PHYSICAL_LIMITS`.

This is a heuristic on logged trials, not a model fit. Each trial was run
with the other parameters at whatever values the search had reached at that
moment, so treat a suggestion as a good place to start the next batch, then
let the new trials confirm it.
"""

import dataclasses
import math
import typing

from nagcsu import ledger, parameters

NEAR_BEST_FRACTION: typing.Final[float] = 0.25
"""A tried value belongs to the basin when its J is within this fraction of
the parameter's J span above its best J."""

MIN_DISTINCT_VALUES: typing.Final[int] = 4
"""Fewer distinct tried values than this and no range is suggested."""

FLAT_J_SPAN: typing.Final[float] = 0.005
"""A parameter whose J moved less than this across everything tried is flat."""

OPEN_SIDE_EXTENSION: typing.Final[float] = 0.5
"""When the best value sits at the edge of what was tried, the suggested
range extends past that edge by this fraction of the tried range."""


@dataclasses.dataclass(frozen=True, slots=True)
class RangeSuggestion:
    """A suggested starting range for one parameter, with the evidence behind it."""

    parameter: str
    group: str | None
    trials: int
    """Scored single-parameter trials used."""

    registered_bounds: tuple[float, float]
    """Bounds from `nagcsu.parameters`."""

    tried_low: float
    tried_high: float
    best_value: float
    best_j: float
    j_span: float
    """Worst minus best J across the trials used."""

    status: str
    """One of: "bracketed" (worse values found on both sides of the best),
    "open_low" / "open_high" / "open_both" (the best sits at the edge of what
    was tried, so the optimum may lie beyond it), "flat" (J barely moved),
    "insufficient" (too few distinct values tried).
    """

    low: float | None
    high: float | None
    """Suggested range, or `None` for flat or insufficient parameters."""

    note: str
    """One line saying why."""

    def range_flag(self) -> str | None:
        """The `--range NAME=LOW:HIGH` text for `match auto`, if a range is suggested."""
        if self.low is None or self.high is None:
            return None
        return f"--range {self.parameter}={self.low:.6g}:{self.high:.6g}"


def suggest_ranges(
    records: list[ledger.RunRecord],
    *,
    near_best_fraction: float = NEAR_BEST_FRACTION,
    min_distinct_values: int = MIN_DISTINCT_VALUES,
    flat_j_span: float = FLAT_J_SPAN,
) -> list[RangeSuggestion]:
    """Suggest a starting range per parameter from the trials in `records`.

    Only scored trials that moved exactly one parameter are used, because a
    trial that moved several cannot be attributed to any one of them.
    Results are ordered by J span, the most responsive parameter first.
    """
    points: dict[str, list[tuple[float, float]]] = {}
    groups: dict[str, str | None] = {}
    for record in records:
        if not record.tuned_parameters or len(record.tuned_parameters) != 1:
            continue
        name = record.tuned_parameters[0]
        value = (record.tuned_values or {}).get(name)
        if value is None or record.j is None or not math.isfinite(record.j):
            continue
        points.setdefault(name, []).append((value, record.j))
        groups[name] = record.group or groups.get(name)

    suggestions = [
        suggest_range(
            name,
            groups.get(name),
            sorted(pairs),
            near_best_fraction=near_best_fraction,
            min_distinct_values=min_distinct_values,
            flat_j_span=flat_j_span,
        )
        for name, pairs in points.items()
        if name in parameters.PARAMETERS
    ]
    suggestions.sort(key=lambda suggestion: -suggestion.j_span)
    return suggestions


def suggest_range(
    name: str,
    group: str | None,
    points: list[tuple[float, float]],
    *,
    near_best_fraction: float,
    min_distinct_values: int,
    flat_j_span: float,
) -> RangeSuggestion:
    registered_low, registered_high = parameters.PARAMETERS[name].bounds
    values = [value for value, _ in points]
    js = [j for _, j in points]
    tried_low, tried_high = min(values), max(values)
    best_index = min(range(len(points)), key=lambda index: js[index])
    best_value, best_j = points[best_index]
    j_span = max(js) - best_j

    def build(status: str, low: float | None, high: float | None, note: str) -> RangeSuggestion:
        return RangeSuggestion(
            parameter=name,
            group=group,
            trials=len(points),
            registered_bounds=(registered_low, registered_high),
            tried_low=tried_low,
            tried_high=tried_high,
            best_value=best_value,
            best_j=best_j,
            j_span=j_span,
            status=status,
            low=low,
            high=high,
            note=note,
        )

    if len(set(values)) < min_distinct_values:
        return build(
            "insufficient",
            None,
            None,
            f"only {len(set(values))} distinct value(s) tried; sweep it before trusting a range",
        )
    if j_span < flat_j_span:
        return build(
            "flat",
            None,
            None,
            f"J moved only {j_span:.4f} across everything tried; hold it at {best_value:.6g}",
        )

    threshold = best_j + near_best_fraction * j_span
    low_index = high_index = best_index
    while low_index > 0 and js[low_index - 1] <= threshold:
        low_index -= 1
    while high_index < len(points) - 1 and js[high_index + 1] <= threshold:
        high_index += 1

    basin_low, basin_high = points[low_index][0], points[high_index][0]
    tried_span = tried_high - tried_low
    limit_low, limit_high = parameters.get_physical_limits(name)

    notes: list[str] = []
    if low_index > 0:
        low = (basin_low + points[low_index - 1][0]) / 2.0
        open_low = False
    else:
        open_low = True
        low = basin_low - OPEN_SIDE_EXTENSION * tried_span
        if low <= limit_low:
            low = limit_low
            notes.append(f"reaches the physical lower limit {limit_low:.6g}")

    if high_index < len(points) - 1:
        high = (basin_high + points[high_index + 1][0]) / 2.0
        open_high = False
    else:
        open_high = True
        high = basin_high + OPEN_SIDE_EXTENSION * tried_span
        if high >= limit_high:
            high = limit_high
            notes.append(f"reaches the physical upper limit {limit_high:.6g}")

    low = max(limit_low, low)
    high = min(limit_high, high)
    if high <= low:
        low, high = max(limit_low, tried_low), min(limit_high, tried_high)
    if low < registered_low or high > registered_high:
        notes.append(
            f"goes past the registered bounds {registered_low:.6g} to {registered_high:.6g}; "
            f"allowed when passed with --range"
        )

    status = (
        "open_both"
        if open_low and open_high
        else "open_low"
        if open_low
        else "open_high"
        if open_high
        else "bracketed"
    )
    if status == "bracketed":
        notes.insert(0, f"worse values found on both sides of {best_value:.6g}")
    else:
        side = {"open_low": "below", "open_high": "above", "open_both": "on both sides of"}[status]
        notes.insert(0, f"best sits at the edge of what was tried; the optimum may lie {side} it")
    return build(status, low, high, "; ".join(notes))

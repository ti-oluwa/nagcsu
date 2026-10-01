"""Local one-at-a-time sensitivity: which parameters actually move J.

A lightweight substitute for a full Morris or Sobol sensitivity design.
Each parameter is nudged up and down by a fraction of its bound range
around a base state, holding every other parameter fixed, and the
resulting swing in J is used to rank parameters by how much tuning
attention they are likely to reward.
"""

import dataclasses
import math
import typing

from nagcsu.algorithms.base import EvaluateFunction, Trial, tag_trials
from nagcsu.pipeline import EvaluationBreakdown

DetailedEvaluateFunction = typing.Callable[[dict[str, float]], EvaluationBreakdown]
"""Like `nagcsu.algorithms.base.EvaluateFunction`, but returning the
per-vector breakdown alongside `J`. Built by
`nagcsu.pipeline.make_evaluate_with_breakdown`.
"""


def probe_swing(base: float, at_low: float, at_high: float) -> tuple[float, int]:
    """Swing of a value across a low and a high probe, tolerating failed probes.

    A probe that failed to simulate scores as `inf` (or NaN). Subtracting
    it directly yields an infinite swing, which then ranks that parameter
    first and turns every share computed from a total into NaN. Instead:

    - both probes finite: `abs(at_high - at_low)`.
    - one probe failed: twice the distance from `base` to the surviving
      probe, which assumes the response is roughly symmetric. This is an
      estimate, and the caller is told a probe failed.
    - both failed (or the base itself is not finite): `0.0`, since
      nothing was learned about this parameter.

    :returns: `(swing, failed_probe_count)`.
    """
    low_ok, high_ok = math.isfinite(at_low), math.isfinite(at_high)
    if low_ok and high_ok:
        return abs(at_high - at_low), 0
    if (low_ok or high_ok) and math.isfinite(base):
        return 2.0 * abs((at_low if low_ok else at_high) - base), 1
    return 0.0, (2 if not (low_ok or high_ok) else 1)


def probe_gain(base: float, at_low: float, at_high: float) -> tuple[float, str | None]:
    """How much J can drop by moving one parameter one step in its better direction.

    `probe_swing` measures sensitivity in either direction, so a parameter
    whose every probe makes J worse ranks as high as one that improves it.
    For choosing what to tune that is the wrong question: a parameter that
    only makes things worse is already at its best along that axis (or
    wrongly scaled), and tuning it first buys nothing. The gain is the
    base J minus the better of the two probes, floored at zero.

    :returns: `(gain, side)`, where `side` is "low" or "high" for the
        better probe, or `None` when neither probe beat the base (or
        both failed).
    """
    if not math.isfinite(base):
        return 0.0, None

    options = [
        (j, side)
        for j, side in ((at_low, "low"), (at_high, "high"))
        if math.isfinite(j) and j < base
    ]
    if not options:
        return 0.0, None
    best_j, side = min(options)
    return base - best_j, side


def gap_closed(base_j: float, gain: float, target_j: float | None) -> float | None:
    """Fraction of the remaining distance to `target_j` that `gain` would close.

    :returns: A value from 0 to 1, or `None` when there is no target or the
        base J is already at or below it.
    """
    if target_j is None or not math.isfinite(base_j) or base_j <= target_j:
        return None
    return min(1.0, gain / (base_j - target_j))


@dataclasses.dataclass(frozen=True, slots=True)
class SensitivityResult:
    """How much moving one parameter, alone, changed J."""

    parameter: str
    """Parameter name this result is for."""

    base_j: float
    """J at the unperturbed base state."""

    j_at_low: float
    """J with this parameter set to its perturbed low value, others fixed."""

    j_at_high: float
    """J with this parameter set to its perturbed high value, others fixed."""

    swing: float
    """`abs(j_at_high - j_at_low)`, the ranking score: how much J moved
    when this single parameter was varied across its perturbation range.
    """

    low_value: float = 0.0
    """Parameter value used for the low probe."""

    high_value: float = 0.0
    """Parameter value used for the high probe."""

    failed_probes: int = 0
    """How many of the two probes failed to simulate. When 1, `swing` is
    an estimate from the surviving probe (see `probe_swing`); when 2, it
    is 0 and the parameter should be re-screened over a narrower range.
    """

    gain: float = 0.0
    """How much J dropped at the better probe (see `probe_gain`); 0 when
    neither probe beat the base. The default ranking key."""

    best_side: str | None = None
    """"low" or "high", the probe that gave `gain`; `None` if none helped."""

    gap_closed: float | None = None
    """Fraction of the distance from base J to the target that `gain` alone
    would close, or `None` without a target."""


SORT_KEYS: typing.Final[tuple[str, ...]] = ("gain", "swing")
"""Ways results can be ordered: by J gain (default) or by swing."""


def make_sort_key(sort_by: str) -> typing.Callable[[typing.Any], tuple[float, float]]:
    if sort_by not in SORT_KEYS:
        raise ValueError(f"Unknown `sort_by` {sort_by!r}; use one of {SORT_KEYS}")
    if sort_by == "gain":
        return lambda result: (result.gain, result.swing)
    return lambda result: (result.swing, result.gain)


def run(
    base_state: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    evaluate: EvaluateFunction,
    *,
    perturbation_fraction: float = 0.15,
    parameter_groups: dict[str, str] | None = None,
    target_j: float | None = None,
    sort_by: str = "gain",
) -> tuple[list[SensitivityResult], list[Trial]]:
    """Rank parameters by their local one-at-a-time effect on J.

    :param parameter_bounds: `(low, high)` bound per parameter to
        test; the actual perturbation used is `perturbation_fraction` of
        `high - low`, centered on the parameter's value in `base_state`
        and clamped back into `(low, high)`.
    :param perturbation_fraction: Fraction of each parameter's bound
        range to perturb by, in each direction.
    :param parameter_groups: `{parameter: group}`, used only to label
        each probe's group in the run ledger.
    :param target_j: Objective target, used only to report how much of
        the remaining gap each parameter's gain would close.
    :param sort_by: "gain" (default; parameters that can lower J first,
        ties broken by swing) or "swing" (largest effect in either
        direction first).
    :returns: One `SensitivityResult` per parameter, sorted by `sort_by`
        (most useful first), plus every trial run so the caller can log
        them to the ledger.
    """
    groups = parameter_groups or {}
    trials: list[Trial] = []
    with tag_trials(stage="sensitivity/base"):
        base_j = evaluate(base_state)
    trials.append(Trial(state=dict(base_state), j=base_j))

    results: list[SensitivityResult] = []
    for parameter, (low, high) in parameter_bounds.items():
        span = high - low
        center = base_state.get(parameter, (low + high) / 2.0)
        delta = span * perturbation_fraction
        low_value = max(low, center - delta)
        high_value = min(high, center + delta)

        low_state = dict(base_state)
        low_state[parameter] = low_value
        with tag_trials(
            group=groups.get(parameter),
            parameters=(parameter,),
            stage="sensitivity/low",
        ):
            j_low = evaluate(low_state)
        trials.append(Trial(state=low_state, j=j_low))

        high_state = dict(base_state)
        high_state[parameter] = high_value
        with tag_trials(
            group=groups.get(parameter),
            parameters=(parameter,),
            stage="sensitivity/high",
        ):
            j_high = evaluate(high_state)
        trials.append(Trial(state=high_state, j=j_high))

        swing, failed = probe_swing(base_j, j_low, j_high)
        gain, best_side = probe_gain(base_j, j_low, j_high)
        results.append(
            SensitivityResult(
                parameter=parameter,
                base_j=base_j,
                j_at_low=j_low,
                j_at_high=j_high,
                swing=swing,
                low_value=low_value,
                high_value=high_value,
                failed_probes=failed,
                gain=gain,
                best_side=best_side,
                gap_closed=gap_closed(base_j, gain, target_j),
            )
        )

    results.sort(key=make_sort_key(sort_by), reverse=True)
    return results, trials


@dataclasses.dataclass(frozen=True, slots=True)
class DetailedSensitivityResult:
    """How much moving one parameter, alone, changed `J` and each scored vector."""

    parameter: str
    """Parameter name this result is for."""

    base_j: float
    """`J` at the unperturbed base state."""

    swing: float
    """`abs(j_at_high - j_at_low)`; ranking key, same meaning as
    `SensitivityResult.swing`.
    """

    low_value: float = 0.0
    """Parameter value used for the low probe."""

    high_value: float = 0.0
    """Parameter value used for the high probe."""

    failed_probes: int = 0
    """How many of the two probes failed to simulate; see `SensitivityResult`."""

    gain: float = 0.0
    """J drop at the better probe; see `SensitivityResult.gain`."""

    best_side: str | None = None
    """"low" or "high", the probe that gave `gain`."""

    gap_closed: float | None = None
    """Fraction of the distance to the target that `gain` would close."""

    vector_swings: dict[str, float] = dataclasses.field(default_factory=dict)
    """`abs(high - low)` per scored vector's own NRMSE, keyed the same
    way as `nagcsu.objective.SCORED_FIELD_VECTORS`. This is the number
    that answers "did this parameter actually help pressure or water
    cut, or did it only move `J` because it happened to shift a runaway
    vector like GOR": a parameter with a large `swing` but a small
    `vector_swings["pressure"]` is not the one to reach for when the
    pressure match itself is what needs work.
    """


def detailed_run(
    base_state: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    evaluate: DetailedEvaluateFunction,
    *,
    perturbation_fraction: float = 0.15,
    parameter_groups: dict[str, str] | None = None,
    target_j: float | None = None,
    sort_by: str = "gain",
) -> list[DetailedSensitivityResult]:
    """Like `run`, but ranks parameters per scored vector, not only by combined `J`.

    `run`'s combined-`J` ranking can be dominated by whichever parameter
    happens to move a single runaway vector (GOR after a bubble-point
    crossing is the usual case; see
    `nagcsu.config.ObjectiveConfig.nrmse_ceiling`), which can mask a
    parameter that is actually the one hurting a different vector's own
    match. This runs the same low/high perturbation `run` does, but
    keeps each vector's NRMSE swing separate instead of collapsing
    everything into `J` first.

    :param evaluate: A callable from
        `nagcsu.pipeline.make_evaluate_with_breakdown`, not
        `nagcsu.pipeline.make_evaluate`.
    :param target_j: Objective target; see `run`.
    :param sort_by: "gain" or "swing"; see `run`.
    :returns: One `DetailedSensitivityResult` per parameter, sorted by
        `sort_by`; read `vector_swings` directly to rank by one vector
        instead.
    """
    groups = parameter_groups or {}
    with tag_trials(stage="sensitivity/base"):
        base_breakdown = evaluate(base_state)

    results: list[DetailedSensitivityResult] = []
    for parameter, (low, high) in parameter_bounds.items():
        span = high - low
        center = base_state.get(parameter, (low + high) / 2.0)
        delta = span * perturbation_fraction
        low_value = max(low, center - delta)
        high_value = min(high, center + delta)

        low_state = dict(base_state)
        low_state[parameter] = low_value
        with tag_trials(
            group=groups.get(parameter),
            parameters=(parameter,),
            stage="sensitivity/low",
        ):
            breakdown_low = evaluate(low_state)

        high_state = dict(base_state)
        high_state[parameter] = high_value
        with tag_trials(
            group=groups.get(parameter),
            parameters=(parameter,),
            stage="sensitivity/high",
        ):
            breakdown_high = evaluate(high_state)

        vector_names = (
            set(breakdown_low.vector_nrmse)
            | set(breakdown_high.vector_nrmse)
            | set(base_breakdown.vector_nrmse)
        )
        nan = float("nan")
        vector_swings = {
            vector_name: probe_swing(
                base_breakdown.vector_nrmse.get(vector_name, nan),
                breakdown_low.vector_nrmse.get(vector_name, nan),
                breakdown_high.vector_nrmse.get(vector_name, nan),
            )[0]
            for vector_name in vector_names
        }
        swing, failed = probe_swing(base_breakdown.j, breakdown_low.j, breakdown_high.j)
        gain, best_side = probe_gain(base_breakdown.j, breakdown_low.j, breakdown_high.j)
        results.append(
            DetailedSensitivityResult(
                parameter=parameter,
                base_j=base_breakdown.j,
                swing=swing,
                low_value=low_value,
                high_value=high_value,
                failed_probes=failed,
                gain=gain,
                best_side=best_side,
                gap_closed=gap_closed(base_breakdown.j, gain, target_j),
                vector_swings=vector_swings,
            )
        )

    results.sort(key=make_sort_key(sort_by), reverse=True)
    return results


GROUP_RANK_METHODS: typing.Final[tuple[str, ...]] = (
    "mean_rank",
    "rank_sum",
    "swing_share",
    "gain_share",
)
"""Ways `rank_groups` can order tuning groups; see that function."""


@dataclasses.dataclass(frozen=True, slots=True)
class GroupSensitivity:
    """How much tuning one group can do for J, summarized over its parameters."""

    group: str
    """Tuning group name."""

    position: int
    """1-based place in the group ranking; 1 is the group to tune first."""

    parameters: list[str]
    """The group's parameters, best first."""

    parameter_ranks: dict[str, float]
    """Rank of each parameter among ALL tested parameters (1 is best;
    ties share the average of the ranks they span)."""

    rank_sum: float
    """Sum of `parameter_ranks`. Lower is better, but grows with the
    number of parameters in the group."""

    mean_rank: float
    """`rank_sum / len(parameters)`. Lower is better, and is comparable
    between groups of different sizes."""

    best_rank: float
    """Rank of the group's single best parameter."""

    total_swing: float
    """Sum of the group's parameter swings, in units of J."""

    swing_share: float
    """`total_swing` as a fraction of every tested parameter's swing."""

    score: float
    """The number the group was ordered by under the chosen method."""

    total_gain: float = 0.0
    """Sum of the group's parameter gains: how far J can drop if each
    parameter moves one step in its better direction."""

    gain_share: float = 0.0
    """`total_gain` as a fraction of every tested parameter's gain."""

    ranked_by: str = "swing"
    """What the parameter ranks were built from: "gain" or "swing"."""


def _finite(value: float) -> float:
    """`value`, or 0.0 when it is NaN or infinite."""
    return value if math.isfinite(value) else 0.0


def rank_parameters(
    scores: dict[str, float], tiebreak: dict[str, float] | None = None
) -> dict[str, float]:
    """Rank parameters by `scores`, 1 for the largest.

    Parameters with equal `scores` are ordered by `tiebreak` (also larger
    first). Whatever is still tied, most often several parameters scoring
    exactly zero, receives the average of the ranks it spans, so a block
    of equally dead parameters does not look artificially ordered.
    """
    extra = tiebreak or {}
    keys = {
        name: (-_finite(score), -_finite(extra.get(name, 0.0))) for name, score in scores.items()
    }
    ordered = sorted(keys, key=lambda name: keys[name])
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and keys[ordered[end + 1]] == keys[ordered[start]]:
            end += 1
        average = (start + 1 + end + 1) / 2.0
        for name in ordered[start : end + 1]:
            ranks[name] = average
        start = end + 1
    return ranks


def rank_groups(
    parameter_swings: dict[str, float],
    parameter_groups: dict[str, str],
    *,
    method: str = "mean_rank",
    parameter_gains: dict[str, float] | None = None,
) -> list[GroupSensitivity]:
    """Order tuning groups from the most to the least worth tuning first.

    Every parameter is first ranked (see `rank_parameters`). With
    `parameter_gains` the rank is by gain, the J drop available from one
    step in the parameter's better direction, with swing breaking ties.
    That matters: ranking by swing alone puts a parameter whose every
    probe made J WORSE at the top, because swing ignores direction. Without
    `parameter_gains` the rank is by swing, as before.

    A group is then scored from the ranks of its members:

    - `"mean_rank"` (default): average member rank, ascending. Dividing
      by the group size matters: a plain rank sum rewards small groups.
      A one-parameter group whose parameter ranks 5th scores 5, while a
      four-parameter group holding ranks 1, 2, 3 and 4 scores 10 and
      loses, even though it holds the four best parameters.
    - `"rank_sum"`: total of member ranks, ascending. Only fair when every
      group has the same number of parameters.
    - `"swing_share"`: total swing in J units, descending.
    - `"gain_share"`: total gain in J units, descending. Ranks throw away
      magnitude, so this is the one to read when the ranking and the raw
      numbers seem to disagree. Needs `parameter_gains`.

    Ties on the primary score are broken by the best single-parameter
    rank, then by total gain, then by total swing.

    :param parameter_swings: `{parameter: swing}` for every tested parameter.
    :param parameter_groups: `{parameter: group}` covering the same names.
    :param method: One of `GROUP_RANK_METHODS`.
    :param parameter_gains: `{parameter: gain}` for the same names.
    :raises ValueError: if `method` is not recognized, or is `"gain_share"`
        without `parameter_gains`.
    """
    if method not in GROUP_RANK_METHODS:
        raise ValueError(
            f"Unknown group ranking method {method!r}; use one of {GROUP_RANK_METHODS}"
        )
    if method == "gain_share" and parameter_gains is None:
        raise ValueError("`gain_share` needs `parameter_gains`")

    swings = {name: _finite(swing) for name, swing in parameter_swings.items()}
    gains = {name: _finite((parameter_gains or {}).get(name, 0.0)) for name in swings}
    if parameter_gains is not None:
        ranks = rank_parameters(gains, tiebreak=swings)
        ranked_by = "gain"
    else:
        ranks = rank_parameters(swings)
        ranked_by = "swing"

    grand_swing = sum(swings.values())
    grand_gain = sum(gains.values())
    members: dict[str, list[str]] = {}
    for parameter in swings:
        members.setdefault(parameter_groups[parameter], []).append(parameter)

    summaries: list[GroupSensitivity] = []
    for group, names in members.items():
        names = sorted(names, key=lambda name: ranks[name])
        rank_sum = sum(ranks[name] for name in names)
        mean_rank = rank_sum / len(names)
        total_swing = sum(swings[name] for name in names)
        total_gain = sum(gains[name] for name in names)
        score = {
            "mean_rank": mean_rank,
            "rank_sum": rank_sum,
            "swing_share": total_swing,
            "gain_share": total_gain,
        }[method]
        summaries.append(
            GroupSensitivity(
                group=group,
                position=0,
                parameters=names,
                parameter_ranks={name: ranks[name] for name in names},
                rank_sum=rank_sum,
                mean_rank=mean_rank,
                best_rank=ranks[names[0]],
                total_swing=total_swing,
                swing_share=total_swing / grand_swing if grand_swing > 0 else 0.0,
                score=score,
                total_gain=total_gain,
                gain_share=total_gain / grand_gain if grand_gain > 0 else 0.0,
                ranked_by=ranked_by,
            )
        )

    direction = -1.0 if method in ("swing_share", "gain_share") else 1.0
    summaries.sort(
        key=lambda summary: (
            direction * summary.score,
            summary.best_rank,
            -summary.total_gain,
            -summary.total_swing,
        )
    )
    return [
        dataclasses.replace(summary, position=index) for index, summary in enumerate(summaries, 1)
    ]


def build_tuning_plan(
    group_sensitivities: list[GroupSensitivity],
    parameter_swings: dict[str, float],
    *,
    min_relative_swing: float = 0.0,
) -> list[tuple[str, list[str]]]:
    """Turn a sensitivity screen into a tuning order for coordinate descent.

    Groups keep the order of `group_sensitivities`. Inside a group,
    parameters keep that summary's order (best first). A parameter is
    dropped only when its swing is below `min_relative_swing` times the
    largest swing anywhere, meaning it barely moves J in either direction.
    A parameter that is sensitive but showed no gain at this one probe
    size is kept and simply ordered late, because its best value can shift
    once other parameters have moved. A group left with no parameters is
    dropped too.

    :returns: `[(group, [parameter, ...]), ...]` in tuning order.
    """
    parameter_swings = {name: _finite(swing) for name, swing in parameter_swings.items()}
    largest = max(parameter_swings.values(), default=0.0)
    threshold = largest * min_relative_swing
    plan: list[tuple[str, list[str]]] = []
    for summary in group_sensitivities:
        kept = [name for name in summary.parameters if parameter_swings[name] >= threshold]
        if kept:
            plan.append((summary.group, kept))
    return plan


def summarize(
    results: typing.Sequence[typing.Any],
    parameter_groups: dict[str, str],
    *,
    rank_by: str = "gain",
    method: str = "mean_rank",
) -> tuple[dict[str, float], list[GroupSensitivity]]:
    """Rank parameters and groups from `SensitivityResult` or `DetailedSensitivityResult` rows.

    :param rank_by: "gain" (default) ranks parameters by how far one step in
        their better direction lowers J, swing breaking ties. "swing" ranks
        by effect in either direction, which also rewards parameters whose
        every probe made J worse.
    :param method: How groups are ordered; see `rank_groups`.
    :returns: `(parameter_ranks, ranked_groups)`.
    """
    if rank_by not in SORT_KEYS:
        raise ValueError(f"Unknown `rank_by` {rank_by!r}; use one of {SORT_KEYS}")

    swings = {result.parameter: result.swing for result in results}
    gains = {result.parameter: result.gain for result in results}
    use_gain = rank_by == "gain" or method == "gain_share"
    ranks = (
        rank_parameters(gains, tiebreak=swings) if rank_by == "gain" else rank_parameters(swings)
    )
    groups = rank_groups(
        swings,
        parameter_groups,
        method=method,
        parameter_gains=gains if use_gain else None,
    )
    return ranks, groups

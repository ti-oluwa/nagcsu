"""Local one-at-a-time sensitivity: which parameters actually move J.

A lightweight substitute for a full Morris or Sobol sensitivity design.
Each parameter is nudged up and down by a fraction of its bound range
around a base state, holding every other parameter fixed, and the
resulting swing in J is used to rank parameters by how much tuning
attention they are likely to reward.
"""

import dataclasses
import typing

from nagcsu.algorithms.base import EvaluateFunction, Trial
from nagcsu.pipeline import EvaluationBreakdown

DetailedEvaluateFunction = typing.Callable[[dict[str, float]], EvaluationBreakdown]
"""Like `nagcsu.algorithms.base.EvaluateFunction`, but returning the
per-vector breakdown alongside `J`. Built by
`nagcsu.pipeline.make_evaluate_with_breakdown`.
"""


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


def run(
    base_state: dict[str, float],
    bounds_by_parameter: dict[str, tuple[float, float]],
    evaluate: EvaluateFunction,
    *,
    perturbation_fraction: float = 0.15,
) -> tuple[list[SensitivityResult], list[Trial]]:
    """Rank parameters by their local one-at-a-time effect on J.

    :param bounds_by_parameter: `(low, high)` bound per parameter to
        test; the actual perturbation used is `perturbation_fraction` of
        `high - low`, centered on the parameter's value in `base_state`
        and clamped back into `(low, high)`.
    :param perturbation_fraction: Fraction of each parameter's bound
        range to perturb by, in each direction.
    :returns: One `SensitivityResult` per parameter, sorted by
        descending `swing` (most influential first), plus every trial
        run so the caller can log them to the ledger.
    """
    trials: list[Trial] = []
    base_j = evaluate(base_state)
    trials.append(Trial(state=dict(base_state), j=base_j))

    results: list[SensitivityResult] = []
    for name, (low, high) in bounds_by_parameter.items():
        span = high - low
        center = base_state.get(name, (low + high) / 2.0)
        delta = span * perturbation_fraction
        low_value = max(low, center - delta)
        high_value = min(high, center + delta)

        low_state = dict(base_state)
        low_state[name] = low_value
        j_low = evaluate(low_state)
        trials.append(Trial(state=low_state, j=j_low))

        high_state = dict(base_state)
        high_state[name] = high_value
        j_high = evaluate(high_state)
        trials.append(Trial(state=high_state, j=j_high))

        results.append(
            SensitivityResult(
                parameter=name,
                base_j=base_j,
                j_at_low=j_low,
                j_at_high=j_high,
                swing=abs(j_high - j_low),
            )
        )

    results.sort(key=lambda result: result.swing, reverse=True)
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

    vector_swings: dict[str, float]
    """`abs(high - low)` per scored vector's own NRMSE, keyed the same
    way as `nagcsu.objective.SCORED_FIELD_VECTORS`. This is the number
    that answers "did this parameter actually help pressure or water
    cut, or did it only move `J` because it happened to shift a runaway
    vector like GOR": a parameter with a large `swing` but a small
    `vector_swings["pressure"]` is not the one to reach for when the
    pressure match itself is what needs work.
    """


def run_detailed(
    base_state: dict[str, float],
    bounds_by_parameter: dict[str, tuple[float, float]],
    evaluate: DetailedEvaluateFunction,
    *,
    perturbation_fraction: float = 0.15,
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
    :returns: One `DetailedSensitivityResult` per parameter, sorted by
        descending combined `swing` (most influential on `J` first);
        read `vector_swings` directly to rank by one vector instead.
    """
    base_breakdown = evaluate(base_state)

    results: list[DetailedSensitivityResult] = []
    for name, (low, high) in bounds_by_parameter.items():
        span = high - low
        center = base_state.get(name, (low + high) / 2.0)
        delta = span * perturbation_fraction
        low_value = max(low, center - delta)
        high_value = min(high, center + delta)

        low_state = dict(base_state)
        low_state[name] = low_value
        breakdown_low = evaluate(low_state)

        high_state = dict(base_state)
        high_state[name] = high_value
        breakdown_high = evaluate(high_state)

        vector_names = set(breakdown_low.vector_nrmse) | set(breakdown_high.vector_nrmse)
        vector_swings = {
            vector_name: abs(
                breakdown_high.vector_nrmse.get(vector_name, 0.0)
                - breakdown_low.vector_nrmse.get(vector_name, 0.0)
            )
            for vector_name in vector_names
        }
        results.append(
            DetailedSensitivityResult(
                parameter=name,
                base_j=base_breakdown.j,
                swing=abs(breakdown_high.j - breakdown_low.j),
                vector_swings=vector_swings,
            )
        )

    results.sort(key=lambda result: result.swing, reverse=True)
    return results

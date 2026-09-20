"""Local one-at-a-time sensitivity: which parameters actually move J.

A lightweight substitute for a full Morris or Sobol sensitivity design.
Each parameter is nudged up and down by a fraction of its bound range
around a base state, holding every other parameter fixed, and the
resulting swing in J is used to rank parameters by how much tuning
attention they are likely to reward. This is what
`nagcsu sensitivity run` reports, and what a `nagcsu match auto` summary
points to when suggesting what to try next after stopping.
"""

import dataclasses

from nagcsu.algorithms import EvaluateFunction, Trial


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

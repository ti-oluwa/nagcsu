"""Grid search: evaluate every combination of a set of parameter values.

The direct equivalent of Stage D.4's `sweep()` helper in the Execution
Plan, generalized from one parameter to any number, with a hard cap on
how many combinations it will silently run.
"""

import itertools

from nagcsu.algorithms import EvaluateFunction, SearchResult, Trial, best_of


MAX_EVALUATIONS_DEFAULT = 500
"""Refuse to run a grid larger than this unless the caller raises
`max_evaluations` explicitly. A 5-parameter, 5-value-each grid is
already 3125 simulation runs; this catches that mistake before it burns
an afternoon of compute.
"""


def search(
    base_state: dict[str, float],
    values_by_parameter: dict[str, list[float]],
    evaluate: EvaluateFunction,
    *,
    max_evaluations: int = MAX_EVALUATIONS_DEFAULT,
) -> SearchResult:
    """Evaluate every combination of `values_by_parameter` against `base_state`.

    :param base_state: Full parameter state; every parameter not in
        `values_by_parameter` is held fixed at its value here.
    :param values_by_parameter: The parameter values to combine, keyed
        by parameter name. One entry sweeps a single parameter; more
        than one produces their Cartesian product.
    :raises ValueError: if the Cartesian product of `values_by_parameter`
        would exceed `max_evaluations`.
    """
    parameter_names = list(values_by_parameter.keys())
    combinations = list(itertools.product(*(values_by_parameter[name] for name in parameter_names)))
    if len(combinations) > max_evaluations:
        raise ValueError(
            f"Grid over {parameter_names} has {len(combinations)} combinations, "
            f"exceeding max_evaluations={max_evaluations}. Narrow the value lists "
            f"or raise max_evaluations explicitly."
        )

    trials: list[Trial] = []
    for combination in combinations:
        state = dict(base_state)
        state.update(zip(parameter_names, combination))
        trials.append(Trial(state=state, j=evaluate(state)))

    return SearchResult(trials=trials, best=best_of(trials), strategy="grid")

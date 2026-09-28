"""Grid search: evaluate every combination of a set of parameter values."""

import itertools

from nagcsu.algorithms.base import EvaluateFunction, SearchResult, Trial, best_of

MAX_EVALUATIONS_DEFAULT = 500
"""
Refuse to run a grid larger than this unless the caller raises
`max_evaluations` explicitly. A 5-parameter, 5-value-each grid is
already 3125 simulation runs; this catches that mistake before it burns
an afternoon of compute.
"""


def search(
    base_state: dict[str, float],
    parameter_values: dict[str, list[float]],
    evaluate: EvaluateFunction,
    *,
    max_evaluations: int = MAX_EVALUATIONS_DEFAULT,
) -> SearchResult:
    """
    Evaluate every combination of `parameter_values` against `base_state`.

    :param base_state: Full parameter state; every parameter not in
        `parameter_values` is held fixed at its value here.
    :param parameter_values: The parameter values to combine, keyed
        by parameter name. One entry sweeps a single parameter; more
        than one produces their Cartesian product.
    :raises ValueError: if the Cartesian product of `parameter_values`
        would exceed `max_evaluations`.
    """
    parameters = list(parameter_values.keys())
    combinations = list(
        itertools.product(*(parameter_values[parameter] for parameter in parameters))
    )
    if len(combinations) > max_evaluations:
        raise ValueError(
            f"Grid over {parameters} has {len(combinations)} combinations, "
            f"exceeding `max_evaluations={max_evaluations}`. Narrow the value lists "
            f"or raise max_evaluations explicitly."
        )

    trials: list[Trial] = []
    for combination in combinations:
        state = dict(base_state)
        state.update(zip(parameters, combination, strict=True))
        trials.append(Trial(state=state, j=evaluate(state)))

    return SearchResult(trials=trials, best=best_of(trials), strategy="grid")

"""Random search: uniform sampling within a set of parameter bounds.

A useful default when a parameter group has too many members, or too
wide a range each, for a grid to cover without an unreasonable number
of runs; often finds a decent region faster than a grid at the same
evaluation budget.
"""

import random

from nagcsu.algorithms.base import EvaluateFunction, SearchResult, Trial, best_of


def search(
    base_state: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    evaluate: EvaluateFunction,
    *,
    n_trials: int = 20,
    seed: int | None = None,
) -> SearchResult:
    """Evaluate `n_trials` uniformly random states within `parameter_bounds`.

    :param base_state: Full parameter state; every parameter not in
        `parameter_bounds` is held fixed at its value here.
    :param parameter_bounds: Inclusive `(low, high)` sampling range
        per parameter name to randomize.
    :param seed: Random seed, for a reproducible sequence of trials.
    """
    rng = random.Random(seed)
    trials: list[Trial] = []
    for _ in range(n_trials):
        state = dict(base_state)
        for parameter, (low, high) in parameter_bounds.items():
            state[parameter] = rng.uniform(low, high)
        trials.append(Trial(state=state, j=evaluate(state)))

    return SearchResult(trials=trials, best=best_of(trials), strategy="random")

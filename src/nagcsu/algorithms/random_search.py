"""Random search: uniform sampling within a set of parameter bounds.

A useful default when a parameter group has too many members, or too
wide a range each, for a grid to cover without an unreasonable number
of runs; often finds a decent region faster than a grid at the same
evaluation budget.
"""

import random

from nagcsu.algorithms import EvaluateFunction, SearchResult, Trial, best_of


def search(
    base_state: dict[str, float],
    bounds_by_parameter: dict[str, tuple[float, float]],
    evaluate: EvaluateFunction,
    *,
    num_trials: int = 20,
    seed: int | None = None,
) -> SearchResult:
    """Evaluate `num_trials` uniformly random states within `bounds_by_parameter`.

    :param base_state: Full parameter state; every parameter not in
        `bounds_by_parameter` is held fixed at its value here.
    :param bounds_by_parameter: Inclusive `(low, high)` sampling range
        per parameter name to randomize.
    :param seed: Random seed, for a reproducible sequence of trials.
    """
    rng = random.Random(seed)
    trials: list[Trial] = []
    for _ in range(num_trials):
        state = dict(base_state)
        for name, (low, high) in bounds_by_parameter.items():
            state[name] = rng.uniform(low, high)
        trials.append(Trial(state=state, j=evaluate(state)))

    return SearchResult(trials=trials, best=best_of(trials), strategy="random")

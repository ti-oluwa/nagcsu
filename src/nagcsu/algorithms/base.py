"""Shared types every search strategy module builds on.

`Trial`, `SearchResult` and `EvaluateFunction` are the common currency
between `nagcsu.algorithms.grid`, `random_search`, `coordinate_descent`
and `sensitivity`: each strategy takes an `EvaluateFunction` and returns
a `SearchResult`, without needing to know how the other strategies work.
"""

import dataclasses
import typing

EvaluateFunction = typing.Callable[[dict[str, float]], float]
"""A function that takes a full parameter state and returns J for it.
Expected to be built by the caller around `nagcsu.parameters.apply_state`,
`nagcsu.simulate.run`, `nagcsu.summary.load_summary` and `nagcsu.objective.score`.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class Trial:
    """One evaluated parameter state and the J it produced."""

    state: dict[str, float]
    """Full parameter state evaluated for this trial."""

    j: float
    """Objective value `evaluate(state)` returned."""


@dataclasses.dataclass(frozen=True, slots=True)
class SearchResult:
    """Every trial a search strategy ran, and which one was best."""

    trials: list[Trial]
    """All trials, in the order they were evaluated."""

    best: Trial
    """The trial with the lowest `j`."""

    strategy: str
    """Name of the strategy that produced this result, for the run ledger."""


def best_of(trials: list[Trial]) -> Trial:
    """Return the trial with the lowest `j`.

    :raises ValueError: if `trials` is empty.
    """
    if not trials:
        raise ValueError("Cannot find the best of zero trials")
    return min(trials, key=lambda trial: trial.j)

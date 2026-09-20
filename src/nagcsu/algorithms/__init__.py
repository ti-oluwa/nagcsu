"""Search strategies for finding a parameter state that lowers J.

Each strategy module (`grid`, `random_search`, `coordinate_descent`) is a
self-contained entry point: it takes a starting parameter state, the
bounds to search within, and an `evaluate` callback the caller builds
(typically `nagcsu.parameters.apply_state` -> `nagcsu.simulate.run` ->
`nagcsu.summary.load_summary` -> `nagcsu.objective.score`), and returns
a :class:`SearchResult`. None of them know how to run a simulation
themselves, which keeps every strategy testable against a cheap fake
`evaluate` function instead of a real OPM Flow run.
"""

import dataclasses
import typing


EvaluateFunction = typing.Callable[[dict[str, float]], float]
"""A function that takes a full parameter state and returns J for it.
Expected to be built by the caller around
:func:`nagcsu.parameters.apply_state`,
:func:`nagcsu.simulate.run`, :func:`nagcsu.summary.load_summary` and
:func:`nagcsu.objective.score`.
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

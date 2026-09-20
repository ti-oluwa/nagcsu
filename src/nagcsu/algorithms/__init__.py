"""Search strategies for finding a parameter state that lowers J.

Each strategy module (`grid`, `random_search`, `coordinate_descent`) is a
self-contained entry point: it takes a starting parameter state, the
bounds to search within, and an `evaluate` callback the caller builds
(typically `nagcsu.parameters.apply_state` -> `nagcsu.simulate.run` ->
`nagcsu.summary.load_summary` -> `nagcsu.objective.score`), and returns
a `SearchResult`. None of them know how to run a simulation themselves,
which keeps every strategy testable against a cheap fake `evaluate`
function instead of a real OPM Flow run. The shared types below live in
`nagcsu.algorithms.base`; re-exported here for a shorter import path.
"""

from nagcsu.algorithms.base import EvaluateFunction, SearchResult, Trial, best_of

__all__ = ["EvaluateFunction", "SearchResult", "Trial", "best_of"]

"""Shared types every search strategy module builds on."""

import contextlib
import contextvars
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


@dataclasses.dataclass(frozen=True, slots=True)
class TrialTag:
    """What a search strategy was doing when it asked for one evaluation.

    Set by the strategy around each `evaluate` call (see `tag_trials`) and
    read back when the run is logged (`nagcsu.pipeline.build_run_record`),
    so the ledger records which group and which parameter each trial was
    changing without threading extra arguments through `evaluate`.
    """

    group: str | None = None
    """Tuning group the trial belongs to."""

    parameters: tuple[str, ...] = ()
    """Parameters this trial deliberately moved away from the state it
    was derived from. Empty for a baseline or final re-run.
    """

    stage: str | None = None
    """Short label for the step within the strategy, for example
    "baseline", "descent/pass1", "sensitivity/low" or "final".
    """


CURRENT_TAG: contextvars.ContextVar[TrialTag | None] = contextvars.ContextVar(
    "nagcsu_current_trial_tag", default=None
)


@contextlib.contextmanager
def tag_trials(
    *, group: str | None = None, parameters: typing.Sequence[str] = (), stage: str | None = None
) -> typing.Iterator[None]:
    """Tag every `evaluate` call made inside the `with` block.

    Tags do not nest additively. The innermost block's tag wins, and the
    previous one is restored on exit.
    """
    token = CURRENT_TAG.set(TrialTag(group=group, parameters=tuple(parameters), stage=stage))
    try:
        yield
    finally:
        CURRENT_TAG.reset(token)


def get_current_tag() -> TrialTag | None:
    """Return the tag set by the innermost active `tag_trials` block, if any."""
    return CURRENT_TAG.get()

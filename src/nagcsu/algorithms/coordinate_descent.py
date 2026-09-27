"""Auto-tuning: one parameter group at a time, in priority order."""

import dataclasses
import random

import scipy.optimize

from nagcsu.algorithms.base import EvaluateFunction, SearchResult, Trial, best_of


@dataclasses.dataclass(frozen=True, slots=True)
class GroupOutcome:
    """What happened while tuning one parameter group."""

    group: str
    """Name of the tuning priority group, for example "aquifer"."""

    starting_j: float
    """Best J before this group was touched."""

    ending_j: float
    """Best J after this group was tuned."""

    reached_target: bool
    """Whether `ending_j` is at or below the search's `target_j`."""


def search(
    base_state: dict[str, float],
    groups_in_order: list[str],
    bounds_by_group: dict[str, dict[str, tuple[float, float]]],
    evaluate: EvaluateFunction,
    *,
    target_j: float,
    passes_per_group: int = 2,
) -> tuple[SearchResult, list[GroupOutcome]]:
    """Tune `groups_in_order` one at a time until `target_j` is reached.

    :param base_state: Starting parameter state, normally
        `nagcsu.parameters.default_state`.
    :param groups_in_order: Tuning priority order, normally
        `nagcsu.constants.TUNING_PRIORITY_ORDER`.
    :param bounds_by_group: `{parameter_name: (low, high)}` for every
        parameter in each group, keyed by group name.
    :param target_j: Stop as soon as the best J found is at or below this.
    :param passes_per_group: How many full cycles through a group's
        parameters to run before moving on, if the target is not yet
        reached. Each pass re-optimizes every parameter in the group
        once, holding the others at their current best value.
    :returns: The full `SearchResult` across every trial run, plus one
        `GroupOutcome` per group that was actually touched (a group
        after the target was already reached is skipped and not
        included).
    """
    trials: list[Trial] = []
    current_best_state = dict(base_state)
    current_best_j = evaluate(current_best_state)
    trials.append(Trial(state=dict(current_best_state), j=current_best_j))

    outcomes: list[GroupOutcome] = []
    for group in groups_in_order:
        if current_best_j <= target_j:
            break

        group_bounds = bounds_by_group.get(group, {})
        if not group_bounds:
            continue

        starting_j = current_best_j
        for _ in range(passes_per_group):
            if current_best_j <= target_j:
                break
            for parameter_name, (low, high) in group_bounds.items():

                def objective_along_one_axis(value: float, _name: str = parameter_name) -> float:
                    candidate_state = dict(current_best_state)
                    candidate_state[_name] = value
                    j = evaluate(candidate_state)
                    trials.append(Trial(state=dict(candidate_state), j=j))
                    return j

                result = scipy.optimize.minimize_scalar(
                    objective_along_one_axis,
                    bounds=(low, high),
                    method="bounded",
                )
                if result.fun < current_best_j:  # type: ignore[attr-defined]
                    current_best_state[parameter_name] = result.x  # type: ignore[attr-defined]
                    current_best_j = result.fun  # type: ignore[attr-defined]

        outcomes.append(
            GroupOutcome(
                group=group,
                starting_j=starting_j,
                ending_j=current_best_j,
                reached_target=current_best_j <= target_j,
            )
        )

    return SearchResult(
        trials=trials,
        best=best_of(trials),
        strategy="coordinate_descent",
    ), outcomes


def multi_start_search(
    base_state: dict[str, float],
    groups_in_order: list[str],
    bounds_by_group: dict[str, dict[str, tuple[float, float]]],
    evaluate: EvaluateFunction,
    *,
    target_j: float,
    passes_per_group: int = 2,
    num_starts: int = 1,
    seed: int | None = None,
) -> tuple[SearchResult, list[GroupOutcome]]:
    """Run `search` from several starting states and keep the best.

    Coordinate descent tunes one parameter at a time, holding every
    other parameter fixed, and `scipy.optimize.minimize_scalar` assumes
    the objective is roughly unimodal along that one axis. Neither
    holds up well near a threshold-like nonlinearity, a reservoir's
    pressure trajectory crossing its bubble point is a concrete example,
    where a run started on the wrong side of the threshold can converge
    to a poor local optimum without ever finding the basin containing a
    much better one, and no amount of retuning from that same starting
    point escapes it. Re-running `search` from several different,
    randomly chosen starting states within `bounds_by_group`, and
    keeping whichever run reached the lowest `J`, is a cheap,
    dependency-free way to cover more of the parameter space than a
    single start can, without changing anything about how any one start
    is itself searched.

    :param num_starts: Number of independent runs of `search`, including
        the one from `base_state` itself, which always runs first. `1`
        reproduces `search`'s exact behavior.
    :param seed: Random seed for the additional starting states, for a
        reproducible sequence of starts. Unused when `num_starts <= 1`;
        `base_state` itself is never randomized.
    :returns: The `SearchResult` and `GroupOutcome` list from whichever
        start reached the lowest `best.j`, so the return shape and
        meaning are identical to `search`'s. Every start still runs (and
        so every trial is still logged, if the caller's `evaluate` logs
        to the ledger through `on_outcome`); only the winning start's own
        bookkeeping is returned.
    """
    best_result, best_outcomes = search(
        base_state,
        groups_in_order,
        bounds_by_group,
        evaluate,
        target_j=target_j,
        passes_per_group=passes_per_group,
    )

    rng = random.Random(seed)
    for _ in range(max(0, num_starts - 1)):
        if best_result.best.j <= target_j:
            break

        random_state = dict(base_state)
        for group_bounds in bounds_by_group.values():
            for parameter_name, (low, high) in group_bounds.items():
                random_state[parameter_name] = rng.uniform(low, high)

        result, outcomes = search(
            random_state,
            groups_in_order,
            bounds_by_group,
            evaluate,
            target_j=target_j,
            passes_per_group=passes_per_group,
        )
        if result.best.j < best_result.best.j:
            best_result, best_outcomes = result, outcomes

    return best_result, best_outcomes

"""Auto-tuning: one parameter group at a time, in priority order.

Implements Stage D.1 and D.3 of the Phase 2 Execution Plan directly:
groups are tuned in the priority order the plan gives (aquifer, then
permeability multiplier, then SGOF shape, and so on), one parameter
within the current group at a time, so an improvement or a regression
can always be attributed to a single change. Tuning stops the moment J
reaches the target, exactly as Stage C.4 and D.5 recommend, whether that
happens partway through the first group or only after the last one.

Each parameter is optimized with `scipy.optimize.minimize_scalar`'s
bounded method, holding every other parameter fixed at the best state
found so far. This is a coordinate descent, not a global optimizer: it
will not escape a bad starting region on its own, which is exactly why
the priority order matters and why this should be run from the deck's
shipped defaults rather than an arbitrary starting point.
"""

import dataclasses

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
                if result.fun < current_best_j:
                    current_best_state[parameter_name] = result.x
                    current_best_j = result.fun

        outcomes.append(
            GroupOutcome(
                group=group,
                starting_j=starting_j,
                ending_j=current_best_j,
                reached_target=current_best_j <= target_j,
            )
        )

    return SearchResult(
        trials=trials, best=best_of(trials), strategy="coordinate_descent"
    ), outcomes

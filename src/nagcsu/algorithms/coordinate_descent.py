"""Auto-tuning: one parameter group at a time, in priority order."""

import dataclasses
import random

import scipy.optimize

from nagcsu.algorithms.base import EvaluateFunction, SearchResult, Trial, best_of

DEFAULT_XATOL_FRACTION = 0.02
"""Default `xatol_fraction` for `search`: locate each parameter to within
2 percent of its bound range."""

DEFAULT_MAX_EVALUATIONS_PER_PARAMETER = 12
"""Default `max_evaluations_per_parameter` for `search`."""

DEFAULT_MIN_RELATIVE_IMPROVEMENT = 0.005
"""Default `min_relative_improvement` for `search`: a pass that improves J
by less than half a percent counts as no progress."""


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
    groups: list[str],
    group_parameter_bounds: dict[str, dict[str, tuple[float, float]]],
    evaluate: EvaluateFunction,
    *,
    target_j: float,
    passes_per_group: int = 2,
    xatol_fraction: float = DEFAULT_XATOL_FRACTION,
    max_evaluations_per_parameter: int = DEFAULT_MAX_EVALUATIONS_PER_PARAMETER,
    min_relative_improvement: float = DEFAULT_MIN_RELATIVE_IMPROVEMENT,
) -> tuple[SearchResult, list[GroupOutcome]]:
    """Tune `groups` one at a time until `target_j` is reached.

    :param base_state: Starting parameter state, normally
        `nagcsu.parameters.default_state`.
    :param groups: Tuning priority order, normally
        `nagcsu.constants.GROUP_TUNING_PRIORITY_ORDER`.
    :param group_parameter_bounds: `{parameter: (low, high)}` for every
        parameter in each group, keyed by group name.
    :param target_j: Stop as soon as the best J found is at or below this.
    :param passes_per_group: How many full cycles through a group's
        parameters to run before moving on, if the target is not yet
        reached. Each pass re-optimizes every parameter in the group
        once, holding the others at their current best value.
    :param xatol_fraction: How precisely each parameter is located,
        as a fraction of that parameter's own bound range. The default
        of 2 percent stops a one-parameter search once the bracket is
        that narrow. Without it, scipy's absolute default tolerance
        keeps refining a parameter far past the point where the change
        is visible in J (an aquifer radius searched to four decimals,
        for example), and every extra refinement is a full simulation.
    :param max_evaluations_per_parameter: Hard cap on simulations spent
        on one parameter in one pass.
    :param min_relative_improvement: If a full pass over a group lowers
        J by less than this fraction of its value going in, the group is
        treated as exhausted and the search moves on to the next one
        instead of spending another pass on it. This is what stops a
        group whose parameters barely influence J (an aquifer that does
        not control water cut, for example) from absorbing dozens of
        runs.
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
    for group in groups:
        if current_best_j <= target_j:
            break

        group_bounds = group_parameter_bounds.get(group, {})
        if not group_bounds:
            continue

        starting_j = current_best_j
        for _ in range(passes_per_group):
            if current_best_j <= target_j:
                break
            j_before_pass = current_best_j
            for parameter, (low, high) in group_bounds.items():

                def objective_along_one_axis(value: float, name: str = parameter) -> float:
                    # scipy calls this with numpy.float64 values; cast
                    # to native float immediately so every trial logged
                    # from here down, not just the group's eventual best,
                    # is plain-Python and safe for `write_parameters_snapshot`
                    # to yaml.safe_dump.
                    candidate_state = dict(current_best_state)
                    candidate_state[name] = float(value)
                    j = evaluate(candidate_state)
                    trials.append(Trial(state=dict(candidate_state), j=j))
                    return j

                result = scipy.optimize.minimize_scalar(
                    objective_along_one_axis,
                    bounds=(low, high),
                    method="bounded",
                    options={
                        "xatol": (high - low) * xatol_fraction,
                        "maxiter": max_evaluations_per_parameter,
                    },
                )
                if result.fun < current_best_j:  # type: ignore[attr-defined]
                    # scipy returns numpy scalars here; cast to native
                    # float so this state stays plain-Python all the way
                    # out to `write_parameters_snapshot`'s YAML dump,
                    # which PyYAML's SafeDumper cannot serialize a
                    # numpy.float64 through.
                    current_best_state[parameter] = float(result.x)  # type: ignore[attr-defined]
                    current_best_j = float(result.fun)  # type: ignore[attr-defined]

            if (j_before_pass - current_best_j) < j_before_pass * min_relative_improvement:
                break

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
    groups: list[str],
    group_parameter_bounds: dict[str, dict[str, tuple[float, float]]],
    evaluate: EvaluateFunction,
    *,
    target_j: float,
    passes_per_group: int = 2,
    xatol_fraction: float = DEFAULT_XATOL_FRACTION,
    max_evaluations_per_parameter: int = DEFAULT_MAX_EVALUATIONS_PER_PARAMETER,
    min_relative_improvement: float = DEFAULT_MIN_RELATIVE_IMPROVEMENT,
    n_starts: int = 1,
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
    randomly chosen starting states within `group_parameter_bounds`, and
    keeping whichever run reached the lowest `J`, is a cheap,
    dependency-free way to cover more of the parameter space than a
    single start can, without changing anything about how any one start
    is itself searched.

    :param n_starts: Number of independent runs of `search`, including
        the one from `base_state` itself, which always runs first. `1`
        reproduces `search`'s exact behavior.
    :param seed: Random seed for the additional starting states, for a
        reproducible sequence of starts. Unused when `n_starts <= 1`;
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
        groups,
        group_parameter_bounds,
        evaluate,
        target_j=target_j,
        passes_per_group=passes_per_group,
        xatol_fraction=xatol_fraction,
        max_evaluations_per_parameter=max_evaluations_per_parameter,
        min_relative_improvement=min_relative_improvement,
    )

    rng = random.Random(seed)
    for _ in range(max(0, n_starts - 1)):
        if best_result.best.j <= target_j:
            break

        random_state = dict(base_state)
        for group_bounds in group_parameter_bounds.values():
            for parameter, (low, high) in group_bounds.items():
                random_state[parameter] = rng.uniform(low, high)

        result, outcomes = search(
            random_state,
            groups,
            group_parameter_bounds,
            evaluate,
            target_j=target_j,
            passes_per_group=passes_per_group,
            xatol_fraction=xatol_fraction,
            max_evaluations_per_parameter=max_evaluations_per_parameter,
            min_relative_improvement=min_relative_improvement,
        )
        if result.best.j < best_result.best.j:
            best_result, best_outcomes = result, outcomes

    return best_result, best_outcomes

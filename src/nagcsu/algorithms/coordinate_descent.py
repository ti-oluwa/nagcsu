"""Auto-tuning: one parameter group at a time, in priority order."""

import dataclasses
import random

import scipy.optimize

from nagcsu.algorithms.base import EvaluateFunction, SearchResult, Trial, best_of, tag_trials

DEFAULT_XATOL_FRACTION = 0.02
"""Default `xatol_fraction` for `search`: locate each parameter to within
2 percent of its bound range."""

DEFAULT_MAX_EVALUATIONS_PER_PARAMETER = 12
"""Default `max_evaluations_per_parameter` for `search`."""

DEFAULT_MIN_RELATIVE_IMPROVEMENT = 0.005
"""Default `min_relative_improvement` for `search`: a pass that improves J
by less than half a percent counts as no progress."""


DEFAULT_WINDOW_SHRINK = 0.5
"""Default `window_shrink` for `search`: each pass after the first
searches a window half as wide as the previous pass's, centered on the
best value found so far."""


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterOutcome:
    """What happened while tuning one parameter once, in one pass."""

    group: str
    """Tuning group the parameter belongs to."""

    parameter: str
    """Parameter name."""

    pass_index: int
    """1-based pass over the group this happened in."""

    window: tuple[float, float]
    """`(low, high)` the one-dimensional search was confined to. Equal to
    the parameter's full bounds on pass 1 and narrower afterwards.
    """

    start_value: float
    """Parameter value going into this step."""

    end_value: float
    """Parameter value after this step (unchanged if nothing beat the
    incumbent)."""

    starting_j: float
    """Best J going into this step."""

    ending_j: float
    """Best J after this step."""

    evaluations: int
    """Simulations spent on this step."""


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

    evaluations: int = 0
    """Simulations spent on this group across all its passes."""

    passes: int = 0
    """Passes actually run over this group's parameters."""

    parameter_outcomes: list[ParameterOutcome] = dataclasses.field(default_factory=list)
    """One entry per parameter per pass, in the order they ran, so a
    report can show which parameter inside the group did the work.
    """


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
    window_shrink: float = DEFAULT_WINDOW_SHRINK,
    stage_prefix: str = "",
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
    :param window_shrink: Pass 1 searches each parameter over its full
        bounds. Every later pass searches a window `window_shrink ** (pass - 1)`
        times the bound range wide, centered on the value found so far
        and clipped to the bounds. Without this, a second pass restarts
        the same wide bracket from scratch and mostly re-samples points
        the first pass already saw. `1.0` restores full-range passes.
    :param stage_prefix: Prepended to the `stage` label of every logged
        trial, used by `multi_start_search` to tell starts apart.
    :returns: The full `SearchResult` across every trial run, plus one
        `GroupOutcome` per group that was actually touched (a group
        after the target was already reached is skipped and not
        included).
    """
    trials: list[Trial] = []
    current_best_state = dict(base_state)
    with tag_trials(stage=f"{stage_prefix}baseline"):
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
        group_evaluations = 0
        passes_run = 0
        parameter_outcomes: list[ParameterOutcome] = []
        for pass_number in range(1, passes_per_group + 1):
            if current_best_j <= target_j:
                break
            passes_run += 1
            j_before_pass = current_best_j
            for parameter, (bound_low, bound_high) in group_bounds.items():
                if pass_number == 1:
                    low, high = bound_low, bound_high
                else:
                    half_width = (
                        (bound_high - bound_low) * window_shrink ** (pass_number - 1) / 2.0
                    )
                    center = current_best_state[parameter]
                    low = max(bound_low, center - half_width)
                    high = min(bound_high, center + half_width)
                if high <= low:
                    continue

                start_value = current_best_state[parameter]
                j_before_parameter = current_best_j
                evaluations = 0

                def objective_along_one_axis(value: float, name: str = parameter) -> float:
                    nonlocal evaluations
                    # scipy calls this with numpy.float64 values; cast
                    # to native float immediately so every trial logged
                    # from here down, not just the group's eventual best,
                    # is plain-Python and safe for `write_parameters_snapshot`
                    # to yaml.safe_dump.
                    candidate_state = dict(current_best_state)
                    candidate_state[name] = float(value)
                    with tag_trials(
                        group=group,
                        parameters=(name,),
                        stage=f"{stage_prefix}descent/pass{pass_number}",
                    ):
                        j = evaluate(candidate_state)
                    evaluations += 1
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

                group_evaluations += evaluations
                parameter_outcomes.append(
                    ParameterOutcome(
                        group=group,
                        parameter=parameter,
                        pass_index=pass_number,
                        window=(low, high),
                        start_value=start_value,
                        end_value=current_best_state[parameter],
                        starting_j=j_before_parameter,
                        ending_j=current_best_j,
                        evaluations=evaluations,
                    )
                )

            if (j_before_pass - current_best_j) < j_before_pass * min_relative_improvement:
                break

        outcomes.append(
            GroupOutcome(
                group=group,
                starting_j=starting_j,
                ending_j=current_best_j,
                reached_target=current_best_j <= target_j,
                evaluations=group_evaluations,
                passes=passes_run,
                parameter_outcomes=parameter_outcomes,
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
    window_shrink: float = DEFAULT_WINDOW_SHRINK,
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
        window_shrink=window_shrink,
        stage_prefix="start1/" if n_starts > 1 else "",
    )

    rng = random.Random(seed)
    for start_number in range(2, max(1, n_starts) + 1):
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
            window_shrink=window_shrink,
            stage_prefix=f"start{start_number}/",
        )
        if result.best.j < best_result.best.j:
            best_result, best_outcomes = result, outcomes

    return best_result, best_outcomes

"""Tests for `nagcsu.algorithms.coordinate_descent`."""

import pytest

from nagcsu.algorithms import coordinate_descent


def make_two_basin_evaluate():
    """A two-parameter objective where axis-by-axis descent gets stuck.

    `shallow` sits at `(1, 1)` with value 5; `deep` sits at `(9, 9)` with
    value 0. Optimizing one coordinate at a time while the other stays
    near its shallow-basin value never crosses over into `deep`, since
    the other coordinate's own distance from 9 keeps `deep`'s value high
    no matter how the coordinate being tuned is set. Reaching `deep`
    requires both coordinates to already be close to 9 before either is
    individually optimized, exactly the kind of joint effect one
    parameter at a time cannot discover on its own, but a lucky random
    starting point can.
    """

    def evaluate(state: dict[str, float]) -> float:
        x, y = state["x"], state["y"]
        shallow = 5.0 + (x - 1.0) ** 2 + (y - 1.0) ** 2
        deep = (x - 9.0) ** 2 + (y - 9.0) ** 2
        return min(shallow, deep)

    return evaluate


def test_multi_start_search_with_one_start_matches_plain_search() -> None:
    evaluate = make_two_basin_evaluate()
    bounds_by_group = {"group_a": {"x": (0.0, 10.0), "y": (0.0, 10.0)}}

    plain_result, plain_outcomes = coordinate_descent.search(
        {"x": 0.5, "y": 0.5}, ["group_a"], bounds_by_group, evaluate, target_j=0.0
    )
    multi_result, multi_outcomes = coordinate_descent.multi_start_search(
        {"x": 0.5, "y": 0.5}, ["group_a"], bounds_by_group, evaluate, target_j=0.0, n_starts=1
    )

    assert multi_result.best.j == plain_result.best.j
    assert [outcome.ending_j for outcome in multi_outcomes] == [
        outcome.ending_j for outcome in plain_outcomes
    ]


def test_multi_start_search_escapes_a_local_optimum_a_single_start_cannot() -> None:
    evaluate = make_two_basin_evaluate()
    bounds_by_group = {"group_a": {"x": (0.0, 10.0), "y": (0.0, 10.0)}}

    single_start, _ = coordinate_descent.search(
        {"x": 0.5, "y": 0.5}, ["group_a"], bounds_by_group, evaluate, target_j=0.01
    )
    multi_start, _ = coordinate_descent.multi_start_search(
        {"x": 0.5, "y": 0.5},
        ["group_a"],
        bounds_by_group,
        evaluate,
        target_j=0.01,
        n_starts=8,
        seed=0,
    )

    # A single start near (0.5, 0.5) converges to the shallow basin
    # (J = 5), since optimizing one coordinate at a time while the other
    # sits near 1 never makes the deep basin's value competitive.
    assert single_start.best.j == pytest.approx(5.0, abs=1e-6)
    # Enough random starts should land at least one trial close enough
    # to (9, 9) for coordinate descent to converge into the deep basin.
    assert multi_start.best.j < 1.0
    assert multi_start.best.j <= single_start.best.j


def test_multi_start_search_stops_early_once_target_is_already_reached() -> None:
    def always_below_target(state: dict[str, float]) -> float:
        return 0.0

    bounds_by_group = {"group_a": {"x": (0.0, 10.0)}}
    result, outcomes = coordinate_descent.multi_start_search(
        {"x": 5.0},
        ["group_a"],
        bounds_by_group,
        always_below_target,
        target_j=1.0,
        n_starts=5,
    )

    assert result.best.j == pytest.approx(0.0)
    # The first start already meets target_j, so `search` itself should
    # have skipped every group, and no further random starts should run.
    assert outcomes == []
    assert len(result.trials) == 1


def test_search_abandons_a_group_whose_parameter_barely_moves_j() -> None:
    # `x` has real signal; `flat` barely moves J at all (its coefficient
    # is tiny), so passes over the `flat` group should stop well short
    # of `passes_per_group` once min_relative_improvement kicks in.
    def evaluate(state: dict[str, float]) -> float:
        return (state["x"] - 5.0) ** 2 + 1e-6 * state["flat"]

    bounds_by_group = {"flat_group": {"flat": (0.0, 10.0)}}

    with_early_stop, _ = coordinate_descent.search(
        {"x": 0.0, "flat": 0.0},
        ["flat_group"],
        bounds_by_group,
        evaluate,
        target_j=-1.0,
        passes_per_group=10,
        min_relative_improvement=0.5,
    )
    without_early_stop, _ = coordinate_descent.search(
        {"x": 0.0, "flat": 0.0},
        ["flat_group"],
        bounds_by_group,
        evaluate,
        target_j=-1.0,
        passes_per_group=10,
        min_relative_improvement=0.0,
    )

    assert len(with_early_stop.trials) < len(without_early_stop.trials)


def test_max_evaluations_per_parameter_caps_simulations_on_one_parameter() -> None:
    call_count = 0

    def evaluate(state: dict[str, float]) -> float:
        nonlocal call_count
        call_count += 1
        return (state["x"] - 5.0) ** 2

    bounds_by_group = {"group_a": {"x": (0.0, 10.0)}}
    coordinate_descent.search(
        {"x": 0.0},
        ["group_a"],
        bounds_by_group,
        evaluate,
        target_j=0.0,
        passes_per_group=1,
        max_evaluations_per_parameter=3,
    )

    # +1 for search()'s own baseline evaluation of the starting state,
    # which is not itself a per-parameter probe.
    assert call_count <= 4

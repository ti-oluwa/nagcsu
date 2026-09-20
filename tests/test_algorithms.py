"""Tests for `nagcsu.algorithms`, against a cheap synthetic quadratic
objective rather than a real simulation, so these run in milliseconds
and pin down each strategy's search behavior independently of OPM Flow.
"""

import pytest

from nagcsu.algorithms import coordinate_descent, grid, random_search, sensitivity


def quadratic_bowl(state: dict[str, float]) -> float:
    """Minimum at a=5, b=-3; b weighted twice as strongly as a."""
    return (state["a"] - 5.0) ** 2 * 0.001 + (state["b"] + 3.0) ** 2 * 0.002 + 0.01


def test_grid_search_finds_the_best_combination() -> None:
    result = grid.search({"a": 0.0, "b": 0.0}, {"a": [0, 5, 10], "b": [-5, -3, 0]}, quadratic_bowl)
    assert result.best.state == {"a": 5, "b": -3}
    assert result.best.j == pytest.approx(0.01)
    assert len(result.trials) == 9


def test_grid_search_refuses_a_grid_larger_than_the_cap() -> None:
    with pytest.raises(ValueError):
        grid.search(
            {"a": 0.0, "b": 0.0},
            {"a": list(range(30)), "b": list(range(30))},
            quadratic_bowl,
            max_evaluations=100,
        )


def test_random_search_is_reproducible_with_a_seed() -> None:
    result_one = random_search.search(
        {"a": 0.0, "b": 0.0}, {"a": (-10, 10), "b": (-10, 10)}, quadratic_bowl, num_trials=10, seed=42
    )
    result_two = random_search.search(
        {"a": 0.0, "b": 0.0}, {"a": (-10, 10), "b": (-10, 10)}, quadratic_bowl, num_trials=10, seed=42
    )
    assert [trial.state for trial in result_one.trials] == [trial.state for trial in result_two.trials]


def test_coordinate_descent_reaches_the_minimum() -> None:
    result, outcomes = coordinate_descent.search(
        {"a": 0.0, "b": 0.0},
        ["g1"],
        {"g1": {"a": (-10.0, 10.0), "b": (-10.0, 10.0)}},
        quadratic_bowl,
        target_j=0.011,
    )
    assert result.best.j == pytest.approx(0.01, abs=1e-4)
    assert outcomes[0].reached_target


def test_coordinate_descent_stops_at_target_without_touching_later_groups() -> None:
    # Group g1 alone can already reach the (loose) target, so g2 should
    # never be evaluated.
    calls: list[dict[str, float]] = []

    def counting_objective(state: dict[str, float]) -> float:
        calls.append(state)
        return quadratic_bowl(state)

    result, outcomes = coordinate_descent.search(
        {"a": 0.0, "b": 0.0, "c": 0.0},
        ["g1", "g2"],
        {
            "g1": {"a": (-10.0, 10.0), "b": (-10.0, 10.0)},
            "g2": {"c": (-10.0, 10.0)},
        },
        counting_objective,
        target_j=0.05,
    )
    assert [outcome.group for outcome in outcomes] == ["g1"]


def test_sensitivity_ranks_the_more_influential_parameter_first() -> None:
    results, trials = sensitivity.run(
        {"a": 0.0, "b": 0.0}, {"a": (-10.0, 10.0), "b": (-10.0, 10.0)}, quadratic_bowl
    )
    # b's coefficient (0.002) is twice a's (0.001), so it should swing J more.
    assert results[0].parameter == "b"
    assert results[0].swing > results[1].swing
    assert len(trials) == 5  # base + (low, high) for each of 2 parameters

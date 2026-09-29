"""Tests for `nagcsu.algorithms.sensitivity`."""

import pytest

from nagcsu.algorithms import sensitivity
from nagcsu.pipeline import EvaluationBreakdown


def make_detailed_evaluate():
    """A fake evaluate function where each parameter drives a known vector.

    `alpha` only moves `pressure`'s NRMSE; `beta` only moves `gor`'s, and
    moves it far more per unit of parameter value, mimicking a runaway
    GOR term. Combined `J` is just the sum, so `run`'s J-only ranking
    would put `beta` first even though `alpha` is the one actually
    helping pressure.
    """

    def evaluate(state: dict[str, float]) -> EvaluationBreakdown:
        pressure_nrmse = state.get("alpha", 0.0)
        gor_nrmse = state.get("beta", 0.0) * 1000.0
        return EvaluationBreakdown(
            j=pressure_nrmse + gor_nrmse,
            vector_nrmse={"pressure": pressure_nrmse, "gor": gor_nrmse},
        )

    return evaluate


def test_run_detailed_ranks_by_combined_swing_by_default() -> None:
    results = sensitivity.run_detailed(
        {"alpha": 0.0, "beta": 0.0},
        {"alpha": (-1.0, 1.0), "beta": (-1.0, 1.0)},
        make_detailed_evaluate(),
        perturbation_fraction=0.5,
    )

    assert [result.parameter for result in results] == ["beta", "alpha"]


def test_run_detailed_vector_swings_show_beta_only_moves_gor() -> None:
    results = sensitivity.run_detailed(
        {"alpha": 0.0, "beta": 0.0},
        {"alpha": (-1.0, 1.0), "beta": (-1.0, 1.0)},
        make_detailed_evaluate(),
        perturbation_fraction=0.5,
    )
    by_parameter = {result.parameter: result for result in results}

    # beta dominates combined J (see the ranking test above), but its
    # vector_swings show it does nothing for pressure, exactly the
    # distinction a plain J-only ranking would hide.
    assert by_parameter["beta"].vector_swings["pressure"] == pytest.approx(0.0)
    assert by_parameter["beta"].vector_swings["gor"] > 0.0
    assert by_parameter["alpha"].vector_swings["pressure"] > 0.0
    assert by_parameter["alpha"].vector_swings["gor"] == pytest.approx(0.0)


def test_run_detailed_base_j_is_the_same_across_every_result() -> None:
    results = sensitivity.run_detailed(
        {"alpha": 0.25, "beta": 0.0},
        {"alpha": (-1.0, 1.0), "beta": (-1.0, 1.0)},
        make_detailed_evaluate(),
    )

    assert {result.base_j for result in results} == {0.25}


def test_recommend_groups_maps_parameter_names_to_their_groups() -> None:
    from nagcsu.cli.commands import sensitivity as sensitivity_cli

    groups = sensitivity_cli.recommend_groups([
        "aquifer.radius",
        "aquifer.permeability",
        "sgof.sorg",
    ])

    assert groups == ["aquifer", "sgof_shape"]


def test_recommend_groups_respects_the_limit() -> None:
    from nagcsu.cli.commands import sensitivity as sensitivity_cli

    groups = sensitivity_cli.recommend_groups(
        ["aquifer.radius", "sgof.sorg", "swof.residual_oil_saturation"], limit=1
    )

    assert groups == ["aquifer"]

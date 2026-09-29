"""Group ranking, trial tagging, ledger metadata and `match auto` option parsing."""

import click
import pytest
from click.testing import CliRunner

from nagcsu import ledger
from nagcsu.algorithms import coordinate_descent, sensitivity
from nagcsu.algorithms.base import get_current_tag, tag_trials
from nagcsu.cli import display
from nagcsu.cli.app import cli
from nagcsu.cli.commands import match


def _record(
    run_id: str, *, j: float | None, parameter: str | None, group: str | None, value: float = 1.0
):
    return ledger.RunRecord(
        run_id=run_id,
        created_at="2026-09-29T00:00:00",
        parameter_state={},
        group=group,
        strategy="coordinate_descent",
        j=j,
        vector_nrmse=None,
        prt_is_clean=True,
        note="",
        tuned_parameters=[parameter] if parameter else [],
        tuned_values={parameter: value} if parameter else {},
        stage="descent/pass1" if parameter else "baseline",
    )


def test_mean_rank_is_fair_to_small_groups_where_rank_sum_is_not() -> None:
    swings = {"a1": 10.0, "a2": 9.0, "a3": 8.0, "a4": 7.0, "b1": 1.0}
    groups = {"a1": "big", "a2": "big", "a3": "big", "a4": "big", "b1": "small"}

    by_mean = sensitivity.rank_groups(swings, groups, method="mean_rank")
    by_sum = sensitivity.rank_groups(swings, groups, method="rank_sum")
    by_share = sensitivity.rank_groups(swings, groups, method="swing_share")

    assert [g.group for g in by_mean] == ["big", "small"]
    assert [g.group for g in by_sum] == ["small", "big"]  # the size bias
    assert [g.group for g in by_share] == ["big", "small"]
    assert by_mean[0].mean_rank == pytest.approx(2.5)
    assert by_mean[0].parameters == ["a1", "a2", "a3", "a4"]


def test_tied_zero_swings_share_an_average_rank() -> None:
    ranks = sensitivity.rank_parameters({"x": 5.0, "y": 0.0, "z": 0.0})
    assert ranks == {"x": 1.0, "y": 2.5, "z": 2.5}


def test_unknown_group_rank_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        sensitivity.rank_groups({"a": 1.0}, {"a": "g"}, method="nope")


def test_tuning_plan_orders_and_drops_insensitive_parameters() -> None:
    swings = {"a": 1.0, "b": 0.5, "c": 0.001, "d": 0.0}
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g3"}
    ranked = sensitivity.rank_groups(swings, groups)
    plan = sensitivity.build_tuning_plan(ranked, swings, min_relative_swing=0.05)
    assert plan == [("g1", ["a", "b"])]


def test_tag_trials_nests_and_restores() -> None:
    assert get_current_tag() is None
    with tag_trials(group="outer", stage="s1"):
        with tag_trials(group="inner", parameters=["p"], stage="s2"):
            tag = get_current_tag()
            assert tag is not None and tag.group == "inner" and tag.parameters == ("p",)
        outer = get_current_tag()
        assert outer is not None and outer.group == "outer"
    assert get_current_tag() is None


def test_descent_tags_every_trial_with_group_parameter_and_stage() -> None:
    seen: list[tuple[str | None, tuple[str, ...], str | None]] = []

    def evaluate(state: dict[str, float]) -> float:
        tag = get_current_tag()
        seen.append((tag.group, tag.parameters, tag.stage) if tag else (None, (), None))
        return (state["x"] - 3.0) ** 2 + (state["y"] - 1.0) ** 2

    result, outcomes = coordinate_descent.search(
        {"x": 0.0, "y": 0.0},
        ["g"],
        {"g": {"x": (0.0, 10.0), "y": (0.0, 10.0)}},
        evaluate,
        target_j=1e-6,
    )
    assert seen[0] == (None, (), "baseline")
    probes = seen[1:]
    assert {group for group, _, _ in probes} == {"g"}
    assert {parameters for _, parameters, _ in probes} == {("x",), ("y",)}
    assert {stage for _, _, stage in probes} <= {"descent/pass1", "descent/pass2"}
    assert outcomes[0].evaluations == len(probes) - 0 or outcomes[0].evaluations <= len(probes)
    assert {step.parameter for step in outcomes[0].parameter_outcomes} == {"x", "y"}
    assert result.best.j < 1.0


def test_later_passes_search_a_narrower_window() -> None:
    def evaluate(state: dict[str, float]) -> float:
        return abs(state["x"] - 4.0) + 0.5

    _, outcomes = coordinate_descent.search(
        {"x": 0.0},
        ["g"],
        {"g": {"x": (0.0, 10.0)}},
        evaluate,
        target_j=0.0,
        passes_per_group=2,
        min_relative_improvement=0.0,
    )
    steps = outcomes[0].parameter_outcomes
    assert steps[0].window == (0.0, 10.0)
    width_second = steps[1].window[1] - steps[1].window[0]
    assert width_second < 10.0


def test_parameter_and_group_summaries_from_ledger() -> None:
    records = [
        _record("r0", j=1.0, parameter=None, group=None),
        _record("r1", j=0.8, parameter="aquifer.radius", group="aquifer", value=15000),
        _record("r2", j=0.6, parameter="aquifer.radius", group="aquifer", value=17000),
        _record("r3", j=None, parameter="aquifer.radius", group="aquifer", value=99),
    ]
    (history,) = ledger.summarize_parameters(records)
    assert history.trials == 3 and history.failed == 1
    assert history.best_j == 0.6 and history.best_value == 17000
    assert history.j_span == pytest.approx(0.2)
    (group_history,) = ledger.summarize_groups(records)
    assert group_history.parameters_touched == ["aquifer.radius"]
    assert ledger.filter_records(records, parameter="aquifer.radius") == records[1:]
    assert ledger.filter_records(records, stage="base") == records[:1]


def test_old_ledger_rows_without_new_fields_still_load(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    record = _record("r1", j=0.5, parameter=None, group=None)
    ledger.append(path, record)
    (loaded,) = ledger.load(path)
    assert loaded.tuned_parameters == []


def test_parse_range_and_start_options_validate() -> None:
    assert match.parse_range_options(["aquifer.radius=12000:20000"])["aquifer.radius"] == (
        12000.0,
        20000.0,
    )
    with pytest.raises(click.BadParameter):
        match.parse_range_options(["aquifer.radius=20000:12000"])
    with pytest.raises(click.BadParameter):
        match.parse_range_options(["nope=1:2"])
    with pytest.raises(click.BadParameter):
        match.parse_start_options(["aquifer.radius=-5"])


def test_resolve_tuning_space_filters_by_param_and_range() -> None:
    groups, space = match.resolve_tuning_space(
        groups=None, param_names=["aquifer.radius"], ranges={"aquifer.radius": (12000.0, 20000.0)}
    )
    assert groups == ["aquifer"]
    assert space["aquifer"] == {"aquifer.radius": (12000.0, 20000.0)}
    with pytest.raises(click.UsageError):
        match.resolve_tuning_space(
            groups=["permeability"], param_names=["aquifer.radius"], ranges={}
        )


def test_list_parameters_renders_a_rich_table() -> None:
    result = CliRunner().invoke(cli, ["match", "list-parameters"])
    assert result.exit_code == 0, result.output
    assert "Tunable parameters" in result.output
    assert "aquifer" in result.output


def test_ledger_and_history_tables_render() -> None:
    records = [
        _record("r0", j=1.0, parameter=None, group=None),
        _record("r1", j=0.8, parameter="aquifer.radius", group="aquifer", value=15000),
    ]
    display.console.print(display.ledger_table(records))
    display.console.print(display.parameter_history_table(ledger.summarize_parameters(records)))
    display.console.print(display.group_history_table(ledger.summarize_groups(records)))

"""Failed-probe sensitivity, `nagcsu clean`, run ID safety and range suggestions."""

import math

import click.testing
import pytest

from nagcsu import cleanup, ledger, ranges
from nagcsu import config as config_module
from nagcsu.algorithms import sensitivity
from nagcsu.cli.app import cli
from nagcsu.deck import Deck

# ---------------------------------------------------------------- sensitivity


def test_probe_swing_estimates_from_the_surviving_probe_and_never_returns_inf() -> None:
    assert sensitivity.probe_swing(1.0, 0.5, 2.0) == (1.5, 0)
    assert sensitivity.probe_swing(1.0, math.inf, 1.4) == (pytest.approx(0.8), 1)
    assert sensitivity.probe_swing(1.0, 1.4, math.nan) == (pytest.approx(0.8), 1)
    assert sensitivity.probe_swing(1.0, math.inf, math.inf) == (0.0, 2)
    assert sensitivity.probe_swing(math.inf, 1.0, 2.0) == (1.0, 0)


def test_run_survives_a_failed_probe_and_keeps_shares_finite() -> None:
    def evaluate(state: dict[str, float]) -> float:
        if state["a"] > 1.5:
            return math.inf  # simulation failed at the high probe
        return abs(state["a"] - 1.0) * 2 + abs(state["b"] - 1.0) * 0.5

    results, _ = sensitivity.run(
        {"a": 1.0, "b": 1.0},
        {"a": (0.0, 2.0), "b": (0.0, 2.0)},
        evaluate,
        perturbation_fraction=0.4,
    )
    by_name = {result.parameter: result for result in results}
    assert by_name["a"].failed_probes == 1
    assert all(math.isfinite(result.swing) for result in results)
    swings = {result.parameter: result.swing for result in results}
    groups = sensitivity.rank_groups(swings, {"a": "ga", "b": "gb"}, method="swing_share")
    assert all(math.isfinite(group.swing_share) for group in groups)
    assert sum(group.swing_share for group in groups) == pytest.approx(1.0)


def test_rank_groups_treats_non_finite_swings_as_zero() -> None:
    groups = sensitivity.rank_groups(
        {"a": math.inf, "b": 1.0, "c": math.nan}, {"a": "g1", "b": "g2", "c": "g3"}
    )
    assert all(
        math.isfinite(group.total_swing) and math.isfinite(group.swing_share) for group in groups
    )


# ---------------------------------------------------------------- run ids


def _record(
    run_id: str,
    *,
    j: float | None = 1.0,
    strategy: str | None = None,
    stage: str | None = None,
    parameter: str | None = None,
    value: float = 1.0,
    error: str | None = None,
) -> ledger.RunRecord:
    return ledger.RunRecord(
        run_id=run_id,
        created_at="t",
        parameter_state={},
        group="aquifer" if parameter else None,
        strategy=strategy,
        j=j,
        vector_nrmse=None,
        prt_is_clean=True,
        note="",
        simulation_error=error,
        stage=stage,
        tuned_parameters=[parameter] if parameter else [],
        tuned_values={parameter: value} if parameter else {},
    )


def test_new_run_id_never_reuses_an_id_after_records_were_cleaned() -> None:
    kept = [_record("run_0001"), _record("run_0002")]
    assert ledger.new_run_id(kept) == "run_0003"
    assert ledger.new_run_id([], taken_ids=["run_0007", "auto_final"]) == "run_0008"
    assert ledger.new_run_id([]) == "run_0000"


# ---------------------------------------------------------------- cleanup logic


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    records = [
        _record("run_0000", j=1.0),
        _record("sensitivity_00000", j=0.9, strategy="sensitivity", stage="sensitivity/low"),
        _record("auto_00001", j=0.3, strategy="coordinate_descent", stage="descent/pass1"),
        _record("auto_final", j=0.2, strategy="coordinate_descent", stage="final"),
        _record("bad_00002", j=None, error="boom"),
    ]
    for record in records:
        directory = root / record.run_id
        directory.mkdir(parents=True)
        (directory / "CASE.DATA").write_text("deck")
        (directory / "CASE.UNSMRY").write_text("summary")
        (directory / "CASE.PRT").write_text("x" * 100)
        (directory / "sub").mkdir()
        (directory / "sub" / "big.log").write_text("log")
    (root / "orphan_dir").mkdir()
    (root / "orphan_dir" / "a.txt").write_text("a")
    return root, records


def test_selection_is_and_across_selectors_and_or_within_one(runs) -> None:
    root, records = runs
    targets = cleanup.collect_targets(records, root)
    assert [t.run_id for t in targets][-1] == "orphan_dir"  # directory with no record

    chosen, _ = cleanup.select(targets, cleanup.Selection(prefixes=("sensitivity", "bad")))
    assert {t.run_id for t in chosen} == {"sensitivity_00000", "bad_00002"}

    chosen, _ = cleanup.select(
        targets, cleanup.Selection(prefixes=("sensitivity", "bad"), failed_only=True)
    )
    assert {t.run_id for t in chosen} == {"bad_00002"}

    chosen, _ = cleanup.select(targets, cleanup.Selection(sensitivity_only=True))
    assert {t.run_id for t in chosen} == {"sensitivity_00000"}

    chosen, _ = cleanup.select(
        targets, cleanup.Selection(strategies=("coordinate_descent",), stages=("descent",))
    )
    assert {t.run_id for t in chosen} == {"auto_00001"}

    chosen, _ = cleanup.select(targets, cleanup.Selection(regexes=(r"auto_\d+",)))
    assert {t.run_id for t in chosen} == {"auto_00001"}


def test_keep_rules_protect_and_report_why(runs) -> None:
    root, records = runs
    targets = cleanup.collect_targets(records, root)
    chosen, protected = cleanup.select(
        targets,
        cleanup.Selection(
            select_all=True, keep_best=1, keep_ids=("run_0000",), keep_prefixes=("bad",)
        ),
    )
    assert {t.run_id for t, _ in protected} == {"auto_final", "run_0000", "bad_00002"}
    assert "auto_final" not in {t.run_id for t in chosen}
    assert "orphan_dir" in {t.run_id for t in chosen}
    _latest_chosen, latest_protected = cleanup.select(
        targets, cleanup.Selection(select_all=True, keep_latest=1)
    )
    assert "bad_00002" in {t.run_id for t, _ in latest_protected}


def test_no_selector_selects_nothing() -> None:
    assert not cleanup.Selection().has_selector()
    assert cleanup.Selection(sensitivity_only=True).has_selector()


def test_execute_scope_both_removes_records_and_directories(runs, tmp_path) -> None:
    root, records = runs
    ledger_path = tmp_path / "runs" / "ledger.json"
    for record in records:
        ledger.append(ledger_path, record)
    targets = cleanup.collect_targets(records, root)
    chosen, _ = cleanup.select(targets, cleanup.Selection(sensitivity_only=True))
    actions = cleanup.plan(chosen, scope="both", keep_files=[], output_root=root)
    result = cleanup.execute(actions, ledger_path=ledger_path, output_root=root)
    assert result.records_removed == 1 and result.directories_removed == 1
    assert not (root / "sensitivity_00000").exists()
    assert "sensitivity_00000" not in [r.run_id for r in ledger.load(ledger_path)]
    assert (root / "run_0000").exists()


def test_execute_keep_files_leaves_matching_files_and_drops_empty_subdirs(runs, tmp_path) -> None:
    root, records = runs
    ledger_path = root / "ledger.json"
    for record in records:
        ledger.append(ledger_path, record)
    targets = cleanup.collect_targets(records, root)
    chosen, _ = cleanup.select(targets, cleanup.Selection(prefixes=("auto_00001",)))
    actions = cleanup.plan(
        chosen, scope="both", keep_files=["*.DATA", "*.UNSMRY"], output_root=root
    )
    assert len(actions[0].files_to_keep) == 2 and not actions[0].delete_whole_directory
    cleanup.execute(actions, ledger_path=ledger_path, output_root=root)
    remaining = sorted(path.name for path in (root / "auto_00001").rglob("*"))
    assert remaining == ["CASE.DATA", "CASE.UNSMRY"]
    assert "auto_00001" not in [r.run_id for r in ledger.load(ledger_path)]


def test_scope_files_keeps_records_and_scope_ledger_keeps_files(runs) -> None:
    root, records = runs
    ledger_path = root / "ledger.json"
    for record in records:
        ledger.append(ledger_path, record)
    targets = cleanup.collect_targets(records, root)
    chosen, _ = cleanup.select(targets, cleanup.Selection(ids=("run_0000",)))

    only_files = cleanup.plan(chosen, scope="files", keep_files=["*.PRT"], output_root=root)
    cleanup.execute(only_files, ledger_path=ledger_path, output_root=root)
    assert [p.name for p in (root / "run_0000").rglob("*") if p.is_file()] == ["CASE.PRT"]
    assert "run_0000" in [r.run_id for r in ledger.load(ledger_path)]

    only_ledger = cleanup.plan(chosen, scope="ledger", keep_files=[], output_root=root)
    cleanup.execute(only_ledger, ledger_path=ledger_path, output_root=root)
    assert (root / "run_0000").exists()
    assert "run_0000" not in [r.run_id for r in ledger.load(ledger_path)]
    with pytest.raises(ValueError):
        cleanup.plan(chosen, scope="ledger", keep_files=["*.DATA"], output_root=root)


def test_execute_never_deletes_outside_the_output_root(runs, tmp_path) -> None:
    root, records = runs
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("do not delete")
    link = root / "linked"
    link.symlink_to(outside, target_is_directory=True)
    targets = cleanup.collect_targets(records, root)
    chosen, _ = cleanup.select(targets, cleanup.Selection(ids=("linked",)))
    actions = cleanup.plan(chosen, scope="both", keep_files=[], output_root=root)
    cleanup.execute(actions, ledger_path=root / "ledger.json", output_root=root)
    assert (outside / "keep.txt").exists()


# ---------------------------------------------------------------- clean CLI


@pytest.fixture
def project(tmp_path, sample_deck: Deck):
    project_config = config_module.ProjectConfig(
        deck_path=sample_deck.path,
        output_root=tmp_path / "runs",
        ledger_path=tmp_path / "runs" / "ledger.json",
        root=tmp_path,
    )
    config_path = tmp_path / "nagcsu.yaml"
    config_module.save(project_config, config_path)
    for run_id, strategy in (("run_0000", None), ("sensitivity_00000", "sensitivity")):
        (tmp_path / "runs" / run_id).mkdir(parents=True)
        (tmp_path / "runs" / run_id / "CASE.DATA").write_text("d")
        ledger.append(tmp_path / "runs" / "ledger.json", _record(run_id, strategy=strategy))
    return config_path, tmp_path


def _invoke(config_path, *args):
    return click.testing.CliRunner().invoke(cli, ["--config", str(config_path), "clean", *args])


def test_clean_cli_requires_a_selector_and_supports_dry_run(project) -> None:
    config_path, tmp_path = project
    assert _invoke(config_path).exit_code != 0
    result = _invoke(config_path, "--sensitivity", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert (tmp_path / "runs" / "sensitivity_00000").exists()


def test_clean_cli_removes_sensitivity_runs_with_yes(project) -> None:
    config_path, tmp_path = project
    result = _invoke(config_path, "--sensitivity", "--yes")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "runs" / "sensitivity_00000").exists()
    assert [r.run_id for r in ledger.load(tmp_path / "runs" / "ledger.json")] == ["run_0000"]


def test_clean_cli_asks_before_deleting(project) -> None:
    config_path, tmp_path = project
    result = click.testing.CliRunner().invoke(
        cli, ["--config", str(config_path), "clean", "--all"], input="n\n"
    )
    assert "Cancelled" in result.output
    assert (tmp_path / "runs" / "run_0000").exists()


# ---------------------------------------------------------------- ranges


def _trials(values_to_j: dict[float, float]) -> list[ledger.RunRecord]:
    return [
        _record(f"r{i}", j=j, parameter="aquifer.radius", value=value)
        for i, (value, j) in enumerate(values_to_j.items())
    ]


def test_bracketed_optimum_gets_a_range_around_the_basin() -> None:
    records = _trials({12000: 0.9, 14000: 0.5, 16000: 0.3, 18000: 0.32, 20000: 0.8, 22000: 1.0})
    (suggestion,) = ranges.suggest_ranges(records)
    assert suggestion.status == "bracketed"
    assert suggestion.best_value == 16000
    assert suggestion.low == pytest.approx(15000)  # halfway to the worse neighbour at 14000
    assert suggestion.high == pytest.approx(19000)
    assert suggestion.range_flag() == "--range aquifer.radius=15000:19000"


def test_best_at_the_upper_edge_extends_past_it_instead_of_only_saying_widen() -> None:
    records = _trials({10000: 1.0, 12000: 0.8, 14000: 0.6, 16000: 0.4, 18000: 0.2})
    (suggestion,) = ranges.suggest_ranges(records)
    assert suggestion.status == "open_high"
    assert suggestion.high > 18000
    assert suggestion.low >= 14000


def test_flat_and_insufficient_parameters_get_no_range() -> None:
    flat = ranges.suggest_ranges(_trials({1: 0.500, 2: 0.501, 3: 0.500, 4: 0.502}))
    assert flat[0].status == "flat" and flat[0].range_flag() is None
    few = ranges.suggest_ranges(_trials({1: 0.5, 2: 0.4}))
    assert few[0].status == "insufficient"


def test_ranges_ignore_failed_and_multi_parameter_trials() -> None:
    records = _trials({1: 1.0, 2: 0.8, 3: 0.4, 4: 0.6, 5: 0.9})
    records.append(_record("failed", j=None, parameter="aquifer.radius", value=99))
    records.append(_record("multi", j=0.01, parameter=None))
    (suggestion,) = ranges.suggest_ranges(records)
    assert suggestion.trials == 5 and suggestion.best_value == 3

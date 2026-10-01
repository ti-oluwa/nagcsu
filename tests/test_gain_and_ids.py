"""Direction-aware sensitivity ranking, and the duplicate run ID crash in `clean`."""

import math

import click.testing
import pytest

from nagcsu import cleanup, ledger, parameters, pipeline
from nagcsu import config as config_module
from nagcsu.algorithms import sensitivity
from nagcsu.cli.app import cli
from nagcsu.deck import Deck


# ------------------------------------------------------------------ gain


def test_probe_gain_only_counts_probes_that_beat_the_base() -> None:
    assert sensitivity.probe_gain(1.0, 0.7, 1.3) == (pytest.approx(0.3), "low")
    assert sensitivity.probe_gain(1.0, 1.2, 0.9) == (pytest.approx(0.1), "high")
    assert sensitivity.probe_gain(1.0, 1.2, 1.4) == (0.0, None)  # both worse: no gain
    assert sensitivity.probe_gain(1.0, math.inf, 0.8) == (pytest.approx(0.2), "high")
    assert sensitivity.probe_gain(1.0, math.inf, math.inf) == (0.0, None)
    assert sensitivity.probe_gain(math.inf, 1.0, 2.0) == (0.0, None)


def test_gap_closed_is_capped_and_needs_a_target() -> None:
    assert sensitivity.gap_closed(0.4, 0.1, 0.1) == pytest.approx(1 / 3)
    assert sensitivity.gap_closed(0.4, 0.9, 0.1) == 1.0
    assert sensitivity.gap_closed(0.4, 0.1, None) is None
    assert sensitivity.gap_closed(0.05, 0.01, 0.1) is None  # already at the target


def test_big_swing_that_only_hurts_no_longer_outranks_a_parameter_that_helps() -> None:
    swings = {"hurts": 5.0, "helps": 0.2}
    gains = {"hurts": 0.0, "helps": 0.1}
    groups = {"hurts": "g_hurt", "helps": "g_help"}
    by_swing = sensitivity.rank_groups(swings, groups)
    by_gain = sensitivity.rank_groups(swings, groups, parameter_gains=gains)
    assert by_swing[0].group == "g_hurt"  # the old behaviour
    assert by_gain[0].group == "g_help"
    assert by_gain[0].ranked_by == "gain"
    ranks = sensitivity.rank_parameters(gains, tiebreak=swings)
    assert ranks == {"helps": 1.0, "hurts": 2.0}


def test_equal_gains_fall_back_to_swing_and_full_ties_share_a_rank() -> None:
    ranks = sensitivity.rank_parameters(
        {"a": 0.0, "b": 0.0, "c": 0.0}, tiebreak={"a": 1.0, "b": 3.0, "c": 3.0}
    )
    assert ranks == {"b": 1.5, "c": 1.5, "a": 3.0}


def test_gain_share_method_needs_gains_and_orders_by_total_gain() -> None:
    swings = {"a": 1.0, "b": 1.0}
    groups = {"a": "ga", "b": "gb"}
    with pytest.raises(ValueError):
        sensitivity.rank_groups(swings, groups, method="gain_share")
    ordered = sensitivity.rank_groups(
        swings, groups, method="gain_share", parameter_gains={"a": 0.1, "b": 0.4}
    )
    assert [g.group for g in ordered] == ["gb", "ga"]
    assert ordered[0].gain_share == pytest.approx(0.8)


def test_summarize_defaults_to_gain_and_swing_is_still_available() -> None:
    def evaluate(state: dict[str, float]) -> float:
        # `far` has a distant optimum (one step helps a lot); `near` sits at its optimum.
        return (state["far"] - 8.0) ** 2 * 0.002 + (state["near"] - 0.0) ** 2 * 0.01 + 0.05

    results, _ = sensitivity.run(
        {"far": 0.0, "near": 0.0},
        {"far": (-10.0, 10.0), "near": (-10.0, 10.0)},
        evaluate,
        target_j=0.05,
    )
    assert results[0].parameter == "far" and results[0].best_side == "high"
    assert results[1].gain == 0.0 and results[1].best_side is None
    assert results[0].gap_closed is not None and 0 < results[0].gap_closed <= 1
    groups = {"far": "g1", "near": "g2"}
    gain_ranks, _ = sensitivity.summarize(results, groups)
    swing_ranks, _ = sensitivity.summarize(results, groups, rank_by="swing")
    assert gain_ranks["far"] == 1.0
    assert set(swing_ranks) == {"far", "near"}
    with pytest.raises(ValueError):
        sensitivity.summarize(results, groups, rank_by="nope")


def test_tuning_plan_keeps_sensitive_parameters_with_no_gain_but_orders_them_late() -> None:
    swings = {"helps": 0.3, "hurts": 5.0, "dead": 0.0}
    gains = {"helps": 0.2, "hurts": 0.0, "dead": 0.0}
    groups = {"helps": "g", "hurts": "g", "dead": "g"}
    ranked = sensitivity.rank_groups(swings, groups, parameter_gains=gains)
    plan = sensitivity.build_tuning_plan(ranked, swings, min_relative_swing=0.02)
    assert plan == [("g", ["helps", "hurts"])]  # `dead` barely moves J and is dropped


# ------------------------------------------------------------------ run ids


def _record(run_id: str, *, j: float | None = None, error: str | None = None) -> ledger.RunRecord:
    return ledger.RunRecord(
        run_id=run_id,
        created_at="t",
        parameter_state={},
        group=None,
        strategy="sweep",
        j=j,
        vector_nrmse=None,
        prt_is_clean=None,
        note="",
        simulation_error=error,
        stage="sweep",
    )


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
    return project_config, config_path, tmp_path


def test_clean_failed_with_duplicate_ids_no_longer_crashes(project) -> None:
    """Two ledger records sharing one run directory used to crash the second rmtree."""
    _, config_path, tmp_path = project
    ledger_path = tmp_path / "runs" / "ledger.json"
    run_dir = tmp_path / "runs" / "sweep_00000"
    run_dir.mkdir(parents=True)
    (run_dir / "CASE.DATA").write_text("deck")
    ledger.append(ledger_path, _record("sweep_00000", error="boom"))
    ledger.append(ledger_path, _record("sweep_00000", error="boom again"))
    ledger.append(ledger_path, _record("run_0000", j=0.5))

    result = click.testing.CliRunner().invoke(
        cli, ["--config", str(config_path), "clean", "--failed", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert not run_dir.exists()
    assert [record.run_id for record in ledger.load(ledger_path)] == ["run_0000"]
    assert "Removed 2 record(s)" in result.output


def test_duplicate_ids_collapse_to_one_target_with_the_latest_record(tmp_path) -> None:
    root = tmp_path / "runs"
    (root / "sweep_00000").mkdir(parents=True)
    records = [_record("sweep_00000", j=0.9), _record("sweep_00000", error="boom")]
    (target,) = cleanup.collect_targets(records, root)
    assert target.record_count == 2 and target.record.simulation_error == "boom"


def test_execute_is_safe_when_a_directory_vanished_between_plan_and_run(tmp_path) -> None:
    root = tmp_path / "runs"
    run_dir = root / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "f.txt").write_text("x")
    ledger_path = root / "ledger.json"
    ledger.append(ledger_path, _record("r1", j=1.0))
    targets = cleanup.collect_targets(ledger.load(ledger_path), root)
    actions = cleanup.plan(targets, scope="both", keep_files=[], output_root=root)
    import shutil

    shutil.rmtree(run_dir)
    result = cleanup.execute(actions, ledger_path=ledger_path, output_root=root)
    assert result.records_removed == 1 and result.directories_removed == 0


def test_run_id_allocator_skips_ids_in_the_ledger_and_on_disk(project) -> None:
    project_config, _, tmp_path = project
    ledger.append(tmp_path / "runs" / "ledger.json", _record("sweep_00000", j=1.0))
    (tmp_path / "runs" / "sweep_00001").mkdir(parents=True)
    allocate = pipeline.make_run_id_allocator(project_config, "sweep")
    assert [allocate(), allocate(), allocate()] == ["sweep_00002", "sweep_00003", "sweep_00004"]
    fresh = pipeline.make_run_id_allocator(project_config, "other")
    assert fresh() == "other_00000"

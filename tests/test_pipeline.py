"""Tests for `nagcsu.pipeline`.

The most important behavior covered here: a failed simulation must
never raise out of `execute_run`, since `nagcsu.algorithms` search
strategies call `evaluate()` (built on top of `execute_run`) potentially
hundreds of times in a loop, and one bad parameter combination crashing
OPM Flow should end that one trial, not the whole search.
"""

import pandas
import pytest

from nagcsu import config, pipeline
from nagcsu.deck import Deck
from nagcsu.exceptions import SimulationError


@pytest.fixture
def project_config(tmp_path, sample_deck: Deck) -> config.ProjectConfig:
    return config.ProjectConfig(
        deck_path=sample_deck.path,
        output_root=tmp_path / "runs",
        ledger_path=tmp_path / "runs" / "ledger.json",
        root=tmp_path,
    )


def test_execute_run_reports_simulation_failure_without_raising(
    monkeypatch, project_config: config.ProjectConfig, sample_deck: Deck
) -> None:
    def fake_run(*args, **kwargs):
        raise SimulationError("flow exited 1 and wrote no summary output", returncode=1)

    monkeypatch.setattr(pipeline.simulate, "run", fake_run)

    outcome = pipeline.execute_run(project_config, sample_deck, {}, run_id="run_fail", score=True)

    assert outcome.simulation_error is not None
    assert "flow exited 1" in outcome.simulation_error
    assert outcome.objective_result is None
    assert outcome.prt_report is None
    # The deck should still have been written, even though the run failed,
    # so the failing parameter state can be inspected by hand.
    assert outcome.deck_path.exists()


def test_make_evaluate_returns_inf_for_a_failed_simulation(
    monkeypatch, project_config: config.ProjectConfig, sample_deck: Deck
) -> None:
    def fake_run(*args, **kwargs):
        raise SimulationError("boom")

    monkeypatch.setattr(pipeline.simulate, "run", fake_run)

    seen_outcomes = []
    evaluate = pipeline.make_evaluate(project_config, sample_deck, on_outcome=seen_outcomes.append)

    result = evaluate({"aquifer.radius": 20000.0})

    assert result == float("inf")
    assert len(seen_outcomes) == 1
    assert seen_outcomes[0].simulation_error is not None


def test_to_run_record_carries_simulation_error_through(
    project_config: config.ProjectConfig, sample_deck: Deck, monkeypatch
) -> None:
    def fake_run(*args, **kwargs):
        raise SimulationError("boom")

    monkeypatch.setattr(pipeline.simulate, "run", fake_run)

    outcome = pipeline.execute_run(project_config, sample_deck, {}, run_id="run_fail")
    record = pipeline.to_run_record(outcome, group=None, strategy=None, note="")

    assert record.simulation_error == "boom"
    assert record.j is None


def test_execute_run_scores_a_successful_simulation(
    monkeypatch, project_config: config.ProjectConfig, sample_deck: Deck
) -> None:
    def fake_simulate_run(deck_path, output_dir, *, flow_executable="flow", **kwargs):
        from nagcsu import simulate

        output_dir.mkdir(parents=True, exist_ok=True)
        case_basename = output_dir / "FAKECASE"
        (case_basename.with_suffix(".UNSMRY")).write_bytes(b"\x00")
        return simulate.RunResult(
            output_dir=output_dir,
            case_basename=case_basename,
            returncode=0,
            stdout="",
            stderr="",
            elapsed_seconds=0.01,
        )

    def fake_load_summary(case_basename, *, wells=None):
        return pandas.DataFrame({
            "DATE": pandas.date_range("1976-01-01", periods=2, freq="YS"),
            "FPR": [2751.0, 2700.0],
            "FWCT": [0.0, 0.05],
            "FGOR": [818.0, 820.0],
        })

    def fake_load_observed_history(path, **kwargs):
        return pandas.DataFrame({
            "DATE": pandas.date_range("1976-01-01", periods=2, freq="YS"),
            "FPR": [2751.0, 2695.0],
            "FWCT": [0.0, 0.04],
            "FGOR": [818.0, 819.0],
        })

    monkeypatch.setattr(pipeline.simulate, "run", fake_simulate_run)
    monkeypatch.setattr(pipeline.summary, "load_summary", fake_load_summary)
    monkeypatch.setattr(pipeline.history, "load_observed_history", fake_load_observed_history)

    outcome = pipeline.execute_run(project_config, sample_deck, {}, run_id="run_ok")

    assert outcome.simulation_error is None
    assert outcome.objective_result is not None
    assert outcome.objective_result.j > 0

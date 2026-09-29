"""CLI-level smoke tests for `nagcsu plot` and `nagcsu match sweep`."""

import click.testing
import pytest

from nagcsu import config as config_module
from nagcsu import ledger
from nagcsu.cli.app import cli
from nagcsu.deck import Deck


@pytest.fixture
def project_config_path(tmp_path, sample_deck: Deck) -> "config_module.pathlib.Path":
    project_config = config_module.ProjectConfig(
        deck_path=sample_deck.path,
        output_root=tmp_path / "runs",
        ledger_path=tmp_path / "runs" / "ledger.json",
        root=tmp_path,
    )
    config_path = tmp_path / "nagcsu.yaml"
    config_module.save(project_config, config_path)
    return config_path


def get_scored_record(run_id: str, j: float) -> ledger.RunRecord:
    return ledger.RunRecord(
        run_id=run_id,
        created_at=ledger.timestamp_now(),
        parameter_state={"aquifer.radius": 16137.2},
        group="aquifer",
        strategy="coordinate_descent",
        j=j,
        vector_nrmse={"pressure": 0.3, "watercut": 0.4, "gor": 0.2},
        prt_is_clean=True,
        note="",
    )


def test_plot_convergence_writes_an_html_file_from_the_ledger(
    tmp_path, project_config_path
) -> None:
    project_config = config_module.load(project_config_path)
    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    ledger.append(ledger_path, get_scored_record("run_0000", j=1.0))
    ledger.append(ledger_path, get_scored_record("run_0001", j=0.5))
    output_path = tmp_path / "convergence.html"

    result = click.testing.CliRunner().invoke(
        cli,
        [
            "--config",
            str(project_config_path),
            "plot",
            "convergence",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_plot_convergence_with_no_matching_records_is_a_clean_error(
    tmp_path, project_config_path
) -> None:
    result = click.testing.CliRunner().invoke(
        cli,
        [
            "--config",
            str(project_config_path),
            "plot",
            "convergence",
            "--strategy",
            "nonexistent",
        ],
    )

    assert result.exit_code != 0
    assert "No matching ledger records" in result.output


def test_report_list_filters_by_run_id_prefix(project_config_path) -> None:
    project_config = config_module.load(project_config_path)
    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    ledger.append(ledger_path, get_scored_record("run_0000", j=1.0))
    ledger.append(ledger_path, get_scored_record("auto_0000", j=0.5))

    result = click.testing.CliRunner().invoke(
        cli,
        [
            "--config",
            str(project_config_path),
            "report",
            "list",
            "--run-id-prefix",
            "auto",
            "--limit",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "auto_0000" in result.output
    assert "run_0000" not in result.output


def test_plot_match_writes_an_html_file_for_a_run(
    tmp_path, project_config_path, monkeypatch
) -> None:
    import pandas

    from nagcsu.cli.commands import plot as plot_module

    project_config = config_module.load(project_config_path)
    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    ledger.append(ledger_path, get_scored_record("run_0000", j=1.0))

    output_dir = project_config.get_resolved_path(project_config.output_root) / "run_0000"
    output_dir.mkdir(parents=True)
    (output_dir / "CASE.UNSMRY").write_bytes(b"\x00")

    def fake_load_summary(case_basename, *, wells=None):
        return pandas.DataFrame({
            "DATE": pandas.date_range("1976-01-01", periods=2, freq="YS"),
            "FPR": [2751.0, 2700.0],
            "FWCT": [0.0, 0.05],
            "FGOR": [0.818, 0.82],
        })

    def fake_load_observed_history(config):
        return pandas.DataFrame({
            "DATE": pandas.date_range("1976-01-01", periods=2, freq="YS"),
            "FPR": [2751.0, 2695.0],
            "FWCT": [0.0, 0.04],
            "FGOR": [0.818, 0.819],
        })

    monkeypatch.setattr(plot_module.summary, "load_summary", fake_load_summary)
    monkeypatch.setattr(plot_module.pipeline, "load_observed_history", fake_load_observed_history)
    output_path = tmp_path / "match.html"

    result = click.testing.CliRunner().invoke(
        cli,
        [
            "--config",
            str(project_config_path),
            "plot",
            "match",
            "run_0000",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_sweep_raises_once_more_values_than_max_evaluations_are_given(
    tmp_path, project_config_path
) -> None:
    result = click.testing.CliRunner().invoke(
        cli,
        [
            "--config",
            str(project_config_path),
            "match",
            "sweep",
            "--param",
            "aquifer.radius",
            "--values",
            "10000,20000,30000",
            "--max-evaluations",
            "2",
        ],
    )

    assert result.exit_code != 0
    assert "max_evaluations" in str(result.exception)

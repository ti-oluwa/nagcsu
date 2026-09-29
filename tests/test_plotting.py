"""Tests for `nagcsu.plotting`."""

import pandas
import pytest

from nagcsu import ledger, plotting


def get_scored_record(run_id: str, j: float, vector_nrmse: dict[str, float]) -> ledger.RunRecord:
    return ledger.RunRecord(
        run_id=run_id,
        created_at=ledger.timestamp_now(),
        parameter_state={"aquifer.radius": 16137.2},
        group="aquifer",
        strategy="coordinate_descent",
        j=j,
        vector_nrmse=vector_nrmse,
        prt_is_clean=True,
        note="",
    )


def test_convergence_figure_raises_when_nothing_is_scored() -> None:
    unscored = [
        ledger.RunRecord(
            run_id="run_0000",
            created_at=ledger.timestamp_now(),
            parameter_state={},
            group=None,
            strategy=None,
            j=None,
            vector_nrmse=None,
            prt_is_clean=None,
            note="",
        )
    ]
    with pytest.raises(ValueError, match="No scored records"):
        plotting.plot_convergence(unscored)


def test_convergence_figure_has_one_row_per_vector_plus_j() -> None:
    records = [
        get_scored_record(
            "run_0000", j=1.0, vector_nrmse={"pressure": 0.5, "watercut": 0.3, "gor": 0.2}
        ),
        get_scored_record(
            "run_0001", j=0.8, vector_nrmse={"pressure": 0.4, "watercut": 0.25, "gor": 0.15}
        ),
    ]

    figure = plotting.plot_convergence(records)

    # J plus one row per SCORED_FIELD_VECTORS entry (pressure, watercut, gor).
    assert len(figure.data) == 4
    assert list(figure.data[0].y) == [1.0, 0.8]


def test_history_match_figure_plots_simulated_and_observed_per_vector() -> None:
    simulated = pandas.DataFrame({
        "DATE": pandas.date_range("1976-01-01", periods=3, freq="YS"),
        "FPR": [2751.0, 2700.0, 2650.0],
        "FWCT": [0.0, 0.05, 0.1],
        "FGOR": [0.818, 0.82, 0.83],
    })
    observed = pandas.DataFrame({
        "DATE": pandas.date_range("1976-01-01", periods=3, freq="YS"),
        "FPR": [2751.0, 2695.0, 2640.0],
        "FWCT": [0.0, 0.04, 0.09],
        "FGOR": [0.818, 0.819, 0.825],
    })

    figure = plotting.plot_match(simulated, observed)

    # One simulated + one observed trace per vector (pressure, watercut, gor).
    assert len(figure.data) == 6


def test_save_figure_writes_a_standalone_html_file(tmp_path) -> None:
    figure = plotting.plot_convergence([
        get_scored_record(
            "run_0000", j=1.0, vector_nrmse={"pressure": 0.5, "watercut": 0.3, "gor": 0.2}
        )
    ])
    output_path = tmp_path / "out.html"

    written = plotting.save_figure(figure, output_path)

    assert written == output_path
    content = output_path.read_text()
    assert "plotly" in content.lower()


def test_save_figure_adds_html_suffix_when_none_given(tmp_path) -> None:
    figure = plotting.plot_convergence([
        get_scored_record(
            "run_0000", j=1.0, vector_nrmse={"pressure": 0.5, "watercut": 0.3, "gor": 0.2}
        )
    ])

    written = plotting.save_figure(figure, tmp_path / "out")

    assert written == tmp_path / "out.html"
    assert written.exists()


def test_save_figure_turns_a_missing_kaleido_package_into_a_clear_import_error(
    tmp_path, monkeypatch
) -> None:
    figure = plotting.plot_convergence([
        get_scored_record(
            "run_0000", j=1.0, vector_nrmse={"pressure": 0.5, "watercut": 0.3, "gor": 0.2}
        )
    ])

    def fake_write_image(self, *args, **kwargs):
        raise ValueError("Kaleido is required for image export.")

    monkeypatch.setattr(type(figure), "write_image", fake_write_image)

    with pytest.raises(ImportError, match="pip install kaleido"):
        plotting.save_figure(figure, tmp_path / "out.png")


def test_save_figure_turns_a_missing_chrome_into_a_clear_import_error(
    tmp_path, monkeypatch
) -> None:
    figure = plotting.plot_convergence([
        get_scored_record(
            "run_0000", j=1.0, vector_nrmse={"pressure": 0.5, "watercut": 0.3, "gor": 0.2}
        )
    ])

    def fake_write_image(self, *args, **kwargs):
        raise RuntimeError("Kaleido requires Google Chrome to be installed.")

    monkeypatch.setattr(type(figure), "write_image", fake_write_image)

    with pytest.raises(ImportError, match="plotly_get_chrome"):
        plotting.save_figure(figure, tmp_path / "out.png")


def test_save_figure_reraises_an_unrelated_error(tmp_path, monkeypatch) -> None:
    figure = plotting.plot_convergence([
        get_scored_record(
            "run_0000", j=1.0, vector_nrmse={"pressure": 0.5, "watercut": 0.3, "gor": 0.2}
        )
    ])

    def fake_write_image(self, *args, **kwargs):
        raise ValueError("some unrelated failure")

    monkeypatch.setattr(type(figure), "write_image", fake_write_image)

    with pytest.raises(ValueError, match="unrelated failure"):
        plotting.save_figure(figure, tmp_path / "out.png")

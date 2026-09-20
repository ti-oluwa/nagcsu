"""Tests for `nagcsu.history`."""

import pandas
import pytest

from nagcsu import history


def test_guess_column_map_matches_documented_layout() -> None:
    columns = ["DATE", "FPR", "WWCT_AFIESERE", "WGOR_AFIESERE", "WWCT_ERIEMU", "WGOR_ERIEMU"]
    mapping = history.guess_column_map(columns)
    assert mapping["FPR"] == "FPR"
    assert mapping["WWCT:AFIESERE"] == "WWCT_AFIESERE"
    assert mapping["WGOR:ERIEMU"] == "WGOR_ERIEMU"


def test_guess_column_map_accepts_colon_and_dash_separators() -> None:
    columns = ["WWCT:AFIESERE", "WGOR-AFIESERE"]
    mapping = history.guess_column_map(columns)
    assert mapping["WWCT:AFIESERE"] == "WWCT:AFIESERE"
    assert mapping["WGOR:AFIESERE"] == "WGOR-AFIESERE"


def test_load_observed_history_averages_per_well_columns(tmp_path) -> None:
    frame = pandas.DataFrame({
        "DATE": pandas.date_range("2020-01-01", periods=2, freq="YS"),
        "FPR": [2700, 2650],
        "WWCT_A": [0.10, 0.20],
        "WGOR_A": [800, 810],
        "WWCT_B": [0.20, 0.40],
        "WGOR_B": [820, 830],
    })
    path = tmp_path / "history.xlsx"
    frame.to_excel(path, index=False)

    result = history.load_observed_history(path, wells=["A", "B"])

    assert list(result.columns) == ["DATE", "FPR", "FWCT", "FGOR"]
    assert result["FWCT"].tolist() == pytest.approx([0.15, 0.30])
    assert result["FGOR"].tolist() == pytest.approx([810, 820])


def test_load_observed_history_raises_on_missing_pressure_column(tmp_path) -> None:
    frame = pandas.DataFrame({"DATE": pandas.date_range("2020-01-01", periods=1), "WWCT_A": [0.1]})
    path = tmp_path / "history.xlsx"
    frame.to_excel(path, index=False)

    with pytest.raises(KeyError):
        history.load_observed_history(path, wells=["A"])


def test_load_observed_history_respects_explicit_column_map(tmp_path) -> None:
    frame = pandas.DataFrame({
        "DATE": pandas.date_range("2020-01-01", periods=1),
        "Field Pressure (psia)": [2700],
        "Water Cut": [0.12],
        "GOR": [815],
    })
    path = tmp_path / "history.xlsx"
    frame.to_excel(path, index=False)

    result = history.load_observed_history(
        path,
        column_map={"FPR": "Field Pressure (psia)", "FWCT": "Water Cut", "FGOR": "GOR"},
    )

    assert result["FPR"].iloc[0] == 2700
    assert result["FWCT"].iloc[0] == pytest.approx(0.12)
    assert result["FGOR"].iloc[0] == pytest.approx(815)


def test_load_observed_history_normalizes_string_dates_to_datetime(tmp_path) -> None:
    # A workbook whose DATE column round-trips through Excel as plain
    # strings (rather than a native datetime) must still normalize to
    # midnight Timestamps, since objective.score merges on an exact
    # date match against summary.load_summary's own normalized DATE.
    frame = pandas.DataFrame({
        "DATE": ["2020-01-01", "2021-01-01"],
        "FPR": [2700, 2650],
        "WWCT_A": [0.10, 0.20],
        "WGOR_A": [800, 810],
    })
    path = tmp_path / "history.xlsx"
    frame.to_excel(path, index=False)

    result = history.load_observed_history(path, wells=["A"])

    assert pandas.api.types.is_datetime64_any_dtype(result["DATE"])
    assert result["DATE"].iloc[0] == pandas.Timestamp("2020-01-01")
    # Midnight-normalized: no leftover time-of-day component.
    assert (result["DATE"].dt.time == pandas.Timestamp("2020-01-01").time()).all()


def test_load_observed_history_merges_cleanly_with_a_normalized_simulated_frame(tmp_path) -> None:
    # Regression check for the actual failure mode: an un-normalized
    # observed DATE column merging to nothing against a normalized
    # simulated frame, which objective.score would previously surface
    # only indirectly as a HistoryAlignmentError with no overlapping dates.
    frame = pandas.DataFrame({
        "DATE": pandas.to_datetime(["2020-01-01 00:00:01"]),  # one second past midnight
        "FPR": [2700],
        "WWCT_A": [0.1],
        "WGOR_A": [800],
    })
    path = tmp_path / "history.xlsx"
    frame.to_excel(path, index=False)

    observed = history.load_observed_history(path, wells=["A"])
    simulated = pandas.DataFrame({
        "DATE": pandas.to_datetime(["2020-01-01"]).normalize(),
        "FPR": [2705],
        "FWCT": [0.1],
        "FGOR": [800],
    })

    merged = simulated.merge(observed, on="DATE", suffixes=("_sim", "_obs"))
    assert len(merged) == 1

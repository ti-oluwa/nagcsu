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


def _long_format_frame() -> pandas.DataFrame:
    # Mirrors the real monthly per-well export: one row per well per
    # month, headers like "Field", "Oil_Rate_STB", "Reservoir_Pressure".
    rows = []
    for well, base_oil, base_gas, _base_water, base_pressure in [
        ("AFIESERE", 55.0, 45010.0, 0.0, 2751.0),
        ("ERIEMU", 25.0, 20450.0, 0.0, 2748.0),
    ]:
        for month in range(1, 3):
            rows.append({
                "Field": well,
                "Date": f"1976-{month:02d}-01",
                "Oil_Rate_STB": base_oil - month,
                "Gas_Rate_Mscf": base_gas - month * 10,
                "Water_Rate_STB": month * 0.5,
                "Cum_Oil_STB": base_oil * month,
                "Cum_Gas_Mscf": base_gas * month,
                "Cum_Water_STB": month * 0.5 * month,
                "Reservoir_Pressure": base_pressure - month * 2,
                "Solution_GOR": (base_gas - month * 10) / (base_oil - month),
                "Water_Cut_pct": (month * 0.5) / (base_oil - month + month * 0.5) * 100,
            })
    return pandas.DataFrame(rows)


def test_detect_long_format_columns_recognizes_a_field_well_column() -> None:
    columns = list(_long_format_frame().columns)
    mapping = history.detect_long_format_columns(columns)
    assert mapping is not None
    assert mapping["well"] == "Field"
    assert mapping["oil_rate"] == "Oil_Rate_STB"
    assert mapping["gas_rate"] == "Gas_Rate_Mscf"
    assert mapping["water_rate"] == "Water_Rate_STB"
    assert mapping["pressure"] == "Reservoir_Pressure"


def test_detect_long_format_columns_returns_none_for_a_wide_table() -> None:
    columns = ["DATE", "FPR", "WWCT_AFIESERE", "WGOR_AFIESERE"]
    assert history.detect_long_format_columns(columns) is None


def test_gor_pattern_matches_an_underscore_separated_header() -> None:
    # Regression check: \bgor\b alone does not match "Solution_GOR",
    # since "_" counts as a word character in regex and so is not a
    # word boundary between "_" and "G".
    assert history.GOR_PATTERN.search("Solution_GOR")
    assert history.GOR_PATTERN.search("GOR (scf/stb)")
    assert not history.GOR_PATTERN.search("Category")


def test_load_observed_history_aggregates_long_format_from_summed_rates(tmp_path) -> None:
    path = tmp_path / "history.csv"
    _long_format_frame().to_csv(path, index=False)

    result = history.load_observed_history(path, date_column="Date")

    assert list(result.columns) == ["DATE", "FPR", "FWCT", "FGOR"]
    assert len(result) == 2
    january = result.iloc[0]
    assert january["FPR"] == pytest.approx((2749.0 + 2746.0) / 2)
    assert january["FWCT"] > 0
    assert january["FGOR"] > 0


def test_load_observed_history_reads_csv_and_excel_identically(tmp_path) -> None:
    frame = _long_format_frame()
    csv_path = tmp_path / "history.csv"
    xlsx_path = tmp_path / "history.xlsx"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False)

    from_csv = history.load_observed_history(csv_path, date_column="Date")
    from_xlsx = history.load_observed_history(xlsx_path, date_column="Date")

    pandas.testing.assert_frame_equal(from_csv, from_xlsx)


def test_load_observed_history_falls_back_to_precomputed_ratio_columns(tmp_path) -> None:
    # No rate columns at all, only a percent water-cut and a GOR column;
    # the percent column must be detected and scaled to a 0-1 fraction.
    frame = pandas.DataFrame([
        {
            "Field": "A",
            "Date": "1976-01-01",
            "Reservoir_Pressure": 2750,
            "Water_Cut_pct": 5.0,
            "Solution_GOR": 800,
        },
        {
            "Field": "B",
            "Date": "1976-01-01",
            "Reservoir_Pressure": 2740,
            "Water_Cut_pct": 15.0,
            "Solution_GOR": 820,
        },
    ])
    path = tmp_path / "history.csv"
    frame.to_csv(path, index=False)

    result = history.load_observed_history(path, date_column="Date")

    assert result["FWCT"].iloc[0] == pytest.approx(0.10)
    assert result["FGOR"].iloc[0] == pytest.approx(810.0)


def test_load_observed_history_raises_when_no_pressure_column_exists(tmp_path) -> None:
    frame = pandas.DataFrame([
        {"Field": "A", "Date": "1976-01-01", "Oil_Rate_STB": 50.0, "Water_Rate_STB": 1.0}
    ])
    path = tmp_path / "history.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(KeyError):
        history.load_observed_history(path, date_column="Date")


def test_detect_file_format_uses_extension() -> None:
    import pathlib

    assert history.detect_file_format(pathlib.Path("x.csv")) == "csv"
    assert history.detect_file_format(pathlib.Path("x.xlsx")) == "excel"
    with pytest.raises(ValueError):
        history.detect_file_format(pathlib.Path("x.parquet"))


def test_detect_file_format_respects_explicit_override() -> None:
    import pathlib

    assert history.detect_file_format(pathlib.Path("x.txt"), "csv") == "csv"


def test_load_observed_history_always_outputs_a_date_column_named_date(tmp_path) -> None:
    # Regression check: the source file's own date column name (here
    # "Date", not "DATE") must never leak into the returned frame, since
    # objective.score merges the observed frame against
    # summary.load_summary's output, which always names its date column
    # "DATE" (uppercase) no matter what the source called it.
    frame = _long_format_frame()
    path = tmp_path / "history.csv"
    frame.to_csv(path, index=False)

    result = history.load_observed_history(path, date_column="Date")

    assert "DATE" in result.columns
    assert "Date" not in result.columns

    simulated = pandas.DataFrame({
        "DATE": result["DATE"],
        "FPR": result["FPR"] + 5,
        "FWCT": result["FWCT"],
        "FGOR": result["FGOR"],
    })
    from nagcsu import objective

    score = objective.score(
        simulated, result, weights={"pressure": 1.0, "watercut": 0.0, "gor": 0.0}
    )
    assert score.j > 0


def test_detect_date_column_matches_common_variants() -> None:
    assert history.detect_date_column(["Field", "Date", "Oil_Rate_STB"]) == "Date"
    assert history.detect_date_column(["Well", "Report_Date", "GOR"]) == "Report_Date"
    assert history.detect_date_column(["Well", "Production Date", "GOR"]) == "Production Date"
    assert history.detect_date_column(["Well", "Oil_Rate_STB"]) is None


def test_peek_columns_reads_headers_without_loading_data(tmp_path) -> None:
    frame = _long_format_frame()
    path = tmp_path / "history.csv"
    frame.to_csv(path, index=False)

    columns = history.peek_columns(path)

    assert columns == list(frame.columns)


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

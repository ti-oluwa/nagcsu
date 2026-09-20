"""Tests for `nagcsu.objective`."""

import pandas
import pytest

from nagcsu import objective
from nagcsu.exceptions import HistoryAlignmentError


def _frame(pressure, watercut, gor):
    return pandas.DataFrame(
        {
            "DATE": pandas.date_range("2020-01-01", periods=len(pressure), freq="YS"),
            "FPR": pressure,
            "FWCT": watercut,
            "FGOR": gor,
        }
    )


def test_nrmse_zero_for_identical_series() -> None:
    series = [2700, 2650, 2600, 2550]
    assert objective.nrmse(series, series) == 0.0


def test_nrmse_scales_with_offset_over_range() -> None:
    observed = [2500, 2600, 2700, 2800]  # range = 300
    simulated = [2510, 2610, 2710, 2810]  # constant +10 offset
    assert objective.nrmse(simulated, observed) == pytest.approx(10 / 300, abs=1e-9)


def test_nrmse_falls_back_to_raw_rmse_for_zero_range_observed() -> None:
    observed = [2700, 2700, 2700]
    simulated = [2705, 2705, 2705]
    assert objective.nrmse(simulated, observed) == pytest.approx(5.0)


def test_score_combines_weighted_nrmse() -> None:
    simulated = _frame([2710, 2660, 2610, 2560], [0.10, 0.15, 0.20, 0.25], [820, 830, 840, 850])
    observed = _frame([2700, 2650, 2600, 2550], [0.10, 0.15, 0.20, 0.25], [820, 830, 840, 850])

    result = objective.score(simulated, observed, weights={"pressure": 0.5, "watercut": 0.35, "gor": 0.15})

    assert result.vector_scores["pressure"].nrmse > 0
    assert result.vector_scores["watercut"].nrmse == pytest.approx(0.0)
    assert result.vector_scores["gor"].nrmse == pytest.approx(0.0)
    expected_j = 0.5 * result.vector_scores["pressure"].nrmse
    assert result.j == pytest.approx(expected_j)


def test_score_raises_on_no_overlapping_dates() -> None:
    simulated = _frame([2700], [0.1], [820])
    observed = _frame([2700], [0.1], [820])
    observed["DATE"] = pandas.to_datetime(["2099-01-01"])

    with pytest.raises(HistoryAlignmentError):
        objective.score(simulated, observed, weights={"pressure": 1.0, "watercut": 0.0, "gor": 0.0})


def test_score_raises_on_missing_column() -> None:
    simulated = _frame([2700, 2650], [0.1, 0.15], [820, 830]).drop(columns=["FGOR"])
    observed = _frame([2700, 2650], [0.1, 0.15], [820, 830])

    with pytest.raises(HistoryAlignmentError):
        objective.score(simulated, observed, weights={"pressure": 0.5, "watercut": 0.35, "gor": 0.15})

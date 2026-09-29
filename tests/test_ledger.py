"""Tests for `nagcsu.ledger`."""

import dataclasses

import pytest

from nagcsu import ledger


def get_run_record(run_id: str, j: float | None) -> ledger.RunRecord:
    return ledger.RunRecord(
        run_id=run_id,
        created_at=ledger.timestamp_now(),
        parameter_state={"aquifer.radius": 16137.2},
        group="aquifer",
        strategy="grid",
        j=j,
        vector_nrmse={"pressure": 0.1} if j is not None else None,
        prt_is_clean=True,
        note="test",
    )


def test_load_returns_empty_list_for_missing_file(tmp_path) -> None:
    assert ledger.load(tmp_path / "does_not_exist.json") == []


def test_append_and_load_round_trip(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    ledger.append(path, get_run_record("run_0000", j=0.30))
    ledger.append(path, get_run_record("run_0001", j=0.20))

    records = ledger.load(path)
    assert [record.run_id for record in records] == ["run_0000", "run_0001"]
    assert records[1].j == pytest.approx(0.20)


def test_new_run_id_increments_from_existing_count(tmp_path) -> None:
    assert ledger.new_run_id([]) == "run_0000"
    assert ledger.new_run_id([get_run_record("run_0000", j=0.1)]) == "run_0001"


def test_best_record_ignores_unscored_runs(tmp_path) -> None:
    records = [
        get_run_record("run_0000", j=None),
        get_run_record("run_0001", j=0.25),
        get_run_record("run_0002", j=0.10),
    ]
    best = ledger.get_best_record(records)
    assert best is not None
    assert best.run_id == "run_0002"


def test_best_record_returns_none_when_nothing_is_scored() -> None:
    assert ledger.get_best_record([get_run_record("run_0000", j=None)]) is None


def test_filter_records_by_strategy() -> None:
    records = [
        get_run_record("run_0000", j=0.5),
        dataclasses.replace(get_run_record("run_0001", j=0.4), strategy="random"),
    ]
    filtered = ledger.filter_records(records, strategy="grid")
    assert [record.run_id for record in filtered] == ["run_0000"]


def test_filter_records_by_group() -> None:
    records = [
        get_run_record("run_0000", j=0.5),
        dataclasses.replace(get_run_record("run_0001", j=0.4), group="sgof_shape"),
    ]
    filtered = ledger.filter_records(records, group="sgof_shape")
    assert [record.run_id for record in filtered] == ["run_0001"]


def test_filter_records_limit_keeps_the_most_recent() -> None:
    records = [get_run_record(f"run_{i:04d}", j=float(i)) for i in range(5)]
    filtered = ledger.filter_records(records, limit=2)
    assert [record.run_id for record in filtered] == ["run_0003", "run_0004"]


def test_filter_records_with_no_filters_returns_everything() -> None:
    records = [get_run_record("run_0000", j=0.5), get_run_record("run_0001", j=0.4)]
    assert ledger.filter_records(records) == records

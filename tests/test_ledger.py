"""Tests for `nagcsu.ledger`."""

from nagcsu import ledger


def _record(run_id: str, j: float | None) -> ledger.RunRecord:
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
    ledger.append(path, _record("run_0000", j=0.30))
    ledger.append(path, _record("run_0001", j=0.20))

    records = ledger.load(path)
    assert [record.run_id for record in records] == ["run_0000", "run_0001"]
    assert records[1].j == 0.20


def test_new_run_id_increments_from_existing_count(tmp_path) -> None:
    assert ledger.new_run_id([]) == "run_0000"
    assert ledger.new_run_id([_record("run_0000", j=0.1)]) == "run_0001"


def test_best_record_ignores_unscored_runs(tmp_path) -> None:
    records = [_record("run_0000", j=None), _record("run_0001", j=0.25), _record("run_0002", j=0.10)]
    best = ledger.best_record(records)
    assert best is not None
    assert best.run_id == "run_0002"


def test_best_record_returns_none_when_nothing_is_scored() -> None:
    assert ledger.best_record([_record("run_0000", j=None)]) is None

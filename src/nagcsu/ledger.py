"""A JSON-backed record of every run a project has made.

Every `nagcsu run` and `nagcsu match` invocation appends one
`RunRecord` here. It is what makes `nagcsu report` and
`nagcsu match auto`'s stopping logic possible without re-reading every
run directory's `.PRT` and summary output on every invocation, and it is
the "how it was gotten" record a snapshot report is built from.
"""

import dataclasses
import datetime
import json
import pathlib
import typing

LEDGER_SCHEMA_VERSION: typing.Final[int] = 1
"""
Bumped whenever `RunRecord`'s shape changes in a way that is not
backward compatible with an older ledger file on disk.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class RunRecord:
    """One logged simulation run and, if scored, its result."""

    run_id: str
    """Identifier for this run, also the name of its output subdirectory."""

    created_at: str
    """ISO 8601 timestamp of when this record was appended."""

    parameter_state: dict[str, float]
    """Full resolved parameter state used to build this run's deck (see
    `nagcsu.parameters.resolve_state`), so the deck is
    reproducible from this record alone.
    """

    group: str | None
    """Tuning priority group this run belongs to, if it came from
    `nagcsu match auto` or `nagcsu match sweep`. `None` for an ad hoc run.
    """

    strategy: str | None
    """Name of the search strategy that produced this run (for example
    "grid", "random", "coordinate_descent"), if any.
    """

    j: float | None
    """Combined objective score, or `None` if this run was not scored
    (for example, a baseline sanity-check run).
    """

    vector_nrmse: dict[str, float] | None
    """Per-vector NRMSE from the same scoring pass as `j`, or `None`."""

    prt_is_clean: bool | None
    """`nagcsu.prt.PrtReport.is_clean` for this run, or `None` if the
    `.PRT` was not parsed.
    """

    note: str
    """Free-text note, for example why a run was made or what changed
    since the previous one in its group.
    """

    simulation_error: str | None = None
    """Message from `nagcsu.simulate.run` if the simulation failed to
    produce any summary output, or `None` otherwise. Defaulted so a
    ledger file written before this field existed still loads.
    """


def load(ledger_path: pathlib.Path | str) -> list[RunRecord]:
    """Load every record from a ledger file.

    :returns: An empty list if `ledger_path` does not exist yet, since a
        project's first run has no ledger to load.
    """
    ledger_path = pathlib.Path(ledger_path)
    if not ledger_path.exists():
        return []
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    return [RunRecord(**record) for record in payload.get("runs", [])]


def append(ledger_path: pathlib.Path | str, record: RunRecord) -> None:
    """Append `record` to the ledger at `ledger_path`, creating it if needed."""
    ledger_path = pathlib.Path(ledger_path)
    records = load(ledger_path)
    records.append(record)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "runs": [dataclasses.asdict(saved_record) for saved_record in records],
    }
    ledger_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def new_run_id(existing: list[RunRecord]) -> str:
    """Return the next `run_NNNN` identifier given a project's existing records."""
    return f"run_{len(existing):04d}"


def best_record(records: list[RunRecord]) -> RunRecord | None:
    """Return the scored record with the lowest `j`, or `None` if none is scored."""
    scored = [record for record in records if record.j is not None]
    if not scored:
        return None
    return min(scored, key=lambda record: record.j or 0)


def timestamp_now() -> str:
    """Return the current UTC time as an ISO 8601 string, for `RunRecord.created_at`."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

"""A JSON-backed record of every run a project has made."""

import dataclasses
import datetime
import json
import pathlib
import re
import typing

LEDGER_SCHEMA_VERSION: typing.Final[int] = 2
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

    tuned_parameters: list[str] | None = None
    """Parameters this run deliberately changed relative to the state it
    was derived from (a sweep's swept parameter, one coordinate-descent
    probe's axis, a sensitivity probe's perturbed parameter). `None` when
    unknown, for example a run made before schema version 2.
    """

    tuned_values: dict[str, float] | None = None
    """Value each entry of `tuned_parameters` had in this run."""

    stage: str | None = None
    """Step within the strategy that produced this run, for example
    "baseline", "descent/pass2", "sensitivity/high" or "final".
    """

    objective_weights: dict[str, float] | None = None
    """Vector weights `j` was computed with, so a report can show what
    each vector contributed even after `nagcsu.yaml` or a `--weights`
    override has since changed.
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


def filter_records(
    records: list[RunRecord],
    *,
    strategy: str | None = None,
    group: str | None = None,
    run_id_prefix: str | None = None,
    run_id_regex: str | None = None,
    parameter: str | None = None,
    stage: str | None = None,
    limit: int | None = None,
) -> list[RunRecord]:
    """Narrow a list of records down to one strategy, group, or run ID pattern.

    Order is preserved, so the result is still chronological. Useful
    before plotting, where mixing every strategy onto one trial axis
    (an `auto` session alongside an unrelated `sweep`) would make the
    x-axis meaningless.

    :param strategy: Keep only records with this exact `strategy`.
    :param group: Keep only records with this exact `group`.
    :param run_id_prefix: Keep only records whose `run_id` begins with
        this prefix.
    :param run_id_regex: Keep only records whose `run_id` matches this
        regular expression.
    :param parameter: Keep only records that changed this parameter.
    :param stage: Keep only records whose `stage` starts with this text,
        so "descent" matches "descent/pass1" and "descent/pass2".
    :param limit: Keep only the most recent `limit` records, applied
        after the other filters.
    """
    pattern: re.Pattern[str] | None = None
    if run_id_regex is not None:
        pattern = re.compile(run_id_regex)

    filtered = [
        record
        for record in records
        if (strategy is None or record.strategy == strategy)
        and (group is None or record.group == group)
        and (run_id_prefix is None or record.run_id.startswith(run_id_prefix))
        and (pattern is None or pattern.fullmatch(record.run_id) is not None)
        and (parameter is None or parameter in (record.tuned_parameters or []))
        and (stage is None or (record.stage or "").startswith(stage))
    ]
    return filtered[-limit:] if limit else filtered


def get_best_record(records: list[RunRecord]) -> RunRecord | None:
    """Return the scored record with the lowest `j`, or `None` if none is scored."""
    scored = [record for record in records if record.j is not None]
    if not scored:
        return None
    return min(scored, key=lambda record: record.j or 0)


def timestamp_now() -> str:
    """Return the current UTC time as an ISO 8601 string, for `RunRecord.created_at`."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterHistory:
    """Everything the ledger says about trials that moved one parameter alone."""

    parameter: str
    """Parameter name."""

    group: str | None
    """Tuning group recorded on those trials."""

    trials: int
    """Trials that moved only this parameter, scored or not."""

    failed: int
    """How many of `trials` failed to simulate or were not scored."""

    best_j: float | None
    """Lowest J reached by any of those trials."""

    best_value: float | None
    """Parameter value at `best_j`."""

    worst_j: float | None
    """Highest J reached by any of those trials."""

    j_span: float | None
    """`worst_j - best_j`: how much J moved across everything tried for
    this parameter. Near zero means the parameter was explored and did
    not matter, which is the signal to stop touching it.
    """

    tried_low: float | None
    """Lowest value tried."""

    tried_high: float | None
    """Highest value tried."""

    last_run_id: str
    """Most recent run that moved this parameter."""


def summarize_parameters(records: list[RunRecord]) -> list[ParameterHistory]:
    """Aggregate single-parameter trials by parameter, best result first.

    Only records whose `tuned_parameters` names exactly one parameter are
    counted, because a run that moved several parameters at once (a random
    search) cannot attribute its J to any one of them. Unscored parameters
    sort last.
    """
    grouped: dict[str, list[RunRecord]] = {}
    for record in records:
        if record.tuned_parameters and len(record.tuned_parameters) == 1:
            grouped.setdefault(record.tuned_parameters[0], []).append(record)

    histories: list[ParameterHistory] = []
    for parameter, parameter_records in grouped.items():
        scored = [record for record in parameter_records if record.j is not None]
        values = [
            (record.tuned_values or {}).get(parameter)
            for record in parameter_records
            if (record.tuned_values or {}).get(parameter) is not None
        ]
        values = typing.cast(list[float], values)
        best = min(scored, key=lambda record: record.j or 0.0) if scored else None
        worst_j = max((record.j for record in scored if record.j is not None), default=None)
        histories.append(
            ParameterHistory(
                parameter=parameter,
                group=next((r.group for r in reversed(parameter_records) if r.group), None),
                trials=len(parameter_records),
                failed=len(parameter_records) - len(scored),
                best_j=best.j if best else None,
                best_value=(best.tuned_values or {}).get(parameter) if best else None,
                worst_j=worst_j,
                j_span=(worst_j - best.j)
                if best and best.j is not None and worst_j is not None
                else None,
                tried_low=min(values) if values else None,
                tried_high=max(values) if values else None,
                last_run_id=parameter_records[-1].run_id,
            )
        )
    histories.sort(key=lambda history: (history.best_j is None, history.best_j or 0.0))
    return histories


@dataclasses.dataclass(frozen=True, slots=True)
class GroupHistory:
    """Everything the ledger says about trials attributed to one group."""

    group: str
    """Tuning group name."""

    trials: int
    """Trials logged under this group, whatever parameter they moved."""

    parameters_touched: list[str]
    """Distinct parameters those trials moved, in first-touched order."""

    best_j: float | None
    """Lowest J among the group's scored trials."""

    best_run_id: str | None
    """Run that reached `best_j`."""


def summarize_groups(records: list[RunRecord]) -> list[GroupHistory]:
    """Aggregate trials by their recorded group, best result first."""
    grouped: dict[str, list[RunRecord]] = {}
    for record in records:
        if record.group:
            grouped.setdefault(record.group, []).append(record)

    histories: list[GroupHistory] = []
    for group, group_records in grouped.items():
        touched: list[str] = []
        for record in group_records:
            for parameter in record.tuned_parameters or []:
                if parameter not in touched:
                    touched.append(parameter)
        best = get_best_record(group_records)
        histories.append(
            GroupHistory(
                group=group,
                trials=len(group_records),
                parameters_touched=touched,
                best_j=best.j if best else None,
                best_run_id=best.run_id if best else None,
            )
        )
    histories.sort(key=lambda history: (history.best_j is None, history.best_j or 0.0))
    return histories

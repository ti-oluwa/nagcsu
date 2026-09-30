"""Writing a human-readable summary of a run or a tuning session.

The row-building helpers here are shared by the Markdown renderer below
and by the terminal tables in `nagcsu.cli.display`, so a report file and
what `nagcsu report show` prints always agree.
"""

import dataclasses
import pathlib
import typing

from nagcsu import constants, ledger, parameters
from nagcsu.algorithms.coordinate_descent import GroupOutcome
from nagcsu.algorithms.sensitivity import GroupSensitivity, SensitivityResult
from nagcsu.prt import PrtReport

PINNED_AT_BOUND_FRACTION: typing.Final[float] = 0.02
"""A tuned value within this fraction of a bound range from either edge is
flagged as pinned: the optimum may lie outside the searched range."""

FLAT_PARAMETER_J_SPAN: typing.Final[float] = 0.005
"""A parameter whose tried values moved J by less than this is called flat."""


@dataclasses.dataclass(frozen=True, slots=True)
class StateRow:
    """One parameter's final value set against its default and bounds."""

    name: str
    group: str
    value: float
    default: float
    change_percent: float | None
    """`(value - default) / abs(default) * 100`, `None` for a zero default."""

    bounds: tuple[float, float]
    changed: bool
    pinned: str | None
    """"low" or "high" when the value sits at a bound edge, else `None`."""


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectiveRow:
    """One scored vector's share of J."""

    name: str
    nrmse: float
    weight: float | None
    contribution: float | None
    """`weight * nrmse`, `None` if the weight was not recorded."""

    share: float | None
    """`contribution / sum of contributions`."""


@dataclasses.dataclass(frozen=True, slots=True)
class WellRow:
    """One well's water-cut and GOR mismatch."""

    well: str
    watercut_nrmse: float | None
    gor_nrmse: float | None
    in_objective: bool
    """Whether this well fed J through `wells_watercut` / `wells_gor`."""


def get_well_rows(record: ledger.RunRecord) -> list[WellRow]:
    """Per-well NRMSE rows for a record, worst water-cut mismatch first."""
    if not record.well_nrmse:
        return []
    names = sorted({key.split(":", 1)[1] for key in record.well_nrmse})
    selected = set(record.scored_wells or [])
    rows = [
        WellRow(
            well=name,
            watercut_nrmse=record.well_nrmse.get(f"WWCT:{name}"),
            gor_nrmse=record.well_nrmse.get(f"WGOR:{name}"),
            in_objective=name in selected,
        )
        for name in names
    ]
    rows.sort(key=lambda row: -(row.watercut_nrmse or 0.0))
    return rows


def get_state_rows(state: dict[str, float]) -> list[StateRow]:
    """Describe every registered parameter in `state`, changed ones first."""
    rows: list[StateRow] = []
    for spec in parameters.PARAMETERS.values():
        if spec.name not in state:
            continue
        value = state[spec.name]
        low, high = spec.bounds
        margin = (high - low) * PINNED_AT_BOUND_FRACTION
        pinned = "low" if value <= low + margin else ("high" if value >= high - margin else None)
        changed = abs(value - spec.default) > 1e-12 * max(1.0, abs(spec.default))
        rows.append(
            StateRow(
                name=spec.name,
                group=spec.group,
                value=value,
                default=spec.default,
                change_percent=(value - spec.default) / abs(spec.default) * 100.0
                if spec.default
                else None,
                bounds=spec.bounds,
                changed=changed,
                pinned=pinned if changed else None,
            )
        )
    group_order = {
        group: index for index, group in enumerate(constants.GROUP_TUNING_PRIORITY_ORDER)
    }
    rows.sort(key=lambda row: (not row.changed, group_order.get(row.group, 99), row.name))
    return rows


def get_objective_rows(record: ledger.RunRecord) -> list[ObjectiveRow]:
    """Break a scored record's J into per-vector NRMSE, weight and share."""
    if not record.vector_nrmse:
        return []
    weights = record.objective_weights or {}
    contributions = {
        name: (weights[name] * value if name in weights else None)
        for name, value in record.vector_nrmse.items()
    }
    contributions = typing.cast(dict[str, float], contributions)
    total = sum(value for value in contributions.values() if value is not None)
    return [
        ObjectiveRow(
            name=name,
            nrmse=value,
            weight=weights.get(name),
            contribution=contributions[name],
            share=(contributions[name] / total)
            if contributions[name] is not None and total > 0
            else None,
        )
        for name, value in record.vector_nrmse.items()
    ]


def get_baseline_record(records: list[ledger.RunRecord]) -> ledger.RunRecord | None:
    """Return the reference run improvement is measured from.

    The earliest scored run with no tuned parameters (a plain `nagcsu run`
    or a search's baseline), falling back to the earliest scored run.
    """
    scored = [record for record in records if record.j is not None]
    untouched = [record for record in scored if not record.tuned_parameters]
    return (untouched or scored or [None])[0]


def next_steps(
    records: list[ledger.RunRecord],
    *,
    target_j: float | None = None,
    limit: int = 6,
) -> list[str]:
    """Suggest what to touch next, from what the ledger already shows."""
    steps: list[str] = []
    best = ledger.get_best_record(records)
    if best is not None and target_j is not None:
        if best.j is not None and best.j <= target_j:
            steps.append(
                f"Best J {best.j:.4f} is at or below the target {target_j:.4f}: stop tuning, "
                f"freeze `{best.run_id}` and document it."
            )

    touched_groups = {history.group for history in ledger.summarize_groups(records)}
    untouched = [
        group for group in constants.GROUP_TUNING_PRIORITY_ORDER if group not in touched_groups
    ]
    if untouched:
        steps.append(f"Never touched so far: {', '.join(untouched)}.")

    histories = [h for h in ledger.summarize_parameters(records) if h.j_span is not None]
    pinned_bounds = get_parameters_pinned_at_bound(best)
    for name, side in pinned_bounds:
        steps.append(
            f"`{name}` in the best run sits at its {side} bound: widen its range with "
            f"`match auto --range {name}=LOW:HIGH` (edit `parameters.py` bounds if needed)."
        )

    flat = [h for h in histories if h.trials >= 3 and (h.j_span or 0.0) < FLAT_PARAMETER_J_SPAN]
    if flat:
        names = ", ".join(h.parameter for h in flat[:5])
        steps.append(f"Explored with no measurable effect on J, leave alone: {names}.")

    live = sorted(
        (h for h in histories if (h.j_span or 0.0) >= FLAT_PARAMETER_J_SPAN),
        key=lambda h: -(h.j_span or 0.0),
    )
    if live:
        top = live[0]
        steps.append(
            f"`{top.parameter}` moved J by {top.j_span:.4f} across {top.trials} trials "
            f"(best at {top.best_value:.6g}): the most responsive parameter, "
            f"refine around it with a narrower `--range`."
        )
    return steps[:limit]


def get_parameters_pinned_at_bound(record: ledger.RunRecord | None) -> list[tuple[str, str]]:
    if record is None:
        return []
    return [(row.name, row.pinned) for row in get_state_rows(record.parameter_state) if row.pinned]


def render_run_report(
    record: ledger.RunRecord,
    *,
    prt_report: PrtReport | None = None,
    group_outcomes: list[GroupOutcome] | None = None,
    sensitivity_results: list[SensitivityResult] | None = None,
    group_sensitivities: list[GroupSensitivity] | None = None,
    records: list[ledger.RunRecord] | None = None,
    target_j: float | None = None,
    show_wells: bool = True,
) -> str:
    """Render a single run's snapshot as a Markdown document.

    :param record: The run to report on.
    :param prt_report: That run's parsed `.PRT` health, if available.
    :param group_outcomes: `nagcsu.algorithms.coordinate_descent.GroupOutcome`
        entries, if `record` came from an auto-tune session, in the
        order the groups were tuned.
    :param sensitivity_results: Ranked parameter sensitivity results, if a
        sensitivity pass was run alongside this record.
    :param group_sensitivities: Group ranking built from those results.
    :param records: The whole ledger (or a slice of it). Enables the
        improvement-over-baseline line, per-parameter history and
        data-driven next steps.
    :param target_j: Objective target, used in the next-steps advice.
    :param show_wells: Include the per-well NRMSE section when the record has one.
    """
    lines: list[str] = []
    lines.append(f"# History match snapshot: {record.run_id}")
    lines.append("")
    lines.append(f"Generated from run `{record.run_id}`, logged {record.created_at}.")
    if record.note:
        lines.append("")
        lines.append(record.note)

    lines.append("")
    lines.append("## Run identity")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Strategy | {record.strategy or '-'} |")
    lines.append(f"| Group | {record.group or '-'} |")
    lines.append(f"| Parameters moved | {', '.join(record.tuned_parameters or []) or '-'} |")
    if record.tuned_values:
        moved = ", ".join(f"{name}={value:.6g}" for name, value in record.tuned_values.items())
        lines.append(f"| Values | {moved} |")
    lines.append(f"| Stage | {record.stage or '-'} |")
    lines.append(f"| PRT clean | {_yes_or_no(record.prt_is_clean)} |")

    lines.append("")
    if record.simulation_error:
        lines.append("## Simulation failed")
        lines.append("")
        lines.append(
            f"This run's simulation did not produce usable output: {record.simulation_error}"
        )
    else:
        lines.append("## Objective")
        lines.append("")
        if record.j is None:
            lines.append("This run was not scored against the observed history.")
        else:
            lines.append(f"J = **{record.j:.4f}**")
            if target_j is not None:
                verdict = "at or below" if record.j <= target_j else "above"
                lines.append(f" ({verdict} the target {target_j:.4f})")
            baseline = get_baseline_record(records or [])
            if baseline is not None and baseline.j and baseline.run_id != record.run_id:
                drop = (baseline.j - record.j) / baseline.j * 100.0
                lines.append("")
                lines.append(
                    f"Baseline `{baseline.run_id}` scored {baseline.j:.4f}; this run is "
                    f"{abs(drop):.1f}% {'better' if drop >= 0 else 'worse'}."
                )
            lines.append("")
            lines.append("| Vector | NRMSE | Weight | Weighted | Share of J |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in get_objective_rows(record):
                lines.append(
                    f"| {row.name} | {row.nrmse:.4f} | {_fmt(row.weight)} | "
                    f"{_fmt(row.contribution)} | "
                    f"{'-' if row.share is None else f'{row.share * 100:.1f}%'} |"
                )

    wells = get_well_rows(record) if show_wells else []
    if wells:
        lines.append("")
        lines.append("## Per-well match")
        lines.append("")
        scored = ", ".join(record.scored_wells or []) or "none (field totals only)"
        lines.append(f"Wells counted in J: {scored}. Other wells are shown as diagnostics.")
        lines.append("")
        lines.append("| Well | Water cut NRMSE | GOR NRMSE | In J |")
        lines.append("| --- | --- | --- | --- |")
        for row in wells:
            lines.append(
                f"| {row.well} | {_fmt(row.watercut_nrmse)} | {_fmt(row.gor_nrmse)} | "
                f"{'yes' if row.in_objective else 'no'} |"
            )

    if group_outcomes:
        lines.append("")
        lines.append("## How this was found: tuning path")
        lines.append("")
        lines.append(
            "| Group | Starting J | Ending J | Change | Simulations | Passes | Reached target |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for outcome in group_outcomes:
            lines.append(
                f"| {outcome.group} | {outcome.starting_j:.4f} | {outcome.ending_j:.4f} | "
                f"{outcome.ending_j - outcome.starting_j:+.4f} | {outcome.evaluations} | "
                f"{outcome.passes} | {'yes' if outcome.reached_target else 'no'} |"
            )
        steps = [step for outcome in group_outcomes for step in outcome.parameter_outcomes]
        if steps:
            lines.append("")
            lines.append("### Parameter by parameter")
            lines.append("")
            lines.append(
                "| Group | Pass | Parameter | Window | Start | End | J before | J after | Sims |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for step in steps:
                lines.append(
                    f"| {step.group} | {step.pass_index} | {step.parameter} | "
                    f"{step.window[0]:.6g} to {step.window[1]:.6g} | {step.start_value:.6g} | "
                    f"{step.end_value:.6g} | {step.starting_j:.4f} | {step.ending_j:.4f} | "
                    f"{step.evaluations} |"
                )
        lines.append("")
        lines.append(
            "Groups were tuned one at a time; a group not listed here was never touched "
            "because an earlier group already reached the target."
        )

    lines.append("")
    lines.append("## Final parameter state")
    lines.append("")
    lines.append("| Parameter | Group | Value | Default | Change | Bounds | Note |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in get_state_rows(record.parameter_state):
        change = (
            "-" if row.change_percent is None or not row.changed else f"{row.change_percent:+.1f}%"
        )
        note = (
            f"at {row.pinned} bound" if row.pinned else ("changed" if row.changed else "default")
        )
        lines.append(
            f"| {row.name} | {row.group} | {row.value:.6g} | {row.default:.6g} | {change} | "
            f"{row.bounds[0]:g} to {row.bounds[1]:g} | {note} |"
        )

    if records:
        parameter_histories = ledger.summarize_parameters(records)
        if parameter_histories:
            lines.append("")
            lines.append("## What has been tried, per parameter")
            lines.append("")
            lines.append(
                "Single-parameter trials only; runs that moved several parameters at "
                "once cannot be attributed to any one of them."
            )
            lines.append("")
            lines.append(
                "| Parameter | Group | Trials | Best J | Value at best | J span | Range tried |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for history in parameter_histories:
                lines.append(
                    f"| {history.parameter} | {history.group or '-'} | {history.trials} | "
                    f"{_fmt(history.best_j)} | {_fmt(history.best_value)} | {_fmt(history.j_span)} | "
                    f"{_fmt(history.tried_low)} to {_fmt(history.tried_high)} |"
                )

    if prt_report is not None:
        lines.append("")
        lines.append("## Run health")
        lines.append("")
        lines.append(f"- Report steps completed: {len(prt_report.completed_report_steps)}")
        lines.append(f"- Last simulated date: {prt_report.last_simulated_date}")
        lines.append(
            f"- Errors: {prt_report.errors}, Bugs: {prt_report.bugs}, Warnings: {prt_report.warnings}"
        )
        if prt_report.unconverged_well_counts:
            worst = sorted(prt_report.unconverged_well_counts.items(), key=lambda item: -item[1])[
                :3
            ]
            worst_text = ", ".join(f"{well} ({count})" for well, count in worst)
            lines.append(f"- Wells with the most convergence warnings: {worst_text}")
        lines.append(f"- Considered clean: {'yes' if prt_report.is_clean else 'no'}")

    if group_sensitivities:
        lines.append("")
        lines.append("## Group sensitivity ranking")
        lines.append("")
        lines.append(
            "| Position | Group | Parameters | Mean rank | Rank sum | Total swing | Share |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for group in group_sensitivities:
            lines.append(
                f"| {group.position} | {group.group} | {len(group.parameters)} | "
                f"{group.mean_rank:.2f} | {group.rank_sum:.1f} | {group.total_swing:.4f} | "
                f"{group.swing_share * 100:.1f}% |"
            )

    if sensitivity_results:
        lines.append("")
        lines.append("## Parameter sensitivity")
        lines.append("")
        lines.append(
            "Ranked by local sensitivity (how much J moved when this parameter "
            "alone was perturbed, others held fixed):"
        )
        lines.append("")
        lines.append("| Parameter | Swing in J |")
        lines.append("| --- | --- |")
        for result in sensitivity_results[:8]:
            lines.append(f"| {result.parameter} | {result.swing:.4f} |")

    suggestions = next_steps(records or [record], target_j=target_j)
    if suggestions:
        lines.append("")
        lines.append("## What to try next")
        lines.append("")
        for suggestion in suggestions:
            lines.append(f"- {suggestion}")

    lines.append("")
    return "\n".join(lines)


def write_run_report(
    record: ledger.RunRecord, output_path: pathlib.Path | str, **kwargs: typing.Any
) -> pathlib.Path:
    """Render and write a run report; see `render_run_report` for `kwargs`."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_run_report(record, **kwargs), encoding="utf-8")
    return output_path


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def _yes_or_no(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")

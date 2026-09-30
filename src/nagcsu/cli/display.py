"""Rich tables for every `nagcsu` command.

All terminal formatting lives here so commands stay about behavior. Row
data comes from `nagcsu.reporting` and `nagcsu.ledger`, which keeps the
tables identical to the Markdown a report file contains.
"""

import math
import typing

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nagcsu import cleanup, constants, glossary, ledger, parameters, ranges, reporting
from nagcsu.algorithms import coordinate_descent, sensitivity
from nagcsu.prt import PrtReport

KEY_ENABLED = True
"""Whether `print_key` prints anything. Turned off by the global `--no-key` option."""


def set_key_enabled(enabled: bool) -> None:
    """Turn the table key on or off for this process."""
    global KEY_ENABLED
    KEY_ENABLED = enabled


if typing.TYPE_CHECKING:
    from nagcsu.cli import context

console = Console(highlight=False)
"""Shared console. `file` is resolved at print time, so click's test runner captures it."""


def num(value: float | None, spec: str = ".4f") -> str:
    """Format `value`, showing "-" for missing and "n/a" for NaN or infinite."""
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "n/a"
    return format(value, spec)


def probe_note(failed_probes: int) -> str:
    """Short table note explaining a sensitivity result built from failed probes."""
    if failed_probes == 1:
        return "[yellow]~ one probe failed, swing estimated[/yellow]"
    if failed_probes >= 2:
        return "[red]both probes failed, no information[/red]"
    return ""


def j_cell(record: ledger.RunRecord) -> str:
    if record.simulation_error:
        return "[red]FAILED[/red]"
    return "-" if record.j is None else f"{record.j:.4f}"


def health_cell(is_clean: bool | None) -> str:
    if is_clean is None:
        return "[dim]-[/dim]"
    return "[green]clean[/green]" if is_clean else "[yellow]attention[/yellow]"


def make_table(title: str, *, caption: str | None = None) -> Table:
    return Table(title=title, caption=caption, box=box.SIMPLE_HEAVY, header_style="bold cyan")


def print_outcome_line(record: ledger.RunRecord) -> None:
    """One compact line per finished trial, with what it was changing."""
    if record.simulation_error:
        console.print(f"[red]{record.run_id}: simulation failed:[/red] {record.simulation_error}")
        return

    moved = ", ".join(f"{name}={value:.6g}" for name, value in (record.tuned_values or {}).items())
    detail = " | ".join(part for part in (record.group, record.stage, moved) if part)
    j_text = "J=not scored" if record.j is None else f"J={record.j:.4f}"
    console.print(
        f"{record.run_id}: {j_text}  {health_cell(record.prt_is_clean)}"
        + (f"  [dim]{detail}[/dim]" if detail else ""),
        highlight=False,
    )


def parameters_table() -> Table:
    """Every tunable parameter grouped in tuning priority order."""
    table = make_table("Tunable parameters")
    for column in ("Priority", "Group", "Parameter", "Default", "Low", "High", "Description"):
        table.add_column(column, overflow="fold")

    for priority, group in enumerate(constants.GROUP_TUNING_PRIORITY_ORDER, 1):
        for index, spec in enumerate(parameters.get_parameters_in_group(group)):
            table.add_row(
                str(priority) if index == 0 else "",
                group if index == 0 else "",
                spec.name,
                f"{spec.default:g}",
                f"{spec.bounds[0]:g}",
                f"{spec.bounds[1]:g}",
                spec.description,
            )
        table.add_section()
    return table


def trials_table(
    records: typing.Sequence[ledger.RunRecord], *, title: str, value_parameter: str | None = None
) -> Table:
    """Trials from one search, with per-vector NRMSE and the best row marked."""
    vector_names = sorted({name for record in records for name in (record.vector_nrmse or {})})
    best = ledger.get_best_record(list(records))
    table = make_table(title)
    table.add_column("Run")
    if value_parameter:
        table.add_column(value_parameter, justify="right")

    table.add_column("J", justify="right")
    for name in vector_names:
        table.add_column(name, justify="right")

    table.add_column("Health")
    for record in records:
        is_best = best is not None and record.run_id == best.run_id
        row = [record.run_id + (" *" if is_best else "")]
        if value_parameter:
            row.append(num((record.tuned_values or {}).get(value_parameter), ".6g"))
        row.append(j_cell(record))
        row.extend(num((record.vector_nrmse or {}).get(name)) for name in vector_names)
        row.append(health_cell(record.prt_is_clean))
        table.add_row(*row, style="bold green" if is_best else None)
    table.caption = "* best J"
    return table


def state_table(
    state: dict[str, float], *, title: str = "Parameter state", show_all: bool = True
) -> Table:
    """Final state against defaults and bounds; changed parameters first."""
    table = make_table(title)
    for column, justify in (
        ("Parameter", "left"),
        ("Group", "left"),
        ("Value", "right"),
        ("Default", "right"),
        ("Change", "right"),
        ("Bounds", "left"),
        ("Note", "left"),
    ):
        table.add_column(column, justify=justify, overflow="fold")  # type: ignore[arg-type]

    for row in reporting.get_state_rows(state):
        if not show_all and not row.changed:
            continue
        change = (
            "-" if not row.changed or row.change_percent is None else f"{row.change_percent:+.1f}%"
        )
        note = (
            f"[cyan]{row.beyond} registered bounds[/cyan]"
            if row.beyond
            else f"[yellow]at {row.pinned} bound[/yellow]"
            if row.pinned
            else ("changed" if row.changed else "[dim]default[/dim]")
        )
        table.add_row(
            row.name,
            row.group,
            f"{row.value:.6g}",
            f"{row.default:.6g}",
            change,
            f"{row.bounds[0]:g} to {row.bounds[1]:g}",
            note,
            style=None if row.changed else "dim",
        )
    return table


def objective_table(record: ledger.RunRecord, *, target_j: float | None = None) -> Table:
    """Per-vector NRMSE, weight, weighted contribution and share of J."""
    caption = None
    if record.j is not None:
        caption = f"J = {record.j:.4f}"
        if target_j is not None:
            caption += (
                f" (target {target_j:.4f}: {'reached' if record.j <= target_j else 'not reached'})"
            )

    table = make_table("Objective breakdown", caption=caption)
    for column in ("Vector", "NRMSE", "Weight", "Weighted", "Share of J"):
        table.add_column(column, justify="left" if column == "Vector" else "right")

    for row in reporting.get_objective_rows(record):
        table.add_row(
            row.name,
            f"{row.nrmse:.4f}",
            num(row.weight, ".2f"),
            num(row.contribution),
            "-" if row.share is None else f"{row.share * 100:.1f}%",
        )
    return table


def well_table(record: ledger.RunRecord) -> Table | None:
    """Per-well water-cut and GOR NRMSE, or `None` if the record has none."""
    rows = reporting.get_well_rows(record)
    if not rows:
        return None

    scored = ", ".join(record.scored_wells or []) or "none (field totals only)"
    table = make_table("Per-well match", caption=f"counted in J: {scored}; others are diagnostics")
    for column in ("Well", "Water cut NRMSE", "GOR NRMSE", "In J"):
        table.add_column(column, justify="left" if column == "Well" else "right")

    for row in rows:
        table.add_row(
            row.well,
            num(row.watercut_nrmse),
            num(row.gor_nrmse),
            "[green]yes[/green]" if row.in_objective else "[dim]no[/dim]",
            style=None if row.in_objective else "dim",
        )
    return table


def group_outcomes_table(outcomes: typing.Sequence[coordinate_descent.GroupOutcome]) -> Table:
    """One row per tuned group: what it cost and how much J it bought."""
    table = make_table("Tuning path by group")
    for column in ("Group", "Start J", "End J", "Change", "Sims", "Passes", "Target"):
        table.add_column(column, justify="left" if column == "Group" else "right")

    for outcome in outcomes:
        delta = outcome.ending_j - outcome.starting_j
        table.add_row(
            outcome.group,
            f"{outcome.starting_j:.4f}",
            f"{outcome.ending_j:.4f}",
            f"[green]{delta:+.4f}[/green]" if delta < 0 else f"[dim]{delta:+.4f}[/dim]",
            str(outcome.evaluations),
            str(outcome.passes),
            "[green]reached[/green]" if outcome.reached_target else "no",
        )
    return table


def parameter_steps_table(outcomes: typing.Sequence[coordinate_descent.GroupOutcome]) -> Table:
    """One row per parameter per pass, so the group's real driver is visible."""
    table = make_table("Parameter by parameter")
    for column in (
        "Group",
        "Pass",
        "Parameter",
        "Window",
        "Start",
        "End",
        "J before",
        "J after",
        "Sims",
    ):
        table.add_column(
            column, justify="left" if column in ("Group", "Parameter", "Window") else "right"
        )

    for outcome in outcomes:
        for step in outcome.parameter_outcomes:
            improved = step.ending_j < step.starting_j
            table.add_row(
                step.group,
                str(step.pass_index),
                step.parameter,
                f"{step.window[0]:.6g} to {step.window[1]:.6g}",
                f"{step.start_value:.6g}",
                f"{step.end_value:.6g}",
                f"{step.starting_j:.4f}",
                f"[green]{step.ending_j:.4f}[/green]" if improved else f"{step.ending_j:.4f}",
                str(step.evaluations),
                style=None if improved else "dim",
            )
    return table


def sensitivity_table(
    results: typing.Sequence[sensitivity.SensitivityResult],
    parameter_groups: dict[str, str],
    ranks: dict[str, float],
) -> Table:
    """Parameter sensitivity with group, rank and the probe values used."""
    table = make_table("Parameter sensitivity (most sensitive first)")
    for column in (
        "Rank",
        "Parameter",
        "Group",
        "Low value",
        "High value",
        "J at low",
        "J at high",
        "Swing",
    ):
        table.add_column(column, justify="left" if column in ("Parameter", "Group") else "right")

    for result in results:
        table.add_row(
            f"{ranks.get(result.parameter, 0):g}",
            result.parameter,
            parameter_groups.get(result.parameter, "-"),
            f"{result.low_value:.6g}",
            f"{result.high_value:.6g}",
            num(result.j_at_low) if math.isfinite(result.j_at_low) else "[red]FAILED[/red]",
            num(result.j_at_high) if math.isfinite(result.j_at_high) else "[red]FAILED[/red]",
            num(result.swing),
            probe_note(result.failed_probes),
        )
    return table


def detailed_sensitivity_table(
    results: typing.Sequence[sensitivity.DetailedSensitivityResult],
    parameter_groups: dict[str, str],
    ranks: dict[str, float],
) -> Table:
    """Parameter sensitivity with one swing column per scored vector."""
    vector_names = sorted({name for result in results for name in result.vector_swings})
    table = make_table("Parameter sensitivity by vector (most sensitive first)")
    table.add_column("Rank", justify="right")
    table.add_column("Parameter")
    table.add_column("Group")
    table.add_column("Swing (J)", justify="right")
    for name in vector_names:
        table.add_column(name, justify="right")

    table.add_column("Note", overflow="fold")
    for result in results:
        table.add_row(
            f"{ranks.get(result.parameter, 0):g}",
            result.parameter,
            parameter_groups.get(result.parameter, "-"),
            num(result.swing),
            *(num(result.vector_swings.get(name, 0.0)) for name in vector_names),
            probe_note(result.failed_probes),
        )
    return table


def group_sensitivity_table(
    groups: typing.Sequence[sensitivity.GroupSensitivity], *, method: str
) -> Table:
    """Group ranking with every score, so the ordering can be audited."""
    table = make_table(
        "Group sensitivity ranking",
        caption=f"ordered by {method.replace('_', ' ')}; lower rank means more sensitive",
    )
    for column in (
        "#",
        "Group",
        "Params",
        "Mean rank",
        "Rank sum",
        "Best rank",
        "Total swing",
        "Share",
        "Parameters (rank)",
    ):
        table.add_column(
            column,
            justify="left" if column in ("Group", "Parameters (rank)") else "right",
            overflow="fold",
        )

    for group in groups:
        members = ", ".join(
            f"{name} ({group.parameter_ranks[name]:g})" for name in group.parameters
        )
        table.add_row(
            str(group.position),
            group.group,
            str(len(group.parameters)),
            f"{group.mean_rank:.2f}",
            f"{group.rank_sum:g}",
            f"{group.best_rank:g}",
            f"{group.total_swing:.4f}",
            f"{group.swing_share * 100:.1f}%",
            members,
        )
    return table


def ledger_table(records: typing.Sequence[ledger.RunRecord]) -> Table:
    """The run ledger, with group, tuned parameter, value and stage columns."""
    table = make_table("Run ledger")
    for column in (
        "Run",
        "Strategy",
        "Stage",
        "Group",
        "Parameter(s)",
        "Value(s)",
        "J",
        "Health",
        "Note",
    ):
        table.add_column(column, justify="right" if column == "J" else "left", overflow="fold")

    best = ledger.get_best_record(list(records))
    for record in records:
        values = ", ".join(f"{v:.6g}" for v in (record.tuned_values or {}).values())
        is_best = best is not None and record.run_id == best.run_id
        table.add_row(
            record.run_id + (" *" if is_best else ""),
            record.strategy or "-",
            record.stage or "-",
            record.group or "-",
            ", ".join(record.tuned_parameters or []) or "-",
            values or "-",
            j_cell(record),
            health_cell(record.prt_is_clean),
            record.note,
            style="bold green" if is_best else None,
        )
    table.caption = "* best J in view"
    return table


def parameter_history_table(histories: typing.Sequence[ledger.ParameterHistory]) -> Table:
    """What the ledger says about each parameter tried on its own."""
    table = make_table(
        "Per-parameter history",
        caption="single-parameter trials only; J span near zero means the parameter did not matter",
    )
    for column in (
        "Parameter",
        "Group",
        "Trials",
        "Failed",
        "Best J",
        "Value at best",
        "J span",
        "Range tried",
        "Last run",
    ):
        table.add_column(
            column,
            justify="left"
            if column in ("Parameter", "Group", "Last run", "Range tried")
            else "right",
            overflow="fold",
        )

    for history in histories:
        flat = history.j_span is not None and history.j_span < reporting.FLAT_PARAMETER_J_SPAN
        table.add_row(
            history.parameter,
            history.group or "-",
            str(history.trials),
            str(history.failed),
            num(history.best_j),
            num(history.best_value, ".6g"),
            num(history.j_span),
            f"{num(history.tried_low, '.6g')} to {num(history.tried_high, '.6g')}",
            history.last_run_id,
            style="dim" if flat else None,
        )
    return table


def group_history_table(histories: typing.Sequence[ledger.GroupHistory]) -> Table:
    """What the ledger says about each tuning group."""
    table = make_table("Per-group history")
    for column in ("Group", "Trials", "Best J", "Best run", "Parameters touched"):
        table.add_column(
            column, justify="right" if column in ("Trials", "Best J") else "left", overflow="fold"
        )

    for history in histories:
        table.add_row(
            history.group,
            str(history.trials),
            num(history.best_j),
            history.best_run_id or "-",
            ", ".join(history.parameters_touched) or "-",
        )
    return table


def health_table(prt_report: PrtReport) -> Table:
    """The `.PRT` health numbers as one table."""
    table = make_table("Run health")
    table.add_column("Check")
    table.add_column("Value", justify="right")
    table.add_row("Report steps completed", str(len(prt_report.completed_report_steps)))
    table.add_row("Last simulated date", str(prt_report.last_simulated_date))
    table.add_row("Errors", str(prt_report.errors))
    table.add_row("Bugs", str(prt_report.bugs))
    table.add_row("Warnings", str(prt_report.warnings))
    if prt_report.unconverged_well_counts:
        worst = sorted(prt_report.unconverged_well_counts.items(), key=lambda item: -item[1])[:3]
        table.add_row("Most convergence warnings", ", ".join(f"{w} ({c})" for w, c in worst))

    table.add_row(
        "Considered clean", "[green]yes[/green]" if prt_report.is_clean else "[yellow]no[/yellow]"
    )
    return table


def print_run_report(
    record: ledger.RunRecord,
    *,
    records: list[ledger.RunRecord],
    prt_report: PrtReport | None = None,
    target_j: float | None = None,
    show_wells: bool = True,
) -> None:
    """Terminal version of `reporting.render_run_report`."""
    identity = (
        f"[bold]{record.run_id}[/bold]  logged {record.created_at}\n"
        f"strategy: {record.strategy or '-'}   group: {record.group or '-'}   "
        f"stage: {record.stage or '-'}\n"
        f"baseline: {record.baseline or '-'}   deck: {record.base_deck or '-'}\n"
        f"parameters moved: {', '.join(record.tuned_parameters or []) or '-'}"
    )
    if record.tuned_values:
        identity += "\nvalues: " + ", ".join(
            f"{n}={v:.6g}" for n, v in record.tuned_values.items()
        )
    if record.note:
        identity += f"\nnote: {record.note}"
    console.print(Panel(identity, title="History match snapshot", expand=False))

    if record.simulation_error:
        console.print(f"[red]Simulation failed:[/red] {record.simulation_error}")
    elif record.j is None:
        console.print("This run was not scored against the observed history.")
    else:
        console.print(objective_table(record, target_j=target_j))
        baseline = reporting.get_baseline_record(records)
        if baseline is not None and baseline.j and baseline.run_id != record.run_id:
            drop = (baseline.j - record.j) / baseline.j * 100.0
            console.print(
                f"Baseline {baseline.run_id} scored {baseline.j:.4f}; this run is "
                f"{abs(drop):.1f}% {'better' if drop >= 0 else 'worse'}."
            )

    if show_wells:
        wells = well_table(record)
        if wells is not None:
            console.print(wells)

    console.print(state_table(record.parameter_state, title="Final parameter state"))
    histories = ledger.summarize_parameters(records)
    if histories:
        console.print(parameter_history_table(histories))
    if prt_report is not None:
        console.print(health_table(prt_report))

    suggestions = reporting.next_steps(records, target_j=target_j)
    if suggestions:
        console.print(
            Panel("\n".join(f"- {s}" for s in suggestions), title="What to try next", expand=False)
        )
    print_key(
        glossary.OBJECTIVE if record.j is not None else (),
        glossary.WELLS if show_wells and reporting.get_well_rows(record) else (),
        glossary.STATE,
        glossary.PARAMETER_HISTORY if histories else (),
        ("Health",) if prt_report is not None else (),
    )


def cleanup_table(actions: typing.Sequence[cleanup.Action], *, scope: str) -> Table:
    """What `nagcsu clean` is about to do, one row per run."""
    table = make_table(f"Runs to clean (scope: {scope})")
    for column in ("Run", "Strategy", "Stage", "J", "Ledger", "Files", "Freed"):
        table.add_column(
            column, justify="right" if column in ("J", "Freed") else "left", overflow="fold"
        )

    for action in actions:
        record = action.target.record
        if action.delete_whole_directory:
            files = "whole directory"
        elif action.files_to_delete or action.files_to_keep:
            files = f"delete {len(action.files_to_delete)}, keep {len(action.files_to_keep)}"
        else:
            files = "-" if action.target.directory is None else "untouched"

        size = action.bytes_to_free
        table.add_row(
            action.target.run_id,
            (record.strategy or "-") if record else "[dim]no record[/dim]",
            (record.stage or "-") if record else "-",
            j_cell(record) if record else "-",
            "remove" if action.remove_record else "keep",
            files,
            f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.0f} KB",
        )
    return table


def protected_table(protected: typing.Sequence[tuple[cleanup.Target, str]]) -> Table:
    """Runs that matched the selection but are kept, and why."""
    table = make_table("Protected (not cleaned)")
    table.add_column("Run")
    table.add_column("J", justify="right")
    table.add_column("Why")
    for target, reason in protected:
        table.add_row(target.run_id, j_cell(target.record) if target.record else "-", reason)
    return table


def range_table(suggestions: typing.Sequence[ranges.RangeSuggestion]) -> Table:
    """Suggested starting ranges with the evidence behind each."""
    table = make_table(
        "Suggested search ranges",
        caption="from single-parameter trials; other parameters were wherever the search had them",
    )
    for column in (
        "Parameter",
        "Group",
        "Trials",
        "Registered",
        "Tried",
        "Best",
        "J span",
        "Status",
        "Suggested range",
        "Why",
    ):
        table.add_column(
            column,
            justify="right" if column in ("Trials", "Best", "J span") else "left",
            overflow="fold",
        )

    colors = {"bracketed": "green", "flat": "dim", "insufficient": "dim"}
    for suggestion in suggestions:
        suggested = (
            "-"
            if suggestion.low is None or suggestion.high is None
            else f"{suggestion.low:.6g} to {suggestion.high:.6g}"
        )
        color = colors.get(suggestion.status, "yellow")
        table.add_row(
            suggestion.parameter,
            suggestion.group or "-",
            str(suggestion.trials),
            f"{suggestion.registered_bounds[0]:g} to {suggestion.registered_bounds[1]:g}",
            f"{suggestion.tried_low:.6g} to {suggestion.tried_high:.6g}",
            f"{suggestion.best_value:.6g}",
            num(suggestion.j_span),
            f"[{color}]{suggestion.status}[/{color}]",
            suggested,
            suggestion.note,
        )
    return table


def key_table(terms: typing.Iterable[str]) -> Table | None:
    """A compact key explaining `terms`, or `None` if there is nothing to explain."""
    entries = glossary.lookup(terms)
    if not entries:
        return None
    table = Table(
        title="Key",
        box=box.SIMPLE,
        header_style="bold dim",
        title_style="bold dim",
        show_edge=False,
        pad_edge=False,
    )
    table.add_column("Term", style="bold", no_wrap=True)
    table.add_column("What it is", overflow="fold")
    table.add_column("What it tells you", overflow="fold")
    for entry in entries:
        table.add_row(entry.term, entry.meaning, entry.impact, style="dim")
    return table


def print_key(*term_groups: typing.Iterable[str]) -> None:
    """Print one key covering every term in `term_groups`, unless keys are turned off."""
    if not KEY_ENABLED:
        return

    terms = [term for group in term_groups for term in group]
    table = key_table(terms)
    if table is not None:
        console.print(table)


def print_baseline(baseline: "context.Baseline") -> None:
    """One line saying where this command starts from."""
    if baseline.run_id:
        j_text = "" if baseline.j is None else f", J={baseline.j:.4f}"
        console.print(
            f"Baseline: {baseline.spec} -> run {baseline.run_id}{j_text}; deck: {baseline.deck_source}"
        )
    elif baseline.spec == "default":
        console.print("Baseline: registered defaults; deck: " + baseline.deck_source)
    else:
        console.print(f"Baseline: {baseline.spec}; deck: {baseline.deck_source}")

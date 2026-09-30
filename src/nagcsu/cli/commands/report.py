"""`nagcsu report`: inspect and export runs from the ledger."""

import typing

import click

from nagcsu import glossary, ledger, prt, ranges, reporting
from nagcsu.cli import context, display


@click.group(name="report")
def report() -> None:
    """List logged runs and render a Markdown snapshot for one of them."""


@report.command(name="list")
@click.option("--strategy", default=None, help="Only list records with this exact strategy.")
@click.option(
    "--group", "group_name", default=None, help="Only list records with this exact group."
)
@click.option(
    "--run-id-prefix",
    default=None,
    help="Only list records whose `run_id` starts with this prefix.",
)
@click.option(
    "--run-id-regex",
    default=None,
    help="Only list records whose `run_id` matches this regex pattern.",
)
@click.option(
    "--parameter", "parameter_name", default=None, help="Only list runs that moved this parameter."
)
@click.option(
    "--stage",
    default=None,
    help="Only list runs whose stage starts with this text, e.g. descent or sensitivity.",
)
@click.option("--limit", default=20, show_default=True, help="Most recent matching runs to show.")
@click.pass_context
def list_(
    ctx: click.Context,
    strategy: str | None,
    group_name: str | None,
    run_id_prefix: str | None,
    run_id_regex: str | None,
    parameter_name: str | None,
    stage: str | None,
    limit: int,
) -> None:
    """List logged runs, optionally narrowed to one strategy, group or run ID family."""
    project_config, _ = context.load(ctx)
    records = ledger.load(project_config.get_resolved_path(project_config.ledger_path))
    if not records:
        click.echo("No runs logged yet. Try `nagcsu run` first.")
        return

    filtered_records = ledger.filter_records(
        records,
        strategy=strategy,
        group=group_name,
        run_id_prefix=run_id_prefix,
        run_id_regex=run_id_regex,
        parameter=parameter_name,
        stage=stage,
        limit=limit,
    )
    if not filtered_records:
        click.echo("No runs match the selected filters.")
        return

    display.console.print(display.ledger_table(filtered_records))
    display.print_key(glossary.LEDGER, ("J",))
    best = ledger.get_best_record(filtered_records)
    if best:
        click.echo(f"Best in view: {best.run_id} (J={best.j:.4f})")


@report.command(name="parameters")
@click.option("--group", "group_name", default=None, help="Only show parameters in this group.")
@click.pass_context
def parameters_(ctx: click.Context, group_name: str | None) -> None:
    """Show what has been tried per parameter and per group, to decide what to touch next.

    Built from single-parameter trials in the ledger (sweeps, descent
    probes, sensitivity probes). A parameter with many trials and a J span
    near zero has been explored and does not matter; a parameter with a
    large span still responds and is worth refining.
    """
    project_config, _ = context.load(ctx)
    records = ledger.load(project_config.get_resolved_path(project_config.ledger_path))
    if not records:
        click.echo("No runs logged yet. Try `nagcsu run` first.")
        return

    histories = ledger.summarize_parameters(records)
    if group_name:
        histories = [history for history in histories if history.group == group_name]
    if not histories:
        click.echo("No single-parameter trials logged for the selected filter.")
    else:
        display.console.print(display.parameter_history_table(histories))

    group_histories = ledger.summarize_groups(records)
    if group_name:
        group_histories = [history for history in group_histories if history.group == group_name]
    if group_histories:
        display.console.print(display.group_history_table(group_histories))

    for step in reporting.next_steps(records):
        click.echo(f"- {step}")
    display.print_key(glossary.PARAMETER_HISTORY, glossary.GROUP_HISTORY, ("Group",))


@report.command(name="ranges")
@click.option("--group", "group_name", default=None, help="Only show parameters in this group.")
@click.option(
    "--near-best",
    default=ranges.NEAR_BEST_FRACTION,
    show_default=True,
    help="A tried value counts as good when its J is within this fraction of the J span above the best.",
)
@click.option(
    "--min-values",
    default=ranges.MIN_DISTINCT_VALUES,
    show_default=True,
    help="Distinct tried values needed before a range is suggested.",
)
@click.pass_context
def ranges_(ctx: click.Context, group_name: str | None, near_best: float, min_values: int) -> None:
    """Suggest a starting range per parameter from the trials logged so far.

    For each parameter, finds the values that scored close to its best and
    proposes a range around them. If the best value sits at the edge of
    what was tried, the range extends past that edge instead of only
    saying "widen it". Built from single-parameter trials, run with the
    other parameters wherever the search had them at the time, so use it
    as a starting point for the next batch, not a verdict.
    """
    project_config, _ = context.load(ctx)
    records = ledger.load(project_config.get_resolved_path(project_config.ledger_path))
    suggestions = ranges.suggest_ranges(
        records, near_best_fraction=near_best, min_distinct_values=min_values
    )
    if group_name:
        suggestions = [s for s in suggestions if s.group == group_name]
    if not suggestions:
        click.echo(
            "No single-parameter trials to build ranges from. Run `match sweep` or `match auto` first."
        )
        return
    display.console.print(display.range_table(suggestions))
    display.print_key(glossary.RANGES, ("J span", "Best J", "Trials"))
    flags = [typing.cast(str, s.range_flag()) for s in suggestions if s.range_flag()]
    if flags:
        click.echo("\nStart the next batch with:")
        click.echo("  nagcsu match auto " + " ".join(flags))
    leave = [s.parameter for s in suggestions if s.status == "flat"]
    if leave:
        click.echo(f"Leave out (flat): {', '.join(leave)}")


@report.command(name="show")
@click.argument("run_id", default="latest")
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Write a Markdown report to this path instead of printing tables.",
)
@click.option(
    "--markdown",
    "as_markdown",
    is_flag=True,
    default=False,
    help="Print the Markdown report to the terminal instead of rich tables.",
)
@click.option(
    "--per-well/--no-per-well",
    "per_well",
    default=True,
    help="Include the per-well water cut and GOR match table when the run has one.",
)
@click.pass_context
def show(
    ctx: click.Context, run_id: str, output_path: str | None, as_markdown: bool, per_well: bool
) -> None:
    """Show a detailed report for one logged run (tables by default, Markdown with --output).

    Pass `latest` (the default) for the most recently logged run, `best`
    for the lowest-J run so far, or an explicit run ID such as `run_0003`.
    """
    project_config, _ = context.load(ctx)
    records = ledger.load(project_config.get_resolved_path(project_config.ledger_path))
    if not records:
        raise click.ClickException("No runs logged yet.")

    record = resolve_run_id(records, run_id)
    prt_report = None
    prt_path = project_config.get_resolved_path(project_config.output_root) / record.run_id
    matching_prt_files = list(prt_path.glob("*.PRT"))
    if matching_prt_files:
        prt_report = prt.parse(matching_prt_files[0])

    target_j = project_config.objective.target_j
    if output_path:
        written = reporting.write_run_report(
            record,
            output_path,
            prt_report=prt_report,
            records=records,
            target_j=target_j,
            show_wells=per_well,
        )
        click.echo(f"Wrote {written}")
    elif as_markdown:
        click.echo(
            reporting.render_run_report(
                record,
                prt_report=prt_report,
                records=records,
                target_j=target_j,
                show_wells=per_well,
            )
        )
    else:
        display.print_run_report(
            record,
            records=records,
            prt_report=prt_report,
            target_j=target_j,
            show_wells=per_well,
        )


def resolve_run_id(records: list[ledger.RunRecord], run_id: str) -> ledger.RunRecord:
    """Resolve the `latest`/`best`/explicit-ID argument `nagcsu report show` takes."""
    if run_id == "latest":
        return records[-1]
    if run_id == "best":
        best = ledger.get_best_record(records)
        if best is None:
            raise click.ClickException("No scored runs logged yet, so there is no best run.")
        return best
    for record in records:
        if record.run_id == run_id:
            return record
    raise click.ClickException(f"No run logged with ID {run_id!r}")

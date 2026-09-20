"""`nagcsu report`: inspect and export runs from the ledger."""

import click

from nagcsu import ledger, prt, reporting
from nagcsu.cli import context


@click.group(name="report")
def report() -> None:
    """List logged runs and render a Markdown snapshot for one of them."""


@report.command(name="list")
@click.option("--limit", default=20, show_default=True, help="Most recent runs to show.")
@click.pass_context
def list_(ctx: click.Context, limit: int) -> None:
    """List the most recent logged runs, best-scored first within each group."""
    project_config, _ = context.load(ctx)
    records = ledger.load(project_config.resolved_path(project_config.ledger_path))
    if not records:
        click.echo("No runs logged yet. Try `nagcsu run` first.")
        return

    click.echo(f"{'Run ID':<20}{'Group':<24}{'Strategy':<20}{'J':>10}  Note")
    for record in records[-limit:]:
        if record.simulation_error:
            j_text = "FAILED"
        elif record.j is not None:
            j_text = f"{record.j:.4f}"
        else:
            j_text = "-"
        click.echo(
            f"{record.run_id:<20}{(record.group or '-'):<24}{(record.strategy or '-'):<20}{j_text:>10}  {record.note}"
        )

    best = ledger.best_record(records)
    if best:
        click.echo(f"\nBest so far: {best.run_id} (J={best.j:.4f})")


@report.command(name="show")
@click.argument("run_id", default="latest")
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Write the report to this path instead of printing it.",
)
@click.pass_context
def show(ctx: click.Context, run_id: str, output_path: str | None) -> None:
    """Render a Markdown snapshot report for one logged run.

    Pass `latest` (the default) for the most recently logged run, `best`
    for the lowest-J run so far, or an explicit run ID such as `run_0003`.
    """
    project_config, _ = context.load(ctx)
    records = ledger.load(project_config.resolved_path(project_config.ledger_path))
    if not records:
        raise click.ClickException("No runs logged yet.")

    record = resolve_run_id(records, run_id)
    prt_report = None
    prt_path = project_config.resolved_path(project_config.output_root) / record.run_id
    matching_prt_files = list(prt_path.glob("*.PRT"))
    if matching_prt_files:
        prt_report = prt.parse(matching_prt_files[0])

    if output_path:
        written = reporting.write_run_report(record, output_path, prt_report=prt_report)
        click.echo(f"Wrote {written}")
    else:
        click.echo(reporting.render_run_report(record, prt_report=prt_report))


def resolve_run_id(records: list[ledger.RunRecord], run_id: str) -> ledger.RunRecord:
    """Resolve the `latest`/`best`/explicit-ID argument `nagcsu report show` takes."""
    if run_id == "latest":
        return records[-1]
    if run_id == "best":
        best = ledger.best_record(records)
        if best is None:
            raise click.ClickException("No scored runs logged yet, so there is no best run.")
        return best
    for record in records:
        if record.run_id == run_id:
            return record
    raise click.ClickException(f"No run logged with ID {run_id!r}")

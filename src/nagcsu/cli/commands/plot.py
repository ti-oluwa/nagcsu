"""`nagcsu plot`: interactive plots for tuning sessions and run results."""

import pathlib

import click
import plotly.graph_objects as go

from nagcsu import ledger, pipeline, plotting, simulate, summary
from nagcsu.cli import context
from nagcsu.cli.commands.report import resolve_run_id


@click.group(name="plot")
def plot() -> None:
    """Interactive HTML plots: convergence, and simulated vs observed curves."""


@plot.command(name="convergence")
@click.option("--strategy", default=None, help="Only plot records with this exact strategy.")
@click.option(
    "--group", "group_name", default=None, help="Only plot records with this exact group."
)
@click.option(
    "--run-id-prefix",
    default=None,
    help="Only plot records whose run_id starts with this prefix.",
)
@click.option(
    "--run-id-regex",
    default=None,
    help="Only plot records whose run_id matches this regex pattern.",
)
@click.option(
    "--limit", default=None, type=int, help="Only plot the most recent N matching records."
)
@click.option(
    "--show/--no-show",
    default=False,
    help="Open the plot in a browser window before saving it, if an output path is given.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
    help="Where to write the plot. If omitted and `--show` is not set, defaults to "
    "`convergence.html`. An .html path is interactive and needs no extra packages; "
    "any other extension (.png, .svg, .pdf) needs `pip install kaleido`.",
)
@click.pass_context
def convergence(
    ctx: click.Context,
    strategy: str | None,
    group_name: str | None,
    run_id_prefix: str | None,
    run_id_regex: str | None,
    limit: int | None,
    show: bool,
    output_path: pathlib.Path | None,
) -> None:
    """Plot J and each scored vector's NRMSE across a sequence of ledger trials.

    With no filters, plots every logged record in order, which rarely
    makes sense once you have run more than one kind of search; narrow
    it with `--strategy coordinate_descent`, `--group aquifer`,
    `--run-id-prefix auto`, or a `--run-id-regex '^run_0[01]$'`.
    """
    project_config, _ = context.load(ctx)
    records = ledger.load(project_config.get_resolved_path(project_config.ledger_path))
    filtered = ledger.filter_records(
        records,
        strategy=strategy,
        group=group_name,
        run_id_prefix=run_id_prefix,
        run_id_regex=run_id_regex,
        limit=limit,
    )
    if not filtered:
        raise click.ClickException("No matching ledger records to plot.")

    try:
        figure = plotting.plot_convergence(filtered)
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    resolved_output_path = output_path or (None if show else pathlib.Path("convergence.html"))
    written = show_and_save_figure(figure, show=show, output_path=resolved_output_path)
    if written is not None:
        click.echo(f"Plotted {len(filtered)} trial(s) to {written}")


@plot.command(name="match")
@click.argument("run_id", default="latest")
@click.option(
    "--show/--no-show",
    default=False,
    help="Open the plot in a browser window before saving it, if an output path is given.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
    help="Where to write the plot. If omitted and `--show` is not set, defaults to "
    "<run output dir>/match.html. An .html path is interactive and needs no extra "
    "packages; any other extension (.png, .svg, .pdf) needs `pip install kaleido`.",
)
@click.pass_context
def match_(ctx: click.Context, run_id: str, show: bool, output_path: pathlib.Path | None) -> None:
    """Plot one run's simulated pressure, water cut and GOR against observed history.

    Pass `latest` (the default) for the most recently logged run,
    `best` for the lowest-J run so far, or an explicit run ID such as
    `run_0003` or `auto_final`.
    """
    project_config, _ = context.load(ctx)
    records = ledger.load(project_config.get_resolved_path(project_config.ledger_path))
    if not records:
        raise click.ClickException("No runs logged yet.")
    record = resolve_run_id(records, run_id)

    output_dir = project_config.get_resolved_path(project_config.output_root) / record.run_id
    case_basename = simulate.find_case_basename(output_dir)
    if case_basename is None:
        raise click.ClickException(f"No .UNSMRY output found under {output_dir}")

    simulated_frame = summary.load_summary(case_basename, wells=list(project_config.wells))
    observed_frame = pipeline.load_observed_history(project_config)
    figure = plotting.plot_match(
        simulated_frame, observed_frame, title=f"Simulated vs observed: {record.run_id}"
    )

    resolved_output_path = output_path or (None if show else output_dir / "match.html")
    written = show_and_save_figure(figure, show=show, output_path=resolved_output_path)
    if written is not None:
        click.echo(f"Plotted {record.run_id} to {written}")


def show_and_save_figure(
    figure: go.Figure, *, show: bool, output_path: pathlib.Path | None
) -> pathlib.Path | None:
    """Open the figure, then save it if an output path was provided."""
    if show:
        figure.show()
    if output_path is None:
        return None
    try:
        return plotting.save_figure(figure, output_path)
    except ImportError as error:
        raise click.ClickException(str(error)) from error

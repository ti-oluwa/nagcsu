"""`nagcsu init`: create a project config in the current directory."""

import pathlib

import click

from nagcsu import config as config_module
from nagcsu import history


@click.command(name="init")
@click.option(
    "--deck",
    "deck_path",
    default="Data/NigerDelta UGH1 Composite Field.DATA",
    show_default=True,
    help="Path to the OPM Flow .DATA deck this project tunes.",
)
@click.option(
    "--history",
    "history_path",
    default="Data/NigerDelta Synthetic Production History.xlsx",
    show_default=True,
    help="Path to the observed production/pressure history file (.xlsx, .xls or .csv).",
)
@click.option(
    "--flow-executable",
    default="flow",
    show_default=True,
    help="Name or path of the OPM Flow executable.",
)
@click.pass_context
def init(ctx: click.Context, deck_path: str, history_path: str, flow_executable: str) -> None:
    """Create a `nagcsu.yaml` project config in the current directory.

    Safe to run from the repository root with the defaults, which point
    at the deck and history file already in `Data/`. If the history
    file already exists at this path, its header row is peeked at to
    guess the right `date_column` and (for a long-format, one-row-per-
    well-per-date file) `well_column`, rather than assuming `DATE`;
    both can still be overridden by hand in `nagcsu.yaml` afterwards.
    """
    config_path: pathlib.Path = ctx.obj["config_path"]
    if config_path.exists():
        raise click.ClickException(
            f"{config_path} already exists; delete it first or pass --config to write elsewhere"
        )

    history_config = config_module.HistoryConfig(path=pathlib.Path(history_path))
    resolved_history_path = (pathlib.Path(".").resolve() / history_path).resolve()
    if resolved_history_path.exists():
        try:
            columns = history.peek_columns(resolved_history_path)
        except (ValueError, OSError) as error:
            click.echo(f"Could not read {history_path}'s headers ({error}); using defaults.")
        else:
            detected_date_column = history.detect_date_column(columns)
            if detected_date_column:
                history_config.date_column = detected_date_column
            long_format = history.detect_long_format_columns(columns)
            if long_format:
                history_config.well_column = long_format["well"]

    project_config = config_module.ProjectConfig(
        deck_path=pathlib.Path(deck_path),
        flow_executable=flow_executable,
        history=history_config,
        root=pathlib.Path(".").resolve(),
    )
    config_module.save(project_config, config_path)
    click.echo(f"Wrote {config_path}")
    if history_config.date_column != "DATE" or history_config.well_column:
        click.echo(
            f"Detected history.date_column={history_config.date_column!r}"
            + (
                f", history.well_column={history_config.well_column!r}"
                if history_config.well_column
                else ""
            )
            + " from the file's headers; check nagcsu.yaml if that looks wrong."
        )
    click.echo(
        "Run `nagcsu run` to try a baseline simulation, or `nagcsu match --help` to start tuning."
    )

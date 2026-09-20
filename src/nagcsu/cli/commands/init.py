"""`nagcsu init`: create a project config in the current directory."""

import pathlib

import click

from nagcsu import config as config_module


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
    help="Path to the observed production/pressure history workbook.",
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
    at the deck and workbook already in `Data/`.
    """
    config_path: pathlib.Path = ctx.obj["config_path"]
    if config_path.exists():
        raise click.ClickException(
            f"{config_path} already exists; delete it first or pass --config to write elsewhere"
        )

    project_config = config_module.ProjectConfig(
        deck_path=pathlib.Path(deck_path),
        flow_executable=flow_executable,
        history=config_module.HistoryConfig(path=pathlib.Path(history_path)),
        root=pathlib.Path(".").resolve(),
    )
    config_module.save(project_config, config_path)
    click.echo(f"Wrote {config_path}")
    click.echo(
        "Run `nagcsu run` to try a baseline simulation, or `nagcsu match --help` to start tuning."
    )

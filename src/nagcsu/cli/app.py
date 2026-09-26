"""Assembles every subcommand under the top-level `nagcsu` click group.

Each subcommand lives in its own module under this package; this module
only wires them together and resolves the shared `--config` option every
subcommand reads through `nagcsu.cli._context.load`.
"""

import pathlib

import click

from nagcsu import __version__, constants
from nagcsu.cli.commands.init import init
from nagcsu.cli.commands.match import match
from nagcsu.cli.commands.report import report
from nagcsu.cli.commands.run import run
from nagcsu.cli.commands.sensitivity import sensitivity_


@click.group(name="nagcsu")
@click.version_option(version=__version__)
@click.option(
    "--config",
    "config_path",
    default=constants.DEFAULT_CONFIG_FILE,
    show_default=True,
    help="Path to the project config. Create one with `nagcsu init`.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    """History matching and storage-scheduling CLI for the UGH-1 sector model.

    Start with `nagcsu init` in the repository root, then `nagcsu run`
    for a baseline simulation, `nagcsu match auto` to calibrate, and
    `nagcsu report show` to see how a run's parameter state was found.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = pathlib.Path(config_path)


cli.add_command(init)
cli.add_command(run)
cli.add_command(match)
cli.add_command(sensitivity_)
cli.add_command(report)


def main() -> None:
    """Entry point registered as the `nagcsu` console script."""
    cli()

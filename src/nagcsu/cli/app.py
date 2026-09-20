"""Assembles every subcommand under the top-level `nagcsu` click group.

Each subcommand lives in its own module under this package; this module
only wires them together and resolves the shared `--config` option every
subcommand reads through :func:`nagcsu.cli._context.load`.
"""

import pathlib

import click

from nagcsu import __version__
from nagcsu.cli.init_cmd import init_cmd
from nagcsu.cli.match_cmd import match_cmd
from nagcsu.cli.report_cmd import report_cmd
from nagcsu.cli.run_cmd import run_cmd
from nagcsu.cli.sensitivity_cmd import sensitivity_cmd


@click.group(name="nagcsu")
@click.version_option(version=__version__)
@click.option(
    "--config",
    "config_path",
    default="nagcsu.yaml",
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


cli.add_command(init_cmd)
cli.add_command(run_cmd)
cli.add_command(match_cmd)
cli.add_command(sensitivity_cmd)
cli.add_command(report_cmd)


def main() -> None:
    """Entry point registered as the `nagcsu` console script."""
    cli()

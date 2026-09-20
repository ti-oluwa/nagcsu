"""Shared plumbing every `nagcsu` subcommand uses.

Not part of the public API. Every CLI module in this package imports
from here to load the project config once per invocation and to parse
the repeated `--param NAME=VALUE` option the same way everywhere.
"""

import click

from nagcsu import config as config_module
from nagcsu import parameters
from nagcsu.deck import Deck


def load(ctx: click.Context) -> tuple[config_module.ProjectConfig, Deck]:
    """Load the project config and its base deck for the current invocation.

    :raises click.ClickException: if the config file or the deck it
        points at cannot be found.
    """
    config_path = ctx.obj["config_path"]
    try:
        project_config = config_module.load(config_path)
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    deck_path = project_config.resolved_path(project_config.deck_path)
    try:
        base_deck = Deck.load(deck_path)
    except FileNotFoundError as error:
        raise click.ClickException(f"Deck not found at {deck_path}: {error}") from error

    return project_config, base_deck


def parse_param_options(pairs: tuple[str, ...]) -> dict[str, float]:
    """Parse repeated `--param NAME=VALUE` strings into a state dict.

    :raises click.BadParameter: if a pair is not `NAME=VALUE`, if `NAME`
        is not a registered parameter, or if `VALUE` is not a float.
    """
    state: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"Expected NAME=VALUE, got {pair!r}")
        name, _, raw_value = pair.partition("=")
        name = name.strip()
        if name not in parameters.PARAMETERS:
            valid_names = ", ".join(sorted(parameters.PARAMETERS))
            raise click.BadParameter(f"Unknown parameter {name!r}. Valid names: {valid_names}")
        try:
            state[name] = float(raw_value)
        except ValueError as error:
            raise click.BadParameter(f"{name}={raw_value!r} is not a number") from error
    return state


def echo_outcome_header(run_id: str, j: float | None, prt_is_clean: bool | None) -> None:
    """Print a one-line summary of a run's score and health, consistently."""
    j_text = f"J={j:.4f}" if j is not None else "J=not scored"
    clean_text = "clean" if prt_is_clean else ("NEEDS ATTENTION" if prt_is_clean is not None else "not checked")
    click.echo(f"{run_id}: {j_text}, run health: {clean_text}")

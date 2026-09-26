"""`nagcsu run`: run the deck once, with an optional parameter override."""

import click

from nagcsu import ledger, pipeline
from nagcsu.cli import context


@click.command(name="run")
@click.option(
    "--param",
    "param_pairs",
    multiple=True,
    metavar="NAME=VALUE",
    help="Override one parameter, repeatable. Unset parameters use their default (the deck as shipped).",
)
@click.option(
    "--run-id", default=None, help="Output subdirectory name. Defaults to the next run_NNNN."
)
@click.option("--no-score", is_flag=True, help="Skip scoring against the observed history.")
@click.option("--note", default="", help="Free-text note saved to the run ledger.")
@click.pass_context
def run(
    ctx: click.Context, param_pairs: tuple[str, ...], run_id: str | None, no_score: bool, note: str
) -> None:
    """Run the deck once and log the result to the run ledger.

    With no `--param`, this runs the baseline deck exactly as shipped.
    Every parameter not given a `--param` falls back to its default;
    see `nagcsu match list-parameters` for the full set of names.
    """
    project_config, base_deck = context.load(ctx)
    state = context.parse_param_options(param_pairs)

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    records = ledger.load(ledger_path)
    resolved_run_id = run_id or ledger.new_run_id(records)

    outcome = pipeline.execute_run(
        project_config,
        base_deck,
        state,
        run_id=resolved_run_id,
        score=not no_score,
    )
    record = pipeline.to_run_record(outcome, group=None, strategy=None, note=note)
    ledger.append(ledger_path, record)

    context.echo_outcome_header(record)
    if outcome.simulation_error:
        click.echo(f"  Deck written to {outcome.deck_path}")
        return
    if outcome.prt_report and not outcome.prt_report.is_clean:
        click.echo(f"  See {outcome.prt_report.path} for details.")
    click.echo(f"  Deck written to {outcome.deck_path}")

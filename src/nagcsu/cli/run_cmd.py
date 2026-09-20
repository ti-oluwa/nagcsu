"""`nagcsu run`: run the deck once, with an optional parameter override."""

import click

from nagcsu import ledger, parameters, pipeline
from nagcsu.cli import _context


@click.command(name="run")
@click.option(
    "--param",
    "param_pairs",
    multiple=True,
    metavar="NAME=VALUE",
    help="Override one parameter, repeatable. Unset parameters use their default (the deck as shipped).",
)
@click.option("--run-id", default=None, help="Output subdirectory name. Defaults to the next run_NNNN.")
@click.option("--no-score", is_flag=True, help="Skip scoring against the observed history.")
@click.option("--note", default="", help="Free-text note saved to the run ledger.")
@click.pass_context
def run_cmd(ctx: click.Context, param_pairs: tuple[str, ...], run_id: str | None, no_score: bool, note: str) -> None:
    """Run the deck once and log the result to the run ledger.

    With no `--param`, this runs the baseline deck exactly as shipped.
    Every parameter not given a `--param` falls back to its default;
    see `nagcsu match list-parameters` for the full set of names.
    """
    project_config, base_deck = _context.load(ctx)
    state = _context.parse_param_options(param_pairs)

    ledger_path = project_config.resolved_path(project_config.ledger_path)
    records = ledger.load(ledger_path)
    resolved_run_id = run_id or ledger.new_run_id(records)

    outcome = pipeline.execute_run(project_config, base_deck, state, run_id=resolved_run_id, score=not no_score)

    record = ledger.RunRecord(
        run_id=outcome.run_id,
        created_at=ledger.timestamp_now(),
        parameter_state=outcome.resolved_state,
        group=None,
        strategy=None,
        j=outcome.objective_result.j if outcome.objective_result else None,
        vector_nrmse=(
            {name: vector_score.nrmse for name, vector_score in outcome.objective_result.vector_scores.items()}
            if outcome.objective_result
            else None
        ),
        prt_is_clean=outcome.prt_report.is_clean if outcome.prt_report else None,
        note=note,
    )
    ledger.append(ledger_path, record)

    _context.echo_outcome_header(record.run_id, record.j, record.prt_is_clean)
    if outcome.prt_report and not outcome.prt_report.is_clean:
        click.echo(f"  See {outcome.prt_report.path} for details.")
    click.echo(f"  Deck written to {outcome.deck_path}")

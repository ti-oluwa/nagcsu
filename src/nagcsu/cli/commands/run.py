"""`nagcsu run`: run the deck once, with an optional parameter override."""

import click

from nagcsu import ledger, pipeline
from nagcsu.cli import context, display


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
@context.baseline_options
@context.wells_option
@context.weights_option
@click.pass_context
def run(
    ctx: click.Context,
    param_pairs: tuple[str, ...],
    run_id: str | None,
    no_score: bool,
    note: str,
    baseline_raw: str | None,
    base_deck_raw: str | None,
    wells_raw: str | None,
    weights_raw: str | None,
) -> None:
    """Run the deck once and log the result to the run ledger.

    With no `--param` and no `--baseline`, this runs the deck exactly as shipped.
    With `--baseline best` (or a run ID, or `latest`) it re-runs from that run's
    parameters and deck, and any `--param` overrides are applied on top. Every
    parameter not set by either falls back to its default; see
    `nagcsu match list-parameters` for the full set of names.
    """
    project_config, base_deck = context.load(ctx)
    project_config = context.apply_objective_overrides(
        project_config, wells_raw=wells_raw, weights_raw=weights_raw
    )
    baseline = context.resolve_baseline(
        project_config,
        base_deck,
        baseline_raw=baseline_raw,
        base_deck_raw=base_deck_raw,
    )
    display.print_baseline(baseline)
    state = {**baseline.state, **context.parse_param_options(param_pairs)}

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    records = ledger.load(ledger_path)
    output_root = project_config.get_resolved_path(project_config.output_root)
    on_disk = [path.name for path in output_root.iterdir()] if output_root.is_dir() else []
    resolved_run_id = run_id or ledger.new_run_id(records, taken_ids=on_disk)

    outcome = pipeline.execute(
        project_config,
        baseline.deck,
        state,
        run_id=resolved_run_id,
        score=not no_score,
    )
    record = pipeline.build_run_record(
        outcome,
        group=None,
        strategy=None,
        note=note,
        baseline=baseline.ledger_label(),
        base_deck=baseline.deck_source,
    )
    ledger.append(ledger_path, record)

    context.echo_outcome_header(record)
    if outcome.simulation_error:
        click.echo(f"  Deck written to {outcome.deck_path}")
        return
    if outcome.prt_report and not outcome.prt_report.is_clean:
        click.echo(f"  See {outcome.prt_report.path} for details.")
    click.echo(f"  Deck written to {outcome.deck_path}")

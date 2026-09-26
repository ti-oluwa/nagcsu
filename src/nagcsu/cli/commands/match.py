"""`nagcsu match`: sweep, random-search or auto-tune the deck's parameters."""

import pathlib

import click
import yaml

from nagcsu import constants, ledger, parameters, pipeline, reporting
from nagcsu.algorithms import coordinate_descent, grid, random_search
from nagcsu.cli import context


@click.group(name="match")
def match() -> None:
    """History-matching search commands: sweep, random search and auto-tune."""


@match.command(name="list-parameters")
def list_parameters() -> None:
    """List every tunable parameter, its group, bounds and default."""
    for group in constants.TUNING_PRIORITY_ORDER:
        specs = parameters.parameters_in_group(group)
        if not specs:
            continue
        click.echo(f"\n{group} (priority {constants.TUNING_PRIORITY_ORDER.index(group) + 1})")
        for spec in specs:
            click.echo(
                f"  {spec.name:<40} default={spec.default:<12g} "
                f"bounds=({spec.bounds[0]:g}, {spec.bounds[1]:g})"
            )
            click.echo(f"      {spec.description}")


@match.command(name="sweep")
@click.option(
    "--param", "param_name", required=True, help="Parameter to sweep, e.g. aquifer.radius"
)
@click.option(
    "--values",
    required=True,
    help="Comma-separated values to try, e.g. 12000,14000,16137,18000,20000",
)
@click.option(
    "--group-label",
    default=None,
    help="Group name recorded on each ledger entry. Defaults to the parameter's own group.",
)
@click.pass_context
def sweep(ctx: click.Context, param_name: str, values: str, group_label: str | None) -> None:
    """Run every value of one parameter, holding all others at their default.

    The direct CLI equivalent of Stage D.4's sweep helper: pick one
    parameter, list the values to try, and get back the J for each.
    """
    if param_name not in parameters.PARAMETERS:
        raise click.BadParameter(
            f"Unknown parameter {param_name!r}. Run `nagcsu match list-parameters` to see valid names."
        )

    project_config, base_deck = context.load(ctx)
    parsed_values = [float(value) for value in values.split(",")]

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        record = pipeline.to_run_record(
            outcome,
            group=group_label or parameters.PARAMETERS[param_name].group,
            strategy="sweep",
            note=f"sweep of {param_name}",
        )
        ledger.append(ledger_path, record)
        context.echo_outcome_header(record)

    evaluate = pipeline.make_evaluate(
        project_config, base_deck, run_id_prefix="sweep", on_outcome=on_outcome
    )
    result = grid.search(parameters.default_state(), {param_name: parsed_values}, evaluate)
    context.warn_if_every_trial_failed(result.best.j, command="match sweep")
    click.echo(f"\nBest: {param_name}={result.best.state[param_name]:g}, J={result.best.j:.4f}")


@match.command(name="random")
@click.option(
    "--param",
    "param_names",
    multiple=True,
    required=True,
    help="Parameter to randomize, repeatable.",
)
@click.option("--trials", default=20, show_default=True, help="Number of random trials.")
@click.option("--seed", default=None, type=int, help="Random seed, for reproducible trials.")
@click.pass_context
def random_command(
    ctx: click.Context, param_names: tuple[str, ...], trials: int, seed: int | None
) -> None:
    """Randomly sample one or more parameters within their bounds."""
    unknown = [name for name in param_names if name not in parameters.PARAMETERS]
    if unknown:
        raise click.BadParameter(
            f"Unknown parameter(s): {unknown}. Run `nagcsu match list-parameters` to see valid names."
        )

    project_config, base_deck = context.load(ctx)
    bounds_by_parameter = {name: parameters.PARAMETERS[name].bounds for name in param_names}

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        record = pipeline.to_run_record(
            outcome,
            group=None,
            strategy="random",
            note=f"random search over {list(param_names)}",
        )
        ledger.append(ledger_path, record)
        context.echo_outcome_header(record)

    evaluate = pipeline.make_evaluate(
        project_config, base_deck, run_id_prefix="random", on_outcome=on_outcome
    )
    result = random_search.search(
        parameters.default_state(), bounds_by_parameter, evaluate, num_trials=trials, seed=seed
    )
    context.warn_if_every_trial_failed(result.best.j, command="match random")
    click.echo(f"\nBest J={result.best.j:.4f} at:")
    for name in param_names:
        click.echo(f"  {name} = {result.best.state[name]:g}")


@match.command(name="auto")
@click.option(
    "--target-j",
    default=None,
    type=float,
    help="Stop once J reaches this value. Defaults to the project config's objective.target_j.",
)
@click.option(
    "--groups",
    default=None,
    help="Comma-separated subset of tuning groups to run, in the order given. Defaults to the full Stage D.1 priority order.",
)
@click.option(
    "--passes-per-group",
    default=2,
    show_default=True,
    help="Optimization passes through each group's parameters.",
)
@click.option(
    "--report",
    "report_path",
    default=None,
    help="Write a Markdown summary report to this path once tuning stops.",
)
@click.pass_context
def auto(
    ctx: click.Context,
    target_j: float | None,
    groups: str | None,
    passes_per_group: int,
    report_path: str | None,
) -> None:
    """Auto-tune one parameter group at a time until J reaches its target.

    Follows Stage D.1's priority order (aquifer, then permeability
    multiplier, then SGOF shape, and so on) and Stage D.3's rule of
    changing one group at a time, stopping the moment J is at or below
    the target rather than continuing to chase a smaller number (Stage
    C.4 / D.5). Every trial is logged to the run ledger; the final
    state is written out as its own deck under the winning run's output
    directory, alongside a `parameters.yaml` snapshot of exactly the
    state that produced it.
    """
    project_config, base_deck = context.load(ctx)
    resolved_target_j = target_j if target_j is not None else project_config.objective.target_j
    groups_in_order = groups.split(",") if groups else list(constants.TUNING_PRIORITY_ORDER)

    bounds_by_group = {
        group: {spec.name: spec.bounds for spec in parameters.parameters_in_group(group)}
        for group in groups_in_order
    }

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    failed_trial_count = 0

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        nonlocal failed_trial_count
        if outcome.simulation_error:
            failed_trial_count += 1
        record = pipeline.to_run_record(
            outcome, group=None, strategy="coordinate_descent", note="auto-tune trial"
        )
        ledger.append(ledger_path, record)

    evaluate = pipeline.make_evaluate(
        project_config, base_deck, run_id_prefix="auto", on_outcome=on_outcome
    )
    result, outcomes = coordinate_descent.search(
        parameters.default_state(),
        groups_in_order,
        bounds_by_group,
        evaluate,
        target_j=resolved_target_j,
        passes_per_group=passes_per_group,
    )
    context.warn_if_every_trial_failed(result.best.j, command="match auto")

    click.echo(f"Ran {len(result.trials)} trials across {len(outcomes)} group(s).")
    if failed_trial_count:
        click.echo(f"  ({failed_trial_count} trial(s) failed to simulate and were skipped)")
    for outcome_summary in outcomes:
        click.echo(
            f"  {outcome_summary.group}: J {outcome_summary.starting_j:.4f} -> "
            f"{outcome_summary.ending_j:.4f} (target reached: {outcome_summary.reached_target})"
        )
    click.echo(f"\nBest J={result.best.j:.4f}")

    final_run_id = "auto_final"
    final_outcome = pipeline.execute_run(
        project_config, base_deck, result.best.state, run_id=final_run_id, score=True
    )
    final_record = pipeline.to_run_record(
        final_outcome,
        group=groups_in_order[-1] if groups_in_order else None,
        strategy="coordinate_descent",
        note=f"auto-tune final state after {len(result.trials)} trials across {[o.group for o in outcomes]}",
    )
    ledger.append(ledger_path, final_record)

    if final_outcome.simulation_error:
        raise click.ClickException(
            f"The best trial found during the search ran fine, but re-running its exact "
            f"state for the final deck failed: {final_outcome.simulation_error}. This should "
            f"not normally happen; the deck at {final_outcome.deck_path} is worth inspecting "
            f"by hand."
        )

    parameters_snapshot_path = final_outcome.output_dir / "parameters.yaml"
    write_parameters_snapshot(final_outcome.resolved_state, parameters_snapshot_path)
    click.echo(f"Final calibrated deck: {final_outcome.deck_path}")
    click.echo(f"Parameter snapshot: {parameters_snapshot_path}")

    if report_path:
        written = reporting.write_run_report(
            final_record,
            report_path,
            prt_report=final_outcome.prt_report,
            group_outcomes=outcomes,
        )
        click.echo(f"Report: {written}")


def write_parameters_snapshot(state: dict[str, float], path: pathlib.Path) -> None:
    """Write a resolved parameter state out as a small standalone YAML file.

    This is the practical stand-in for "write the calibrated value to an
    include file": the shipped deck is monolithic rather than split into
    `INCLUDE` files (see `docs/ARCHITECTURE.md`), so there is no single
    `.inc` file to write a value into. This snapshot is what a later
    refactor into `INCLUDE` files would read from.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(state, sort_keys=True), encoding="utf-8")

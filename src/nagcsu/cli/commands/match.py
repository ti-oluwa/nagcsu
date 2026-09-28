"""`nagcsu match`: sweep, random-search or auto-tune the deck's parameters."""

import dataclasses
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
        specs = parameters.get_parameters_in_group(group)
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

    Pick one parameter, list the values to try, and get back the J for each.
    """
    if param_name not in parameters.PARAMETERS:
        raise click.BadParameter(
            f"Unknown parameter {param_name!r}. Run `nagcsu match list-parameters` to see valid names."
        )

    project_config, base_deck = context.load(ctx)
    parsed_values = [float(value) for value in values.split(",")]

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        record = pipeline.build_run_record(
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
    result = grid.search(
        parameters.default_state(), {param_name: parsed_values}, evaluate
    )  # TODO: Expose max_evaluations
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
def random_(
    ctx: click.Context, param_names: tuple[str, ...], trials: int, seed: int | None
) -> None:
    """Randomly sample one or more parameters within their bounds."""
    unknown = [name for name in param_names if name not in parameters.PARAMETERS]
    if unknown:
        raise click.BadParameter(
            f"Unknown parameter(s): {unknown}. Run `nagcsu match list-parameters` to see valid names."
        )

    project_config, base_deck = context.load(ctx)
    parameter_bounds = {name: parameters.PARAMETERS[name].bounds for name in param_names}

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        record = pipeline.build_run_record(
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
        parameters.default_state(),
        parameter_bounds,
        evaluate,
        n_trials=trials,
        seed=seed,
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
    help="Comma-separated subset of tuning groups to run, in the order given. Defaults to the full priority order.",
)
@click.option(
    "--passes-per-group",
    default=2,
    show_default=True,
    help="Optimization passes through each group's parameters.",
)
@click.option(
    "--report",
    "report_path",  # TODO: Use proper type here. I think click support pathlib.Path and other path validation options
    default=None,
    help="Write a Markdown summary report to this path once tuning stops.",
)
@click.option(
    "--xatol-fraction",
    default=coordinate_descent.DEFAULT_XATOL_FRACTION,
    show_default=True,
    help="How precisely each parameter is located, as a fraction of its bound range.",
)
@click.option(
    "--max-evals-per-parameter",
    default=coordinate_descent.DEFAULT_MAX_EVALUATIONS_PER_PARAMETER,
    show_default=True,
    help="Most simulations spent on one parameter in one pass.",
)
@click.option(
    "--min-improvement",
    default=coordinate_descent.DEFAULT_MIN_RELATIVE_IMPROVEMENT,
    show_default=True,
    help="Move on to the next group when a full pass improves J by less than this fraction.",
)
@click.option(
    "--starts",
    default=1,
    show_default=True,
    help=(
        "Independent coordinate-descent runs, the first from the default state, "
        "the rest from randomized starting points within bounds; the best is kept. "
        "Cheap protection against a single run converging to a poor local optimum "
        "near a threshold-like nonlinearity (see `coordinate_descent.multi_start_search`)."
    ),
)
@click.option(
    "--weights",
    default=None,
    help=(
        "Override objective weights for this run only, e.g. 'pressure=1,watercut=0,gor=0' "
        "to match pressure and water cut first (Phase A) before re-enabling GOR to tune "
        "SGOF (Phase B). Vectors not listed keep their nagcsu.yaml weight; nagcsu.yaml "
        "itself is never modified."
    ),
)
@click.pass_context
def auto(
    ctx: click.Context,
    target_j: float | None,
    groups: str | None,
    passes_per_group: int,
    report_path: str | None,
    xatol_fraction: float,
    max_evals_per_parameter: int,
    min_improvement: float,
    starts: int,
    weights: str | None,
) -> None:
    """Auto-tune one parameter group at a time until J reaches its target.

    Works through the groups in priority order (aquifer, then permeability
    multiplier, then SGOF shape, and so on), changing one parameter at a
    time and stopping the moment J is at or below the target rather than
    continuing to chase a smaller number. A group whose parameters barely
    move J is abandoned after one pass so the search reaches the groups
    that do matter. Every trial is logged to the run ledger; the final
    state is written out as its own deck under the winning run's output
    directory, alongside a `parameters.yaml` snapshot of exactly the
    state that produced it.
    """
    project_config, base_deck = context.load(ctx)
    resolved_target_j = target_j if target_j is not None else project_config.objective.target_j
    groups_in_order = groups.split(",") if groups else list(constants.TUNING_PRIORITY_ORDER)

    if weights:
        overrides = parse_weight_overrides(weights)
        merged_weights = {**project_config.objective.weights, **overrides}
        project_config = dataclasses.replace(
            project_config,
            objective=dataclasses.replace(project_config.objective, weights=merged_weights),
        )

    bounds_by_group = {
        group: {spec.name: spec.bounds for spec in parameters.get_parameters_in_group(group)}
        for group in groups_in_order
    }

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    failed_trial_count = 0

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        nonlocal failed_trial_count
        if outcome.simulation_error:
            failed_trial_count += 1
        record = pipeline.build_run_record(
            outcome,
            group=None,
            strategy="coordinate_descent",
            note="auto-tune trial",
        )
        ledger.append(ledger_path, record)

    evaluate = pipeline.make_evaluate(
        project_config, base_deck, run_id_prefix="auto", on_outcome=on_outcome
    )
    result, outcomes = coordinate_descent.multi_start_search(
        parameters.default_state(),
        groups_in_order,
        bounds_by_group,
        evaluate,
        target_j=resolved_target_j,
        passes_per_group=passes_per_group,
        xatol_fraction=xatol_fraction,
        max_evaluations_per_parameter=max_evals_per_parameter,
        min_relative_improvement=min_improvement,
        n_starts=starts,
    )
    context.warn_if_every_trial_failed(result.best.j, command="match auto")

    click.echo(f"Ran {len(result.trials)} trials across {len(outcomes)} group(s).")
    if failed_trial_count:
        click.echo(f"  ({failed_trial_count} trial(s) failed to simulate and were skipped)")

    for outcome in outcomes:
        click.echo(
            f"  {outcome.group}: J {outcome.starting_j:.4f} -> "
            f"{outcome.ending_j:.4f} (target reached: {outcome.reached_target})"
        )
    click.echo(f"\nBest J={result.best.j:.4f}")

    final_run_id = "auto_final"
    final_outcome = pipeline.execute(
        project_config,
        base_deck,
        result.best.state,
        run_id=final_run_id,
        score=True,
    )
    final_record = pipeline.build_run_record(
        final_outcome,
        group=groups_in_order[-1] if groups_in_order else None,
        strategy="coordinate_descent",
        note=f"auto-tune final state after {len(result.trials)} trials across {[outcome.group for outcome in outcomes]}",
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


def parse_weight_overrides(raw: str) -> dict[str, float]:
    """Parse a `--weights` option value into `{vector_name: weight}`.

    :param raw: Comma-separated `name=value` pairs, for example
        `"pressure=1,watercut=0,gor=0"`.
    :raises click.BadParameter: if a pair is malformed or a weight is
        not a valid float.
    """
    overrides: dict[str, float] = {}
    for pair in raw.split(","):
        name, _, value = pair.partition("=")
        name = name.strip()
        if not name or not value:
            raise click.BadParameter(
                f"Malformed weight override {pair!r}; expected 'name=value', e.g. 'gor=0'."
            )
        try:
            overrides[name] = float(value)
        except ValueError as error:
            raise click.BadParameter(f"Weight for {name!r} is not a number: {value!r}") from error
    return overrides


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

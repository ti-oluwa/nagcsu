"""`nagcsu match`: sweep, random-search or auto-tune the deck's parameters."""

import pathlib
import time
import typing

import click
import yaml

from nagcsu import constants, glossary, ledger, parameters, pipeline, reporting
from nagcsu.algorithms import coordinate_descent, grid, random_search, sensitivity
from nagcsu.algorithms.base import get_current_tag
from nagcsu.cli import context, display


@click.group(name="match")
def match() -> None:
    """History-matching search commands: sweep, random search and auto-tune."""


@match.command(name="list-parameters")
def list_parameters() -> None:
    """List every tunable parameter, its group, bounds and default."""
    display.console.print(display.parameters_table())


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
@click.option(
    "--run-id-prefix",
    default="sweep",
    show_default=True,
    help="Prefix for the generated run IDs in this sweep, e.g. sweep_00000.",
)
@click.option(
    "--max-evaluations",
    default=grid.MAX_EVALUATIONS_DEFAULT,
    show_default=True,
    help="Refuse to run more than this many values without raising the limit explicitly.",
)
@context.baseline_options
@context.wells_option
@context.weights_option
@click.pass_context
def sweep(
    ctx: click.Context,
    param_name: str,
    values: str,
    group_label: str | None,
    run_id_prefix: str,
    max_evaluations: int,
    baseline_raw: str | None,
    base_deck_raw: str | None,
    wells_raw: str | None,
    weights_raw: str | None,
) -> None:
    """Run every value of one parameter, holding all others at their default.

    Pick one parameter, list the values to try, and get back the J for each.
    """
    if param_name not in parameters.PARAMETERS:
        raise click.BadParameter(
            f"Unknown parameter {param_name!r}. Run `nagcsu match list-parameters` to see valid names."
        )

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
    parsed_values = [float(value) for value in values.split(",")]
    check_values_against_limits(param_name, parsed_values)
    ledger_path = project_config.get_resolved_path(project_config.ledger_path)

    sweep_records: list[ledger.RunRecord] = []

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        record = pipeline.build_run_record(
            outcome,
            group=group_label or parameters.PARAMETERS[param_name].group,
            strategy="sweep",
            note=f"sweep of {param_name}",
            tuned_parameters=[param_name],
            stage="sweep",
            baseline=baseline.ledger_label(),
            base_deck=baseline.deck_source,
        )
        ledger.append(ledger_path, record)
        sweep_records.append(record)
        display.print_outcome_line(record)

    evaluate = pipeline.make_evaluate(
        project_config,
        baseline.deck,
        run_id_prefix=run_id_prefix,
        on_outcome=on_outcome,
    )
    result = grid.search(
        baseline.state,
        {param_name: parsed_values},
        evaluate,
        max_evaluations=max_evaluations,
    )
    context.warn_if_every_trial_failed(result.best.j, command="match sweep")
    display.console.print(
        display.trials_table(
            sweep_records, title=f"Sweep of {param_name}", value_parameter=param_name
        )
    )
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
@click.option(
    "--run-id-prefix",
    default="random",
    show_default=True,
    help="Prefix for the generated run IDs in this random search, e.g. random_00000.",
)
@click.option("--seed", default=None, type=int, help="Random seed, for reproducible trials.")
@click.option(
    "--range",
    "range_options",
    multiple=True,
    help=(
        "Sampling range for one --param as NAME=LOW:HIGH, repeatable. May go past the "
        "registered bounds (limited only by hard physical limits)."
    ),
)
@context.baseline_options
@context.wells_option
@context.weights_option
@click.pass_context
def random_(
    ctx: click.Context,
    param_names: tuple[str, ...],
    trials: int,
    run_id_prefix: str,
    seed: int | None,
    range_options: tuple[str, ...],
    baseline_raw: str | None,
    base_deck_raw: str | None,
    wells_raw: str | None,
    weights_raw: str | None,
) -> None:
    """Randomly sample one or more parameters within their bounds."""
    unknown_params = [name for name in param_names if name not in parameters.PARAMETERS]
    if unknown_params:
        raise click.BadParameter(
            f"Unknown parameter(s): {unknown_params}. Run `nagcsu match list-parameters` to see valid names."
        )

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
    sampling_ranges = parse_range_options(range_options)
    stray = sorted(set(sampling_ranges) - set(param_names))
    if stray:
        raise click.BadParameter(f"--range given for {stray} but they are not in --param.")

    parameter_bounds = {
        name: sampling_ranges.get(name, parameters.PARAMETERS[name].bounds) for name in param_names
    }
    ledger_path = project_config.get_resolved_path(project_config.ledger_path)

    involved_groups = sorted({parameters.PARAMETERS[name].group for name in param_names})
    random_records: list[ledger.RunRecord] = []

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        record = pipeline.build_run_record(
            outcome,
            group=", ".join(involved_groups),
            strategy="random",
            note=f"random search over {list(param_names)}",
            tuned_parameters=list(param_names),
            stage="random",
            baseline=baseline.ledger_label(),
            base_deck=baseline.deck_source,
        )
        ledger.append(ledger_path, record)
        random_records.append(record)
        display.print_outcome_line(record)

    evaluate = pipeline.make_evaluate(
        project_config,
        baseline.deck,
        run_id_prefix=run_id_prefix,
        on_outcome=on_outcome,
    )
    result = random_search.search(
        baseline.state,
        parameter_bounds,
        evaluate,
        n_trials=trials,
        seed=seed,
    )
    context.warn_if_every_trial_failed(result.best.j, command="match random")
    ranked = sorted(random_records, key=lambda r: (r.j is None, r.j if r.j is not None else 0.0))
    display.console.print(
        display.trials_table(ranked[:10], title="Ten best random trials (values in ledger)")
    )
    display.print_key(("J", "NRMSE", "Health", "*"))
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
    "--run-id-prefix",
    default="auto",
    show_default=True,
    help="Prefix for the generated run IDs during the optimization, e.g. auto_00000.",
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
    "report_path",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
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
    "--param",
    "param_names",
    multiple=True,
    help=(
        "Tune only this parameter, repeatable (e.g. --param aquifer.radius --param "
        "sgof.sorg). Groups are derived from the parameters and keep priority order. "
        "Run `nagcsu match list-parameters` for valid names."
    ),
)
@click.option(
    "--range",
    "range_options",
    multiple=True,
    help=(
        "Search range for one parameter as NAME=LOW:HIGH, repeatable, e.g. "
        "--range aquifer.radius=12000:20000. Must sit inside the parameter's registered "
        "bounds. A parameter given only here is tuned even without --param."
    ),
)
@click.option(
    "--start",
    "start_options",
    multiple=True,
    help=(
        "Start the search from this value instead of the default, as NAME=VALUE, "
        "repeatable. Use it to continue from a value found earlier."
    ),
)
@click.option(
    "--order",
    type=click.Choice(["priority", "sensitivity"]),
    default="priority",
    show_default=True,
    help=(
        "priority: fixed group order from the methodology. sensitivity: first run a "
        "one-at-a-time screen, then tune groups from most to least sensitive, most "
        "sensitive parameter first, and skip parameters below --min-relative-swing. "
        "Costs 2 simulations per parameter up front and usually saves more than that."
    ),
)
@click.option(
    "--group-rank",
    "group_rank_method",
    type=click.Choice(sensitivity.GROUP_RANK_METHODS),
    default="mean_rank",
    show_default=True,
    help="How groups are ordered when '--order sensitivity' is used.",
)
@click.option(
    "--min-relative-swing",
    default=0.02,
    show_default=True,
    help="With '--order sensitivity', skip parameters whose swing is below this fraction of the largest swing.",
)
@click.option(
    "--perturbation-fraction",
    default=0.15,
    show_default=True,
    help="With '--order sensitivity', fraction of each parameter's range perturbed in each direction.",
)
@click.option(
    "--window-shrink",
    default=coordinate_descent.DEFAULT_WINDOW_SHRINK,
    show_default=True,
    help="Each pass after the first searches a window this fraction as wide as the last, around the value found so far. 1 keeps full-range passes.",
)
@click.option("--quiet", is_flag=True, default=False, help="Do not print a line per trial.")
@context.baseline_options
@context.wells_option
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
    run_id_prefix: str,
    groups: str | None,
    passes_per_group: int,
    report_path: pathlib.Path | None,
    xatol_fraction: float,
    max_evals_per_parameter: int,
    min_improvement: float,
    starts: int,
    param_names: tuple[str, ...],
    range_options: tuple[str, ...],
    start_options: tuple[str, ...],
    order: str,
    group_rank_method: str,
    min_relative_swing: float,
    perturbation_fraction: float,
    window_shrink: float,
    quiet: bool,
    baseline_raw: str | None,
    base_deck_raw: str | None,
    wells_raw: str | None,
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
    started_at = time.perf_counter()
    project_config, base_deck = context.load(ctx)
    resolved_target_j = target_j if target_j is not None else project_config.objective.target_j

    project_config = context.apply_objective_overrides(
        project_config, wells_raw=wells_raw, weights_raw=weights
    )

    ranges = parse_range_options(range_options)
    start_overrides = parse_start_options(start_options)
    ordered_groups, group_parameter_bounds = resolve_tuning_space(
        groups=groups.split(",") if groups else None,
        param_names=param_names,
        ranges=ranges,
    )
    baseline = context.resolve_baseline(
        project_config,
        base_deck,
        baseline_raw=baseline_raw,
        base_deck_raw=base_deck_raw,
    )
    display.print_baseline(baseline)
    start_state = dict(baseline.state)
    start_state.update(start_overrides)
    group_parameter_bounds = widen_bounds_for_start_values(
        group_parameter_bounds, start_state, explicit_ranges=ranges
    )

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    failed_trial_count = 0

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        nonlocal failed_trial_count
        if outcome.simulation_error:
            failed_trial_count += 1

        tag = get_current_tag()
        is_screen = tag is not None and (tag.stage or "").startswith("sensitivity")
        record = pipeline.build_run_record(
            outcome,
            group=None,
            strategy="sensitivity" if is_screen else "coordinate_descent",
            note=f"auto-tune {tag.stage}" if tag and tag.stage else "auto-tune trial",
            baseline=baseline.ledger_label(),
            base_deck=baseline.deck_source,
        )
        ledger.append(ledger_path, record)
        if not quiet:
            display.print_outcome_line(record)

    evaluate = pipeline.make_evaluate(
        project_config,
        baseline.deck,
        run_id_prefix=run_id_prefix,
        on_outcome=on_outcome,
    )

    screen_results: list[sensitivity.SensitivityResult] = []
    ranked_groups: list[sensitivity.GroupSensitivity] = []
    if order == "sensitivity":
        flat_bounds = {
            name: bounds
            for group in group_parameter_bounds.values()
            for name, bounds in group.items()
        }
        parameter_groups = {
            name: group for group, members in group_parameter_bounds.items() for name in members
        }
        click.echo(f"Screening {len(flat_bounds)} parameter(s) for sensitivity first...")
        screen_results, _ = sensitivity.run(
            start_state,
            flat_bounds,
            evaluate,
            perturbation_fraction=perturbation_fraction,
            parameter_groups=parameter_groups,
        )
        context.warn_if_every_trial_failed(
            min((result.base_j for result in screen_results), default=float("inf")),
            command="match auto (sensitivity screen)",
        )
        swings = {result.parameter: result.swing for result in screen_results}
        ranked_groups = sensitivity.rank_groups(swings, parameter_groups, method=group_rank_method)
        plan = sensitivity.build_tuning_plan(
            ranked_groups, swings, min_relative_swing=min_relative_swing
        )
        if not plan:
            raise click.ClickException("The sensitivity screen left no parameter worth tuning.")

        ranks = sensitivity.rank_parameters(swings)
        display.console.print(display.sensitivity_table(screen_results, parameter_groups, ranks))
        display.console.print(
            display.group_sensitivity_table(ranked_groups, method=group_rank_method)
        )
        skipped = sorted(set(flat_bounds) - {name for _, names in plan for name in names})
        if skipped:
            click.echo(
                f"Skipped as insensitive (< {min_relative_swing:g} of largest swing): {', '.join(skipped)}"
            )
        ordered_groups = [group for group, _ in plan]
        group_parameter_bounds = {
            group: {name: flat_bounds[name] for name in names} for group, names in plan
        }
        click.echo(f"Tuning order: {' > '.join(ordered_groups)}")

    result, outcomes = coordinate_descent.multi_start_search(
        start_state,
        ordered_groups,
        group_parameter_bounds,
        evaluate,
        target_j=resolved_target_j,
        passes_per_group=passes_per_group,
        xatol_fraction=xatol_fraction,
        max_evaluations_per_parameter=max_evals_per_parameter,
        min_relative_improvement=min_improvement,
        window_shrink=window_shrink,
        n_starts=starts,
    )
    context.warn_if_every_trial_failed(result.best.j, command="match auto")

    click.echo(f"Ran {len(result.trials)} trials across {len(outcomes)} group(s).")
    if failed_trial_count:
        click.echo(f"  ({failed_trial_count} trial(s) failed to simulate and were skipped)")

    display.console.print(display.group_outcomes_table(outcomes))
    display.console.print(display.parameter_steps_table(outcomes))
    click.echo(f"\nBest J={result.best.j:.4f}")

    changed = [
        name
        for name, value in result.best.state.items()
        if abs(value - start_state.get(name, value)) > 1e-12 and name in parameters.PARAMETERS
    ]
    final_run_id = unique_run_id(
        f"{run_id_prefix}_final",
        [record.run_id for record in ledger.load(ledger_path)],
        output_root=project_config.get_resolved_path(project_config.output_root),
    )
    final_outcome = pipeline.execute(
        project_config,
        baseline.deck,
        result.best.state,
        run_id=final_run_id,
        score=True,
    )
    final_record = pipeline.build_run_record(
        final_outcome,
        group=", ".join(sorted({parameters.PARAMETERS[name].group for name in changed})) or None,
        strategy="coordinate_descent",
        note=f"auto-tune final state after {len(result.trials)} trials across {[outcome.group for outcome in outcomes]}",
        tuned_parameters=changed,
        stage="final",
        baseline=baseline.ledger_label(),
        base_deck=baseline.deck_source,
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

    all_records = ledger.load(ledger_path)
    starting_j = result.trials[0].j if result.trials else None
    if final_record.j is not None:
        display.console.print(display.objective_table(final_record, target_j=resolved_target_j))
    display.console.print(
        display.state_table(
            final_record.parameter_state, title="Parameters changed", show_all=False
        )
    )
    display.print_key(
        glossary.SENSITIVITY if screen_results else (),
        glossary.GROUP_RANKING if ranked_groups else (),
        glossary.TUNING_PATH,
        glossary.OBJECTIVE,
        glossary.STATE,
    )
    summary = f"{len(result.trials)} trials in {time.perf_counter() - started_at:.0f}s"
    if starting_j and final_record.j is not None and starting_j != float("inf"):
        summary += f", J {starting_j:.4f} -> {final_record.j:.4f} ({(starting_j - final_record.j) / starting_j * 100:.1f}% better)"
    click.echo(summary)
    for step in reporting.next_steps(all_records, target_j=resolved_target_j):
        click.echo(f"- {step}")

    click.echo(f"Final calibrated deck: {final_outcome.deck_path}")
    click.echo(f"Parameter snapshot: {parameters_snapshot_path}")

    if report_path:
        written = reporting.write_run_report(
            final_record,
            report_path,
            prt_report=final_outcome.prt_report,
            group_outcomes=outcomes,
            sensitivity_results=screen_results or None,
            group_sensitivities=ranked_groups or None,
            records=all_records,
            target_j=resolved_target_j,
        )
        click.echo(f"Report: {written}")


def parse_range_options(raw: typing.Sequence[str]) -> dict[str, tuple[float, float]]:
    """Parse repeated `--range NAME=LOW:HIGH` values.

    A range may extend past the parameter's recommended `bounds`; it is
    only required to stay inside the hard physical limits (see
    `nagcsu.parameters.PHYSICAL_LIMITS`). A note is printed when it does
    go past the recommended bounds.

    :raises click.BadParameter: for malformed text, an unknown parameter,
        `low >= high`, or a range outside the physical limits.
    """
    ranges: dict[str, tuple[float, float]] = {}
    for item in raw:
        name, _, span = item.partition("=")
        low_text, _, high_text = span.partition(":")
        if name not in parameters.PARAMETERS or not low_text or not high_text:
            raise click.BadParameter(
                f"Malformed or unknown --range {item!r}; expected NAME=LOW:HIGH with a name "
                f"from `nagcsu match list-parameters`."
            )

        try:
            low, high = float(low_text), float(high_text)
        except ValueError as error:
            raise click.BadParameter(f"--range {item!r} has a non-numeric bound.") from error

        limit_low, limit_high = parameters.get_physical_limits(name)
        if low >= high or low < limit_low or high > limit_high:
            raise click.BadParameter(
                f"--range for {name} must satisfy {limit_low:g} <= LOW < HIGH <= {limit_high:g} "
                f"(the physical limits); got {low:g}:{high:g}."
            )

        bound_low, bound_high = parameters.PARAMETERS[name].bounds
        if low < bound_low or high > bound_high:
            click.echo(
                f"Note: --range {name}={low:g}:{high:g} goes past the recommended bounds "
                f"{bound_low:g} to {bound_high:g}; allowed.",
                err=True,
            )
        ranges[name] = (low, high)
    return ranges


def parse_start_options(raw: typing.Sequence[str]) -> dict[str, float]:
    """Parse repeated `--start NAME=VALUE` values, checking physical limits only."""
    starts: dict[str, float] = {}
    for item in raw:
        name, _, value_text = item.partition("=")
        if name not in parameters.PARAMETERS or not value_text:
            raise click.BadParameter(
                f"Malformed or unknown --start {item!r}; expected NAME=VALUE."
            )
        try:
            value = float(value_text)
        except ValueError as error:
            raise click.BadParameter(f"--start {item!r} is not a number.") from error
        limit_low, limit_high = parameters.get_physical_limits(name)
        if not limit_low <= value <= limit_high:
            raise click.BadParameter(
                f"--start for {name} must be within the physical limits {limit_low:g} to {limit_high:g}."
            )
        starts[name] = value
    return starts


def check_values_against_limits(name: str, values: typing.Sequence[float]) -> None:
    """Reject sweep values outside the physical limits and note those past the bounds.

    :raises click.BadParameter: if any value is outside the physical limits.
    """
    limit_low, limit_high = parameters.get_physical_limits(name)
    invalid = [value for value in values if not limit_low <= value <= limit_high]
    if invalid:
        raise click.BadParameter(
            f"{name} values {invalid} are outside the physical limits {limit_low:g} to {limit_high:g}."
        )
    beyond = [value for value in values if parameters.get_bound_violation(name, value)]
    if beyond:
        low, high = parameters.PARAMETERS[name].bounds
        click.echo(
            f"Note: {name} values {beyond} are outside the recommended bounds {low:g} to "
            f"{high:g}; running them anyway.",
            err=True,
        )


def widen_bounds_for_start_values(
    group_parameter_bounds: dict[str, dict[str, tuple[float, float]]],
    start_state: dict[str, float],
    *,
    explicit_ranges: dict[str, tuple[float, float]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Make sure each search range contains the state the search starts from.

    A baseline taken from an earlier run may sit outside the recommended
    bounds (a range that was deliberately extended). Without this, the
    search would be confined to a range that excludes its own starting
    point. Ranges the user gave explicitly are left exactly as given.
    """
    widened: dict[str, dict[str, tuple[float, float]]] = {}
    for group, members in group_parameter_bounds.items():
        widened[group] = {
            name: bounds
            if name in explicit_ranges
            else context.widen_bounds_to_include(bounds, start_state[name])
            for name, bounds in members.items()
        }
    return widened


def unique_run_id(base: str, taken: typing.Iterable[str], *, output_root: pathlib.Path) -> str:
    """`base`, or `base2`, `base3`... so a later session never overwrites an earlier one.

    A name counts as taken if it is in `taken` or already a directory under `output_root`.
    """
    used = set(taken)
    candidate, counter = base, 1
    while candidate in used or (output_root / candidate).exists():
        counter += 1
        candidate = f"{base}{counter}"
    return candidate


def resolve_tuning_space(
    *,
    groups: list[str] | None,
    param_names: typing.Sequence[str],
    ranges: dict[str, tuple[float, float]],
) -> tuple[list[str], dict[str, dict[str, tuple[float, float]]]]:
    """Work out which groups and parameters `match auto` may tune, and over what range.

    :param groups: Groups to consider, in the order given, or `None` for
        the full priority order.
    :param param_names: Parameters to restrict tuning to; empty means no
        restriction, apart from any parameter named in `ranges`.
    :param ranges: Per-parameter search range overrides.
    :returns: `(ordered_groups, {group: {parameter: (low, high)}})`, with
        groups that end up empty removed.
    :raises click.BadParameter: for unknown groups or parameters.
    :raises click.UsageError: when the filters leave nothing to tune.
    """
    ordered = groups if groups else list(constants.GROUP_TUNING_PRIORITY_ORDER)
    unknown_groups = [g for g in ordered if g not in constants.GROUP_TUNING_PRIORITY_ORDER]
    if unknown_groups:
        raise click.BadParameter(
            f"Unknown group(s) {unknown_groups}. Valid groups: {list(constants.GROUP_TUNING_PRIORITY_ORDER)}"
        )
    unknown = [name for name in param_names if name not in parameters.PARAMETERS]
    if unknown:
        raise click.BadParameter(
            f"Unknown parameter(s): {unknown}. Run `nagcsu match list-parameters` to see valid names."
        )
    requested = set(param_names) | set(ranges)
    space: dict[str, dict[str, tuple[float, float]]] = {}
    for group in ordered:
        members = {
            spec.name: ranges.get(spec.name, spec.bounds)
            for spec in parameters.get_parameters_in_group(group)
            if not requested or spec.name in requested
        }
        if members:
            space[group] = members
    if not space:
        raise click.UsageError("The --groups, --param and --range filters leave nothing to tune.")
    return list(space), space


def parse_weight_overrides(raw: str) -> dict[str, float]:
    """Parse a `--weights` option value; see `nagcsu.cli.context.parse_weight_overrides`."""
    return context.parse_weight_overrides(raw)


def write_parameters_snapshot(state: dict[str, float], path: pathlib.Path) -> None:
    """Write a resolved parameter state out as a small standalone YAML file.

    This is the practical stand-in for "write the calibrated value to an
    include file": the shipped deck is monolithic rather than split into
    `INCLUDE` files (see `docs/ARCHITECTURE.md`), so there is no single
    `.inc` file to write a value into. This snapshot is what a later
    refactor into `INCLUDE` files would read from.
    """
    # Cast every value to a native float: PyYAML's SafeDumper cannot
    # serialize a numpy scalar (numpy.float64 and friends), and a
    # search strategy built on scipy is a plausible source of one even
    # after `coordinate_descent`'s own fix, so this stays defensive
    # rather than trusting every caller to have already converted.
    normalized_state = {name: float(value) for name, value in state.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(normalized_state, sort_keys=True), encoding="utf-8")

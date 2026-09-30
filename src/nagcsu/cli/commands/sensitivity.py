"""`nagcsu sensitivity`: rank parameters by local effect on J."""

import typing

import click

from nagcsu import constants, glossary, ledger, parameters, pipeline
from nagcsu.algorithms import sensitivity
from nagcsu.cli import context, display


@click.group(name="sensitivity")
def sensitivity_() -> None:
    """Local one-at-a-time sensitivity: which parameters move J the most."""


@sensitivity_.command(name="run")
@click.option(
    "--group",
    "group_name",
    default=None,
    help="Only test parameters in this tuning group. Defaults to every tunable parameter.",
)
@click.option(
    "--perturbation-fraction",
    default=0.15,
    show_default=True,
    help="Fraction of each parameter's bound range to perturb by, each direction.",
)
@click.option(
    "--top",
    "top_n",
    default=None,
    type=int,
    help="Show only the N most sensitive parameters, and recommend groups from them. Shows all by default.",
)
@click.option(
    "--detailed",
    is_flag=True,
    default=False,
    help=(
        "Rank by each scored vector's own NRMSE swing, not only combined J. Use this "
        "when J is dominated by one runaway vector (see nagcsu.config.ObjectiveConfig."
        "nrmse_ceiling), so a parameter that only moves that vector is not mistaken "
        "for one that actually helps pressure or water cut."
    ),
)
@click.option(
    "--group-rank",
    "group_rank_method",
    type=click.Choice(sensitivity.GROUP_RANK_METHODS),
    default="mean_rank",
    show_default=True,
    help=(
        "How groups are ordered. mean_rank: average of each member's swing rank, "
        "ascending (fair between groups of different sizes). rank_sum: total of member "
        "ranks, ascending (favors small groups). swing_share: total swing in J, descending."
    ),
)
@context.baseline_options
@context.wells_option
@context.weights_option
@click.pass_context
def run(
    ctx: click.Context,
    group_name: str | None,
    perturbation_fraction: float,
    top_n: int | None,
    detailed: bool,
    group_rank_method: str,
    baseline_raw: str | None,
    base_deck_raw: str | None,
    wells_raw: str | None,
    weights_raw: str | None,
) -> None:
    """Perturb each parameter up and down and rank them by how much J moved.

    Useful both on its own, to see where tuning effort is likely to
    pay off before running `nagcsu match auto` or `nagcsu match sweep`,
    and after auto-tuning stops short of the target, to see what is
    worth trying by hand next. Results are always sorted most-sensitive
    first; `--top` trims the list and adds a suggested `--groups` value
    for `match auto` built from whichever groups the top parameters
    belong to.
    """
    if group_name is not None and group_name not in constants.GROUP_TUNING_PRIORITY_ORDER:
        raise click.BadParameter(
            f"Unknown group {group_name!r}. Valid groups: {list(constants.GROUP_TUNING_PRIORITY_ORDER)}"
        )

    project_config, base_deck = context.load(ctx)
    project_config = context.apply_objective_overrides(
        project_config, wells_raw=wells_raw, weights_raw=weights_raw
    )
    specs = (
        parameters.get_parameters_in_group(group_name)
        if group_name
        else list(parameters.PARAMETERS.values())
    )
    baseline = context.resolve_baseline(
        project_config,
        base_deck,
        baseline_raw=baseline_raw,
        base_deck_raw=base_deck_raw,
    )
    display.print_baseline(baseline)
    # A baseline taken from an earlier run may sit outside the recommended
    # bounds; stretch each range to include it so the probes stay around it.
    parameter_bounds = {
        spec.name: context.widen_bounds_to_include(spec.bounds, baseline.state[spec.name])
        for spec in specs
    }
    parameter_groups = {spec.name: spec.group for spec in specs}

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        record = pipeline.build_run_record(
            outcome,
            group=None,
            strategy="sensitivity",
            note="sensitivity probe",
            baseline=baseline.ledger_label(),
            base_deck=baseline.deck_source,
        )
        ledger.append(ledger_path, record)

    if detailed:
        evaluate = pipeline.make_evaluate_with_breakdown(
            project_config,
            baseline.deck,
            run_id_prefix="sensitivity",
            on_outcome=on_outcome,
        )
        results = sensitivity.detailed_run(
            baseline.state,
            parameter_bounds,
            evaluate,
            perturbation_fraction=perturbation_fraction,
            parameter_groups=parameter_groups,
        )
        swings = {result.parameter: result.swing for result in results}
        ranks = sensitivity.rank_parameters(swings)
        ranked_groups = sensitivity.rank_groups(swings, parameter_groups, method=group_rank_method)
        shown_results = results[:top_n] if top_n else results
        if results:
            display.console.print(f"Base J = {results[0].base_j:.4f}")

        display.console.print(
            display.detailed_sensitivity_table(shown_results, parameter_groups, ranks)
        )
        display.console.print(
            display.group_sensitivity_table(ranked_groups, method=group_rank_method)
        )
        display.print_key(glossary.DETAILED_SENSITIVITY, glossary.GROUP_RANKING, ("Group",))
        echo_group_recommendation(shown_results, ranked_groups=ranked_groups)
        return

    evaluate = pipeline.make_evaluate(
        project_config,
        baseline.deck,
        run_id_prefix="sensitivity",
        on_outcome=on_outcome,
    )
    results, _ = sensitivity.run(
        baseline.state,
        parameter_bounds,
        evaluate,
        perturbation_fraction=perturbation_fraction,
        parameter_groups=parameter_groups,
    )

    shown_results = results[:top_n] if top_n else results
    click.echo(f"Base J = {results[0].base_j:.4f}" if results else "No parameters to test.")
    swings = {result.parameter: result.swing for result in results}
    ranks = sensitivity.rank_parameters(swings)
    ranked_groups = sensitivity.rank_groups(swings, parameter_groups, method=group_rank_method)
    display.console.print(display.sensitivity_table(shown_results, parameter_groups, ranks))
    display.console.print(display.group_sensitivity_table(ranked_groups, method=group_rank_method))
    display.print_key(glossary.SENSITIVITY, glossary.GROUP_RANKING, ("Group",))
    echo_group_recommendation(shown_results, ranked_groups=ranked_groups)


def recommend_groups(parameter_names: list[str], *, limit: int = 3) -> list[str]:
    """Map ranked parameter names back to their tuning groups.

    :param parameter_names: Parameter names, most sensitive first (the
        order `sensitivity.run`/`detailed_run` already return).
    :param limit: Most distinct groups to return.
    :returns: Group names in the order their first (most sensitive)
        parameter appeared, deduplicated, for example `["aquifer",
        "swof_endpoints"]`, a value directly usable as `match auto
        --groups`.
    """
    seen: list[str] = []
    for name in parameter_names:
        group = parameters.PARAMETERS[name].group
        if group not in seen:
            seen.append(group)
        if len(seen) >= limit:
            break
    return seen


def echo_group_recommendation(
    results: typing.Sequence[typing.Any],
    *,
    ranked_groups: typing.Sequence[sensitivity.GroupSensitivity] | None = None,
) -> None:
    """Print a `match auto --groups=...` suggestion.

    Built from the group ranking when `ranked_groups` is given, otherwise
    from the groups of the top `results`.
    """
    if not results:
        return

    groups = (
        [group.group for group in ranked_groups[:3]]
        if ranked_groups
        else recommend_groups([result.parameter for result in results])
    )
    click.echo(f"\nMost sensitive groups: {','.join(groups)}")
    click.echo(f"Try:  nagcsu match auto --groups {','.join(groups)}")
    click.echo("Or:   nagcsu match auto --order sensitivity   (screens, then tunes in this order)")


def echo_detailed_results(results: list[sensitivity.DetailedSensitivityResult]) -> None:
    """Print a `detailed_run` ranking, one column per scored vector plus combined J."""
    if not results:
        click.echo("No parameters to test.\n")
        return

    groups = {
        result.parameter: parameters.PARAMETERS[result.parameter].group for result in results
    }
    ranks = sensitivity.rank_parameters({result.parameter: result.swing for result in results})
    click.echo(f"Base J = {results[0].base_j:.4f}\n")
    display.console.print(display.detailed_sensitivity_table(results, groups, ranks))

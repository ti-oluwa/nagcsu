"""`nagcsu sensitivity`: rank parameters by local effect on J."""

import typing

import click

from nagcsu import constants, ledger, parameters, pipeline
from nagcsu.algorithms import sensitivity
from nagcsu.cli import context


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
@click.pass_context
def run(
    ctx: click.Context,
    group_name: str | None,
    perturbation_fraction: float,
    top_n: int | None,
    detailed: bool,
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
    specs = (
        parameters.get_parameters_in_group(group_name)
        if group_name
        else list(parameters.PARAMETERS.values())
    )
    parameter_bounds = {spec.name: spec.bounds for spec in specs}

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        record = pipeline.build_run_record(
            outcome,
            group=group_name,
            strategy="sensitivity",
            note="sensitivity probe",
        )
        ledger.append(ledger_path, record)

    if detailed:
        evaluate = pipeline.make_evaluate_with_breakdown(
            project_config,
            base_deck,
            run_id_prefix="sensitivity",
            on_outcome=on_outcome,
        )
        results = sensitivity.run_detailed(
            parameters.default_state(),
            parameter_bounds,
            evaluate,
            perturbation_fraction=perturbation_fraction,
        )
        echo_detailed_results(results[:top_n] if top_n else results)
        echo_group_recommendation(results[:top_n] if top_n else results)
        return

    evaluate = pipeline.make_evaluate(
        project_config,
        base_deck,
        run_id_prefix="sensitivity",
        on_outcome=on_outcome,
    )
    results, _ = sensitivity.run(
        parameters.default_state(),
        parameter_bounds,
        evaluate,
        perturbation_fraction=perturbation_fraction,
    )

    shown = results[:top_n] if top_n else results
    click.echo(f"Base J = {results[0].base_j:.4f}\n" if results else "No parameters to test.\n")
    click.echo(f"{'Parameter':<40}{'J at low':>12}{'J at high':>12}{'Swing':>12}")
    for result in shown:
        click.echo(
            f"{result.parameter:<40}{result.j_at_low:>12.4f}{result.j_at_high:>12.4f}{result.swing:>12.4f}"
        )
    echo_group_recommendation(shown)


def recommend_groups(parameter_names: list[str], *, limit: int = 3) -> list[str]:
    """Map ranked parameter names back to their tuning groups.

    :param parameter_names: Parameter names, most sensitive first (the
        order `sensitivity.run`/`run_detailed` already return).
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


def echo_group_recommendation(results: typing.Sequence[typing.Any]) -> None:
    """Print a `match auto --groups=...` suggestion built from `results`."""
    if not results:
        return
    groups = recommend_groups([result.parameter for result in results])
    click.echo(f"\nMost sensitive groups: {','.join(groups)}")
    click.echo(f"Try:  nagcsu match auto --groups {','.join(groups)}")


def echo_detailed_results(results: list[sensitivity.DetailedSensitivityResult]) -> None:
    """Print a `run_detailed` ranking, one column per scored vector plus combined J."""
    if not results:
        click.echo("No parameters to test.\n")
        return

    vector_names = sorted({name for result in results for name in result.vector_swings})
    click.echo(f"Base J = {results[0].base_j:.4f}\n")
    header = f"{'Parameter':<40}{'Swing (J)':>12}"
    for vector_name in vector_names:
        header += f"{vector_name:>14}"
    click.echo(header)

    for result in results:
        row = f"{result.parameter:<40}{result.swing:>12.4f}"
        for vector_name in vector_names:
            row += f"{result.vector_swings.get(vector_name, 0.0):>14.4f}"
        click.echo(row)

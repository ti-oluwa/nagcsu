"""`nagcsu sensitivity`: rank parameters by local effect on J."""

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
@click.pass_context
def run(ctx: click.Context, group_name: str | None, perturbation_fraction: float) -> None:
    """Perturb each parameter up and down and rank them by how much J moved.

    Useful both on its own, to see where tuning effort is likely to
    pay off before running `nagcsu match auto`, and after auto-tuning
    stops short of the target, to see what is worth trying by hand next.
    """
    if group_name is not None and group_name not in constants.TUNING_PRIORITY_ORDER:
        raise click.BadParameter(
            f"Unknown group {group_name!r}. Valid groups: {list(constants.TUNING_PRIORITY_ORDER)}"
        )

    project_config, base_deck = context.load(ctx)
    specs = (
        parameters.parameters_in_group(group_name)
        if group_name
        else list(parameters.PARAMETERS.values())
    )
    bounds_by_parameter = {spec.name: spec.bounds for spec in specs}

    ledger_path = project_config.get_resolved_path(project_config.ledger_path)

    def on_outcome(outcome: pipeline.RunOutcome) -> None:
        record = pipeline.to_run_record(
            outcome, group=group_name, strategy="sensitivity", note="sensitivity probe"
        )
        ledger.append(ledger_path, record)

    evaluate = pipeline.make_evaluate(
        project_config, base_deck, run_id_prefix="sensitivity", on_outcome=on_outcome
    )

    results, _ = sensitivity.run(
        parameters.default_state(),
        bounds_by_parameter,
        evaluate,
        perturbation_fraction=perturbation_fraction,
    )

    click.echo(f"Base J = {results[0].base_j:.4f}\n" if results else "No parameters to test.\n")
    click.echo(f"{'Parameter':<40}{'J at low':>12}{'J at high':>12}{'Swing':>12}")
    for result in results:
        click.echo(
            f"{result.parameter:<40}{result.j_at_low:>12.4f}{result.j_at_high:>12.4f}{result.swing:>12.4f}"
        )

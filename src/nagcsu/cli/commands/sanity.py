"""`nagcsu sanity`: physical-plausibility checks a clean `.PRT` file misses."""

import click

from nagcsu import init_check, ledger, parameters, simulate
from nagcsu.cli import context


@click.group(name="sanity")
def sanity() -> None:
    """Checks a clean `.PRT` file does not by itself cover."""


@sanity.command(name="check")
@click.option(
    "--run-id", default=None, help="Run to check. Defaults to the most recently logged run."
)
@click.option(
    "--tolerance",
    default=0.05,
    show_default=True,
    help="How far above Swc the mean initial water saturation may sit before this fails.",
)
@click.pass_context
def check(ctx: click.Context, run_id: str | None, tolerance: float) -> None:
    """Compare a run's pre-production water saturation against the deck's own Swc.

    Catches an EQUIL contact-depth, PVT/density, or capillary-pressure
    setup that starts the model already near residual oil saturation,
    something a clean `.PRT` file does not check for, and that no
    history-match parameter (aquifer, relperm shape, and so on) can fix.
    Run this once against a fresh baseline run before spending any time
    on `nagcsu match auto`.
    """
    project_config, _ = context.load(ctx)
    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    records = ledger.load(ledger_path)
    if not records:
        raise click.ClickException("No runs logged yet; run `nagcsu run` first.")
    resolved_run_id = run_id or records[-1].run_id

    output_dir = project_config.get_resolved_path(project_config.output_root) / resolved_run_id
    case_basename = simulate.find_case_basename(output_dir)
    if case_basename is None:
        raise click.ClickException(f"No .UNSMRY/.UNRST output found under {output_dir}")

    try:
        report = init_check.check_initial_water_saturation(
            case_basename,
            expected_water_saturation=parameters.CONNATE_WATER_SATURATION,
            tolerance=tolerance,
        )
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Run {resolved_run_id}: {report.cell_count} active cells")
    click.echo(
        f"  Initial water saturation: min={report.min_water_saturation:.4f} "
        f"mean={report.mean_water_saturation:.4f} max={report.max_water_saturation:.4f}"
    )
    click.echo(
        f"  Expected (Swc from SWOF): {report.expected_water_saturation:.4f} "
        f"(tolerance +{report.tolerance:.4f})"
    )
    if report.is_plausible:
        click.echo(
            "  PLAUSIBLE: starts near connate water saturation, as a fresh reservoir should."
        )
        return

    click.echo(
        "  NOT PLAUSIBLE: mean initial water saturation is far above Swc. The model starts "
        "already watered out, before any production. Check EQUIL's WOC/GOC depths against "
        "the grid's actual depth range, the density/PVT inputs behind the capillary-gravity "
        "balance, and the SWOF Pcow column, rather than tuning aquifer or relperm parameters."
    )
    raise SystemExit(1)

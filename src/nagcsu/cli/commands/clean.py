"""`nagcsu clean`: remove old runs from the ledger, from disk, or both."""

import re

import click

from nagcsu import cleanup, ledger
from nagcsu.cli import context, display


def humanize_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{count} B"


@click.command(name="clean")
@click.option("--all", "select_all", is_flag=True, help="Select every run.")
@click.option(
    "--prefix",
    "prefixes",
    multiple=True,
    help="Select runs whose ID starts with this. Repeatable.",
)
@click.option("--id", "ids", multiple=True, help="Select this exact run ID. Repeatable.")
@click.option(
    "--regex",
    "regexes",
    multiple=True,
    help="Select runs whose ID fully matches this regex. Repeatable.",
)
@click.option(
    "--strategy",
    "strategies",
    multiple=True,
    help="Select runs logged by this strategy (sensitivity, coordinate_descent, sweep, random...). Repeatable.",
)
@click.option(
    "--group",
    "groups",
    multiple=True,
    help="Select runs logged under this tuning group. Repeatable.",
)
@click.option(
    "--stage",
    "stages",
    multiple=True,
    help="Select runs whose stage starts with this, e.g. descent or sensitivity. Repeatable.",
)
@click.option(
    "--sensitivity",
    "sensitivity_only",
    is_flag=True,
    help="Select sensitivity trials (by strategy, stage, or a run ID starting with 'sensitivity').",
)
@click.option("--failed", "failed_only", is_flag=True, help="Select runs whose simulation failed.")
@click.option("--keep-id", "keep_ids", multiple=True, help="Never clean this run ID. Repeatable.")
@click.option(
    "--keep-prefix",
    "keep_prefixes",
    multiple=True,
    help="Never clean runs whose ID starts with this. Repeatable.",
)
@click.option(
    "--keep-regex",
    "keep_regexes",
    multiple=True,
    help="Never clean runs whose ID matches this regex. Repeatable.",
)
@click.option(
    "--keep-best",
    default=0,
    show_default=True,
    help="Never clean the N lowest-J runs in the whole ledger.",
)
@click.option(
    "--keep-latest",
    default=0,
    show_default=True,
    help="Never clean the N most recently logged runs.",
)
@click.option(
    "--scope",
    type=click.Choice(cleanup.SCOPES),
    default="both",
    show_default=True,
    help="both: remove ledger records and run files. ledger: remove records only, leave files. files: delete run files only, keep the records.",
)
@click.option(
    "--keep-files",
    "keep_files",
    multiple=True,
    help="Glob of files to leave in each cleaned run directory, e.g. --keep-files '*.DATA' --keep-files '*.UNSMRY'. Everything else in it is deleted. Repeatable.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be removed and change nothing.")
@click.option("--yes", "-y", is_flag=True, help="Do not ask for confirmation.")
@click.pass_context
def clean(
    ctx: click.Context,
    select_all: bool,
    prefixes: tuple[str, ...],
    ids: tuple[str, ...],
    regexes: tuple[str, ...],
    strategies: tuple[str, ...],
    groups: tuple[str, ...],
    stages: tuple[str, ...],
    sensitivity_only: bool,
    failed_only: bool,
    keep_ids: tuple[str, ...],
    keep_prefixes: tuple[str, ...],
    keep_regexes: tuple[str, ...],
    keep_best: int,
    keep_latest: int,
    scope: str,
    keep_files: tuple[str, ...],
    dry_run: bool,
    yes: bool,
) -> None:
    """Remove old runs from the ledger, the output directory, or both.

    A run is selected when it matches EVERY selector you give (values of
    one selector are alternatives), then anything matched by a --keep-*
    rule is protected. With no selector at all nothing happens; use --all
    to mean everything. Use --dry-run first.

    \b
    Examples:
      nagcsu clean --sensitivity                          drop sensitivity trials
      nagcsu clean --prefix sensitivity --dry-run         same idea, by run ID
      nagcsu clean --all --keep-best 1 --keep-id auto_final
      nagcsu clean --all --keep-best 1 --scope files \\
          --keep-files '*.DATA' --keep-files '*.UNSMRY'   slim the run folders
      nagcsu clean --failed --scope ledger                forget failed runs only
    """
    selection = cleanup.Selection(
        select_all=select_all,
        prefixes=prefixes,
        ids=ids,
        regexes=regexes,
        strategies=strategies,
        groups=groups,
        stages=stages,
        sensitivity_only=sensitivity_only,
        failed_only=failed_only,
        keep_ids=keep_ids,
        keep_prefixes=keep_prefixes,
        keep_regexes=keep_regexes,
        keep_best=keep_best,
        keep_latest=keep_latest,
    )
    if not selection.has_selector():
        raise click.UsageError(
            "Nothing selected. Give at least one of --all, --prefix, --id, --regex, --strategy, "
            "--group, --stage, --sensitivity or --failed."
        )
    for pattern in (*regexes, *keep_regexes):
        try:
            re.compile(pattern)
        except re.error as error:
            raise click.BadParameter(f"Invalid regex {pattern!r}: {error}") from error

    project_config, _ = context.load(ctx)
    ledger_path = project_config.get_resolved_path(project_config.ledger_path)
    output_root = project_config.get_resolved_path(project_config.output_root)
    records = ledger.load(ledger_path)
    targets = cleanup.collect_targets(records, output_root)
    to_clean, protected = cleanup.select(targets, selection)

    try:
        actions = cleanup.plan(
            to_clean, scope=scope, keep_files=keep_files, output_root=output_root
        )
    except ValueError as error:
        raise click.UsageError(str(error)) from error

    if protected:
        display.console.print(display.protected_table(protected))
    if not actions:
        click.echo("No runs matched.")
        return

    display.console.print(display.cleanup_table(actions, scope=scope))
    freed = sum(action.bytes_to_free for action in actions)
    records_gone = sum(1 for action in actions if action.remove_record)
    click.echo(
        f"{len(actions)} run(s) selected: {records_gone} ledger record(s) to remove, "
        f"{humanize_bytes(freed)} to free on disk, {len(protected)} protected."
    )
    if dry_run:
        click.echo("Dry run: nothing changed.")
        return
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Cancelled.")
        return

    result = cleanup.execute(actions, ledger_path=ledger_path, output_root=output_root)
    click.echo(
        f"Removed {result.records_removed} record(s), {result.files_removed} file(s) "
        f"({result.directories_removed} whole director{'y' if result.directories_removed == 1 else 'ies'}), "
        f"freed {humanize_bytes(result.bytes_freed)}."
    )

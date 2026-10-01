"""Selecting and removing old runs from the ledger and from disk.

Selection is deliberately explicit: a run is chosen only when it matches
every selector given (a selector with several values matches if any one
does), and anything matched by a `keep_*` rule is protected afterwards,
so "everything except the best run and `auto_final`" is one command.

Nothing here prompts or prints. `nagcsu.cli.commands.clean` does that;
`plan` only describes what would happen and `execute` does it.
"""

import dataclasses
import fnmatch
import pathlib
import re
import shutil
import typing

from nagcsu import ledger

SCOPES: typing.Final[tuple[str, ...]] = ("both", "ledger", "files")
"""What a clean touches: ledger records and run files, only the ledger, or only run files."""


@dataclasses.dataclass(frozen=True, slots=True)
class Selection:
    """Which runs to clean, and which to protect from it."""

    select_all: bool = False
    """Select every run, before protection is applied."""

    prefixes: tuple[str, ...] = ()
    """Run ID starts with any of these."""

    ids: tuple[str, ...] = ()
    """Run ID equals any of these."""

    regexes: tuple[str, ...] = ()
    """Run ID fully matches any of these patterns."""

    strategies: tuple[str, ...] = ()
    """Record's strategy equals any of these. Never matches a directory with no record."""

    groups: tuple[str, ...] = ()
    """Record's group equals any of these."""

    stages: tuple[str, ...] = ()
    """Record's stage starts with any of these, so "descent" matches "descent/pass1"."""

    sensitivity_only: bool = False
    """Only sensitivity trials: strategy or stage says sensitivity, or the ID starts with it."""

    failed_only: bool = False
    """Only runs whose simulation failed."""

    keep_ids: tuple[str, ...] = ()
    """Protect runs with exactly these IDs."""

    keep_prefixes: tuple[str, ...] = ()
    """Protect runs whose ID starts with any of these."""

    keep_regexes: tuple[str, ...] = ()
    """Protect runs whose ID fully matches any of these patterns."""

    keep_best: int = 0
    """Protect the N lowest-J scored runs in the whole ledger."""

    keep_latest: int = 0
    """Protect the N most recently logged runs."""

    def has_selector(self) -> bool:
        """Whether anything at all was chosen to select runs by."""
        return bool(
            self.select_all
            or self.prefixes
            or self.ids
            or self.regexes
            or self.strategies
            or self.groups
            or self.stages
            or self.sensitivity_only
            or self.failed_only
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Target:
    """One run known to the ledger, to the output directory, or to both."""

    run_id: str
    record: ledger.RunRecord | None
    """`None` for a directory on disk that the ledger has no record of."""

    directory: pathlib.Path | None
    """`None` when the run has a record but no directory on disk."""

    record_count: int = 1
    """How many ledger records share this run ID. More than 1 happens when
    an older version reused an ID (for example two sweeps both writing
    `sweep_00000`); `record` is then the latest of them, and cleaning the
    run removes all of them, since they shared one directory."""


@dataclasses.dataclass(frozen=True, slots=True)
class Action:
    """What `execute` will do to one run."""

    target: Target
    remove_record: bool
    delete_whole_directory: bool
    files_to_delete: list[pathlib.Path]
    files_to_keep: list[pathlib.Path]
    bytes_to_free: int


@dataclasses.dataclass(frozen=True, slots=True)
class Result:
    """What `execute` actually did."""

    records_removed: int
    directories_removed: int
    files_removed: int
    bytes_freed: int


def collect_targets(records: list[ledger.RunRecord], output_root: pathlib.Path) -> list[Target]:
    """List every run in the ledger, then every extra directory under `output_root`.

    One target per run ID. If several records share an ID, they share one
    directory too (the later run overwrote the earlier), so they are one
    target whose `record` is the latest and whose `record_count` says how
    many records it stands for.
    """
    latest: dict[str, ledger.RunRecord] = {}
    counts: dict[str, int] = {}
    for record in records:
        latest[record.run_id] = record
        counts[record.run_id] = counts.get(record.run_id, 0) + 1

    directories = (
        {path.name: path for path in sorted(output_root.iterdir()) if path.is_dir()}
        if output_root.is_dir()
        else {}
    )
    targets = [
        Target(
            run_id=run_id,
            record=record,
            directory=directories.get(run_id),
            record_count=counts[run_id],
        )
        for run_id, record in latest.items()
    ]
    targets.extend(
        Target(run_id=name, record=None, directory=path)
        for name, path in directories.items()
        if name not in latest
    )
    return targets


def check_matches_any_pattern(run_id: str, patterns: typing.Iterable[str]) -> bool:
    return any(re.fullmatch(pattern, run_id) for pattern in patterns)


def is_sensitivity(target: Target) -> bool:
    record = target.record
    return (
        target.run_id.startswith("sensitivity")
        or (record is not None and record.strategy == "sensitivity")
        or (record is not None and (record.stage or "").startswith("sensitivity"))
    )


def is_selected(target: Target, selection: Selection) -> bool:
    if selection.select_all:
        return True
    record = target.record
    checks: list[bool] = []
    if selection.prefixes:
        checks.append(any(target.run_id.startswith(prefix) for prefix in selection.prefixes))
    if selection.ids:
        checks.append(target.run_id in selection.ids)
    if selection.regexes:
        checks.append(check_matches_any_pattern(target.run_id, selection.regexes))
    if selection.strategies:
        checks.append(record is not None and record.strategy in selection.strategies)
    if selection.groups:
        checks.append(record is not None and record.group in selection.groups)
    if selection.stages:
        checks.append(
            record is not None
            and any((record.stage or "").startswith(stage) for stage in selection.stages)
        )
    if selection.sensitivity_only:
        checks.append(is_sensitivity(target))
    if selection.failed_only:
        checks.append(record is not None and record.simulation_error is not None)
    return bool(checks) and all(checks)


def select(
    targets: list[Target], selection: Selection
) -> tuple[list[Target], list[tuple[Target, str]]]:
    """Split the selected runs into those to clean and those protected.

    :returns: `(to_clean, protected)`, where `protected` pairs each
        selected-but-kept run with the rule that kept it.
    """
    scored = sorted(
        (target for target in targets if target.record and target.record.j is not None),
        key=lambda target: target.record.j,  # type: ignore[union-attr]
    )
    best_ids = {target.run_id for target in scored[: max(0, selection.keep_best)]}
    recorded = [target for target in targets if target.record is not None]
    latest_ids = (
        {target.run_id for target in recorded[-selection.keep_latest :]}
        if selection.keep_latest > 0
        else set()
    )

    to_clean: list[Target] = []
    protected: list[tuple[Target, str]] = []
    for target in targets:
        if not is_selected(target, selection):
            continue
        reason: str | None = None
        if target.run_id in selection.keep_ids:
            reason = "kept by --keep-id"
        elif any(target.run_id.startswith(prefix) for prefix in selection.keep_prefixes):
            reason = "kept by --keep-prefix"
        elif check_matches_any_pattern(target.run_id, selection.keep_regexes):
            reason = "kept by --keep-regex"
        elif target.run_id in best_ids:
            reason = "kept: among the best runs"
        elif target.run_id in latest_ids:
            reason = "kept: among the latest runs"
        if reason:
            protected.append((target, reason))
        else:
            to_clean.append(target)
    return to_clean, protected


def _matches_keep_glob(relative: pathlib.PurePosixPath, patterns: typing.Sequence[str]) -> bool:
    return any(
        fnmatch.fnmatch(relative.name, pattern) or fnmatch.fnmatch(str(relative), pattern)
        for pattern in patterns
    )


def is_inside(directory: pathlib.Path, output_root: pathlib.Path) -> bool:
    """Whether `directory` is a real (non-symlink) directory strictly inside `output_root`."""
    if directory.is_symlink():
        return False
    resolved_root = output_root.resolve()
    resolved = directory.resolve()
    return resolved != resolved_root and resolved.is_relative_to(resolved_root)


def plan(
    targets: list[Target],
    *,
    scope: str,
    keep_files: typing.Sequence[str],
    output_root: pathlib.Path,
) -> list[Action]:
    """Work out, without touching anything, what cleaning `targets` would do.

    :param scope: One of `SCOPES`.
    :param keep_files: Glob patterns of files to leave in a run's
        directory (matched against the file name and the path relative
        to the run directory). Empty means delete the whole directory.
    :raises ValueError: for an unknown `scope`, or `keep_files` with a
        scope that never touches files.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; use one of {SCOPES}")
    if keep_files and scope == "ledger":
        raise ValueError(
            "--keep-files has no effect with --scope ledger, which never touches files"
        )

    actions: list[Action] = []
    for target in targets:
        remove_record = scope in ("both", "ledger") and target.record is not None
        delete: list[pathlib.Path] = []
        keep: list[pathlib.Path] = []
        freed = 0
        whole = False
        directory = target.directory
        if (
            scope in ("both", "files")
            and directory is not None
            and is_inside(directory, output_root)
        ):
            for path in sorted(directory.rglob("*")):
                if not path.is_file() and not path.is_symlink():
                    continue
                relative = pathlib.PurePosixPath(path.relative_to(directory).as_posix())
                if keep_files and _matches_keep_glob(relative, keep_files):
                    keep.append(path)
                else:
                    delete.append(path)
                    freed += path.lstat().st_size
            whole = not keep
        actions.append(
            Action(
                target=target,
                remove_record=remove_record,
                delete_whole_directory=whole and directory is not None and scope != "ledger",
                files_to_delete=delete,
                files_to_keep=keep,
                bytes_to_free=freed,
            )
        )
    return actions


def execute(
    actions: list[Action], *, ledger_path: pathlib.Path, output_root: pathlib.Path
) -> Result:
    """Carry out `actions`: rewrite the ledger once, then delete files.

    Files are deleted only inside a run directory that is strictly inside
    `output_root`; a directory that is a symlink is skipped, never followed.
    """
    removed_ids = {action.target.run_id for action in actions if action.remove_record}
    records_removed = 0
    if removed_ids:
        before = ledger.load(ledger_path)
        remaining = [record for record in before if record.run_id not in removed_ids]
        records_removed = len(before) - len(remaining)
        ledger.save(ledger_path, remaining)

    directories_removed = files_removed = bytes_freed = 0
    handled: set[pathlib.Path] = set()
    for action in actions:
        directory = action.target.directory
        if directory is None or not directory.exists() or not is_inside(directory, output_root):
            continue
        resolved = directory.resolve()
        if resolved in handled:
            continue
        handled.add(resolved)
        if action.delete_whole_directory:
            shutil.rmtree(directory, ignore_errors=True)
            directories_removed += 1
            files_removed += len(action.files_to_delete)
            bytes_freed += action.bytes_to_free
            continue
        for path in action.files_to_delete:
            path.unlink(missing_ok=True)
            files_removed += 1
        bytes_freed += action.bytes_to_free
        for sub in sorted(directory.rglob("*"), reverse=True):
            if sub.is_dir() and not any(sub.iterdir()):
                sub.rmdir()
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            directories_removed += 1
    return Result(
        records_removed=records_removed,
        directories_removed=directories_removed,
        files_removed=files_removed,
        bytes_freed=bytes_freed,
    )

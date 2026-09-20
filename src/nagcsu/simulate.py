"""Running OPM Flow and locating the summary files it wrote.

OPM Flow does not necessarily write its output under the same basename
as the input `.DATA` file: a run in this project's `Data/` folder shows
Flow folding the case name to upper case when it differs from the input
filename's own case ("NigerDelta UGH1 Composite Field.DATA" produced
"NIGERDELTA UGH1 COMPOSITE FIELD.PRT/.UNSMRY/.SMSPEC"). Rather than
guess at that transform, :func:`find_case_basename` globs the output
directory for whatever `.UNSMRY` file actually appeared.
"""

import dataclasses
import pathlib
import subprocess
import time

from nagcsu.exceptions import RunOutputNotFoundError, SimulationError


@dataclasses.dataclass(frozen=True, slots=True)
class RunResult:
    """Where a completed (or failed) OPM Flow run's files ended up."""

    output_dir: pathlib.Path
    """Directory OPM Flow was told to write its output to."""

    case_basename: pathlib.Path | None
    """Output files' basename (without extension), for example
    `output_dir / "NIGERDELTA UGH1 COMPOSITE FIELD"`. `None` if no
    `.UNSMRY` file was found, which usually means Flow exited before
    writing any summary output.
    """

    returncode: int
    """Exit code of the `flow` process."""

    stdout: str
    """Captured standard output of the `flow` process."""

    stderr: str
    """Captured standard error of the `flow` process."""

    elapsed_seconds: float
    """Wall-clock time the `flow` process ran for."""

    @property
    def prt_path(self) -> pathlib.Path | None:
        """Path to the run's `.PRT` file, or `None` if none was found."""
        return self.case_basename.with_suffix(".PRT") if self.case_basename else None


def find_case_basename(output_dir: pathlib.Path) -> pathlib.Path | None:
    """Find the basename OPM Flow actually wrote its output under.

    :returns: `output_dir / "<CASE>"` (no extension) for the single
        `.UNSMRY` file found in `output_dir`, or `None` if none exists.
    :raises RunOutputNotFoundError: if more than one `.UNSMRY` file is
        found, since it is then ambiguous which one this run produced.
    """
    matches = sorted(output_dir.glob("*.UNSMRY"))
    if not matches:
        return None
    if len(matches) > 1:
        raise RunOutputNotFoundError(
            f"Found {len(matches)} .UNSMRY files in {output_dir}, expected at most one: "
            f"{[str(match) for match in matches]}. Give each run its own output directory."
        )
    return matches[0].with_suffix("")


def run(
    deck_path: pathlib.Path | str,
    output_dir: pathlib.Path | str,
    *,
    flow_executable: str = "flow",
    extra_args: list[str] = (),
    timeout_seconds: float | None = None,
) -> RunResult:
    """Run OPM Flow against `deck_path`, writing output to `output_dir`.

    Does not raise on a nonzero `flow` exit code by itself; a nonzero
    exit with no summary output raises :class:`SimulationError` since
    there is nothing left to score, but a run that wrote a `.PRT` and
    `.UNSMRY` before failing (for example, a run that could not reach
    its last report step) is returned as a normal `RunResult` so the
    caller can inspect `prt.parse(result.prt_path)` to see how far it got.

    :raises SimulationError: if `flow` could not be launched at all, or
        exited nonzero without producing any summary output.
    """
    deck_path = pathlib.Path(deck_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    try:
        completed = subprocess.run(
            [flow_executable, str(deck_path), f"--output-dir={output_dir}", *extra_args],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SimulationError(f"Could not run {flow_executable!r} on {deck_path}: {error}") from error
    elapsed_seconds = time.monotonic() - started

    case_basename = find_case_basename(output_dir)
    if completed.returncode != 0 and case_basename is None:
        raise SimulationError(
            f"{flow_executable} exited {completed.returncode} and wrote no summary output "
            f"to {output_dir}",
            returncode=completed.returncode,
            stderr=completed.stderr,
        )

    return RunResult(
        output_dir=output_dir,
        case_basename=case_basename,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_seconds=elapsed_seconds,
    )

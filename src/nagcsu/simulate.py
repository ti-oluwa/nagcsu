"""Running OPM Flow and locating the summary files it wrote.

OPM Flow does not necessarily write its output under the same basename
as the input `.DATA` file: a run in this project's `Data/` folder shows
Flow folding the case name to upper case when it differs from the input
filename's own case ("NigerDelta UGH1 Composite Field.DATA" produced
"NIGERDELTA UGH1 COMPOSITE FIELD.PRT/.UNSMRY/.SMSPEC"). Rather than
guess at that transform, `find_case_basename` globs the output
directory for whatever `.UNSMRY` file actually appeared.

A second, easy-to-miss gotcha: on many installs `flow` is not the real
OPM Flow binary at all, but a wrapper script that runs it inside a
Docker container, mounting only the *current working directory* into
that container (see the opmflow-setup-guide installer, which is what
this module's `cwd`/relative-path handling below is written against).
A subprocess invocation that passes an absolute path outside whatever
directory the wrapper happens to run from, or that does not set the
subprocess's own working directory at all, can silently fail to find
files that genuinely exist on disk, since the container simply cannot
see them. `run()` always launches `flow` from the common parent
directory of `deck_path` and `output_dir`, and passes both as paths
relative to that directory, so this works whether `flow` is a native
binary or a Docker-wrapped one, on every platform.
"""

import dataclasses
import os
import pathlib
import platform
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


def common_working_directory(paths: list[pathlib.Path]) -> pathlib.Path:
    """Return the deepest directory that is an ancestor of every path in `paths`.

    Used to pick the `cwd` a Docker-wrapped `flow` is launched from, so
    that its "mount whatever directory you ran me from" behavior covers
    both the deck and the output directory. Falls back to the first
    path's own parent directory if the paths share no common ancestor
    at all (different drives on Windows, for example), which is the
    best available approximation.
    """
    resolved = [path.resolve() for path in paths]
    try:
        return pathlib.Path(os.path.commonpath([str(path) for path in resolved]))
    except ValueError:
        return resolved[0].parent


def format_extra_mounts_env(extra_mounts: list[str]) -> str:
    """Format `extra_mounts` into the `OPM_FLOW_EXTRA_MOUNTS` value for this platform.

    Each entry in `extra_mounts` is either a bare host path (mounted at
    the same path inside the container, the default the installer's
    wrapper script itself falls back to) or a `HOST=CONTAINER` pair to
    mount at a different path inside the container. Entries are joined
    with `;` on Windows and `:` everywhere else, matching the syntax the
    opmflow-setup-guide installer's wrapper script documents for each
    platform (Windows paths already use `:` for the drive letter, so
    `;` is used as the separator there instead).
    """
    separator = ";" if platform.system() == "Windows" else ":"
    return separator.join(extra_mounts)


def run(
    deck_path: pathlib.Path | str,
    output_dir: pathlib.Path | str,
    *,
    flow_executable: str = "flow",
    extra_args: list[str] | None = None,
    extra_mounts: list[str] | None = None,
    timeout_seconds: float | None = None,
) -> RunResult:
    """Run OPM Flow against `deck_path`, writing output to `output_dir`.

    Launches `flow` with its working directory set to the common parent
    of `deck_path` and `output_dir`, passing both as paths relative to
    that directory (see the module docstring for why: a Docker-wrapped
    `flow` only mounts whatever directory it is run from). This has no
    effect on a native, non-Docker `flow` binary beyond changing how the
    path happens to be spelled on the command line.

    Does not raise on a nonzero `flow` exit code by itself; a nonzero
    exit with no summary output raises `SimulationError` since there is
    nothing left to score, but a run that wrote a `.PRT` and `.UNSMRY`
    before failing (for example, a run that could not reach its last
    report step) is returned as a normal `RunResult` so the caller can
    inspect `prt.parse(result.prt_path)` to see how far it got.

    :param extra_mounts: Host directories a Docker-wrapped `flow` needs
        to see beyond `deck_path` and `output_dir`'s common parent, for
        example because the deck has an `INCLUDE` pointing elsewhere.
        Passed through as the `OPM_FLOW_EXTRA_MOUNTS` environment
        variable for this one invocation; see
        `nagcsu.config.ProjectConfig.extra_mounts`. Harmless to set for
        a native, non-Docker `flow`, which simply never reads it.
    :raises SimulationError: if `flow` could not be launched at all, or
        exited nonzero without producing any summary output.
    """
    deck_path = pathlib.Path(deck_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    working_directory = common_working_directory([deck_path, output_dir])
    try:
        deck_arg = str(deck_path.resolve().relative_to(working_directory))
        output_dir_arg = str(output_dir.resolve().relative_to(working_directory))
    except ValueError:
        # deck_path and output_dir share no common ancestor at all; fall
        # back to absolute paths, which still works for a native flow
        # binary, just not for a Docker-wrapped one.
        deck_arg = str(deck_path.resolve())
        output_dir_arg = str(output_dir.resolve())

    env = os.environ.copy()
    if extra_mounts:
        env["OPM_FLOW_EXTRA_MOUNTS"] = format_extra_mounts_env(extra_mounts)

    started = time.monotonic()
    try:
        completed = subprocess.run(
            [flow_executable, deck_arg, f"--output-dir={output_dir_arg}", *(extra_args or [])],
            cwd=working_directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SimulationError(
            f"Could not run {flow_executable!r} on {deck_path}: {error}"
        ) from error
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

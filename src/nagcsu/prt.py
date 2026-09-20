"""Parsing an OPM Flow `.PRT` file for run health.

Stage A of the Phase 2 Execution Plan describes a material-balance-error
grep that assumes an ECLIPSE-style "MATERIAL BALANCE" line. OPM Flow
2026.04 does not print one; a real baseline `.PRT` from this project
instead ends with an `Error summary:` block (Warnings/Info/Errors/Bugs/
Problems counts) and an `Overall Newton Iterations` block reporting how
much solver work was wasted on retried timesteps, and marks failed well
convergence per report step as
`Warning: Inner well iterations failed for well <NAME> Treat the well
as unconverged.` paired with a preceding `Problem: [...] Error when
inverting local well equations for well <NAME>` line. This module is
built against that real format rather than the plan's placeholder one.
"""

import dataclasses
import pathlib
import re

ERROR_SUMMARY_PATTERN = re.compile(
    r"Error summary:\s*\n"
    r"Warnings\s+(?P<warnings>\d+)\s*\n"
    r"Info\s+(?P<info>\d+)\s*\n"
    r"Errors\s+(?P<errors>\d+)\s*\n"
    r"Bugs\s+(?P<bugs>\d+)\s*\n"
    r"Debug\s+(?P<debug>\d+)\s*\n"
    r"Problems\s+(?P<problems>\d+)"
)
"""Matches the summary block OPM Flow prints at the end of a `.PRT` file."""

UNCONVERGED_WELL_PATTERN = re.compile(
    r"Inner well iterations failed for well (?P<well>\S+) Treat the well as unconverged\."
)
"""Matches one well-convergence-failure warning line."""

REPORT_STEP_PATTERN = re.compile(
    r"Complete report step (?P<step>\d+) \(\d+ DAYS\) at (?P<date>\d{4}-\d{2}-\d{2})"
)
"""Matches one report-step completion line."""

OVERALL_ITERATIONS_PATTERN = re.compile(
    r"Overall (?P<kind>Linearizations|Newton Iterations|Linear Iterations):\s+"
    r"(?P<total>\d+)\s+\(Wasted:\s+(?P<wasted>\d+);\s+(?P<wasted_pct>[\d.]+)%\)"
)
"""Matches one of the three "Overall ... (Wasted: N; P%)" summary lines."""

NEGATIVE_STATE_PATTERN = re.compile(r"negative (?:saturation|pressure)", re.IGNORECASE)
"""Not observed in any sample run so far, but OPM Flow can in principle
emit this for an unphysical cell; kept as a defensive check.
"""

TOO_SMALL_TIMESTEP_PATTERN = re.compile(r"too small a timestep", re.IGNORECASE)
"""Not observed in any sample run so far; kept as a defensive check for
a timestep that was cut below the solver's minimum and could not recover.
"""


@dataclasses.dataclass(frozen=True, slots=True)
class WastedWork:
    """One "Overall ... (Wasted: N; P%)" line from the end of a `.PRT` file."""

    total: int
    """Total count of this kind of solver work over the whole run."""

    wasted: int
    """How many of `total` were thrown away by a retried timestep."""

    wasted_percent: float
    """`wasted` as a percentage of `total`, as printed by OPM Flow."""


@dataclasses.dataclass(frozen=True, slots=True)
class PrtReport:
    """Health summary extracted from one OPM Flow `.PRT` file."""

    path: pathlib.Path
    """Path of the `.PRT` file this report was parsed from."""

    warnings: int
    """Warning count from the run's `Error summary:` block."""

    errors: int
    """Fatal error count from the run's `Error summary:` block. A clean
    run should always have this at zero.
    """

    bugs: int
    """Bug count from the run's `Error summary:` block. A clean run
    should always have this at zero.
    """

    problems: int
    """Problem count from the run's `Error summary:` block. Each one
    generally pairs with an unconverged-well warning.
    """

    unconverged_well_counts: dict[str, int]
    """Number of "Treat the well as unconverged" warnings per well name,
    over the whole run. A handful scattered across a multi-decade
    history is normal; a count concentrated at one report step, or one
    that only appears after a parameter change, is worth investigating.
    """

    completed_report_steps: list[int]
    """Report step numbers OPM Flow reported completing, in order. The
    caller compares `len(completed_report_steps)` (or its last entry)
    against how many report steps the deck's schedule defines to know
    whether the run reached its final `DATES` entry.
    """

    last_simulated_date: str | None
    """ISO date of the last completed report step, or `None` if no
    report step completed at all.
    """

    newton_iterations: WastedWork | None
    """Overall Newton iteration count and wasted fraction, or `None` if
    the run ended before printing a final summary (for example, a crash).
    """

    has_negative_state_warning: bool
    """Whether any negative-saturation or negative-pressure warning was
    seen anywhere in the file.
    """

    has_fatal_timestep_warning: bool
    """Whether a "too small a timestep" warning was seen anywhere in the file."""

    @property
    def is_clean(self) -> bool:
        """Whether this run is worth trusting without a closer look.

        A run is clean when it reported zero fatal errors and bugs, saw
        no negative-saturation/pressure or fatal-timestep warning, and
        completed at least one report step. This does not by itself mean
        the run reached its final `DATES` entry; check
        `completed_report_steps` against the deck's schedule for that.
        """
        return (
            self.errors == 0
            and self.bugs == 0
            and not self.has_negative_state_warning
            and not self.has_fatal_timestep_warning
            and len(self.completed_report_steps) > 0
        )


def parse(prt_path: pathlib.Path | str) -> PrtReport:
    """Parse a `.PRT` file into a `PrtReport`.

    :raises FileNotFoundError: if `prt_path` does not exist.
    """
    prt_path = pathlib.Path(prt_path)
    text = prt_path.read_text(encoding="utf-8", errors="ignore")

    unconverged_well_counts: dict[str, int] = {}
    for match in UNCONVERGED_WELL_PATTERN.finditer(text):
        well = match["well"]
        unconverged_well_counts[well] = unconverged_well_counts.get(well, 0) + 1

    step_matches = list(REPORT_STEP_PATTERN.finditer(text))
    completed_report_steps = [int(match["step"]) for match in step_matches]
    last_simulated_date = step_matches[-1]["date"] if step_matches else None

    summary_match = ERROR_SUMMARY_PATTERN.search(text)
    warnings = int(summary_match["warnings"]) if summary_match else len(unconverged_well_counts)
    errors = int(summary_match["errors"]) if summary_match else 0
    bugs = int(summary_match["bugs"]) if summary_match else 0
    problems = int(summary_match["problems"]) if summary_match else 0

    newton_iterations = None
    for match in OVERALL_ITERATIONS_PATTERN.finditer(text):
        if match["kind"] == "Newton Iterations":
            newton_iterations = WastedWork(
                total=int(match["total"]),
                wasted=int(match["wasted"]),
                wasted_percent=float(match["wasted_pct"]),
            )

    return PrtReport(
        path=prt_path,
        warnings=warnings,
        errors=errors,
        bugs=bugs,
        problems=problems,
        unconverged_well_counts=unconverged_well_counts,
        completed_report_steps=completed_report_steps,
        last_simulated_date=last_simulated_date,
        newton_iterations=newton_iterations,
        has_negative_state_warning=bool(NEGATIVE_STATE_PATTERN.search(text)),
        has_fatal_timestep_warning=bool(TOO_SMALL_TIMESTEP_PATTERN.search(text)),
    )

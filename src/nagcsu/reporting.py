"""Writing a human-readable summary of a run or a tuning session.

This is the "generate a summary file on how it was gotten" piece: given
a run record (or a whole `nagcsu match auto` session's records), produce
one Markdown file documenting the final parameter state, how J got
there, and what to look at next, in roughly the shape Stage E.2 of the
Execution Plan's methods-section template asks for.
"""

import pathlib
import typing

from nagcsu import ledger
from nagcsu.algorithms.coordinate_descent import GroupOutcome
from nagcsu.algorithms.sensitivity import SensitivityResult
from nagcsu.prt import PrtReport


def render_run_report(
    record: ledger.RunRecord,
    *,
    prt_report: PrtReport | None = None,
    group_outcomes: list[GroupOutcome] | None = None,
    sensitivity_results: list[SensitivityResult] | None = None,
) -> str:
    """Render a single run's snapshot as a Markdown document.

    :param record: The run to report on.
    :param prt_report: That run's parsed `.PRT` health, if available.
    :param group_outcomes: `nagcsu.algorithms.coordinate_descent.GroupOutcome`
        entries, if `record` came from an auto-tune session, in the
        order the groups were tuned.
    :param sensitivity_results: Ranked sensitivity results to suggest
        what to try next, if a sensitivity pass was run alongside this record.
    """
    lines: list[str] = []
    lines.append(f"# History match snapshot: {record.run_id}")
    lines.append("")
    lines.append(f"Generated from run `{record.run_id}`, logged {record.created_at}.")
    if record.note:
        lines.append("")
        lines.append(record.note)

    lines.append("")
    lines.append("## Objective")
    lines.append("")
    if record.j is None:
        lines.append("This run was not scored against the observed history.")
    else:
        lines.append(f"J = **{record.j:.4f}**")
        lines.append("")
        lines.append("| Vector | NRMSE |")
        lines.append("| --- | --- |")
        for name, value in (record.vector_nrmse or {}).items():
            lines.append(f"| {name} | {value:.4f} |")

    if group_outcomes:
        lines.append("")
        lines.append("## How this was found: tuning path")
        lines.append("")
        lines.append("| Group | Starting J | Ending J | Reached target |")
        lines.append("| --- | --- | --- | --- |")
        for outcome in group_outcomes:
            lines.append(
                f"| {outcome.group} | {outcome.starting_j:.4f} | {outcome.ending_j:.4f} | "
                f"{'yes' if outcome.reached_target else 'no'} |"
            )
        lines.append("")
        lines.append(
            "Groups were tuned in priority order (Stage D.1 of the Execution Plan); "
            "a group not listed here was never touched because an earlier group "
            "already reached the target."
        )

    lines.append("")
    lines.append("## Final parameter state")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("| --- | --- |")
    for name, value in sorted(record.parameter_state.items()):
        lines.append(f"| {name} | {value:.6g} |")

    if prt_report is not None:
        lines.append("")
        lines.append("## Run health")
        lines.append("")
        lines.append(f"- Report steps completed: {len(prt_report.completed_report_steps)}")
        lines.append(f"- Last simulated date: {prt_report.last_simulated_date}")
        lines.append(
            f"- Errors: {prt_report.errors}, Bugs: {prt_report.bugs}, Warnings: {prt_report.warnings}"
        )
        if prt_report.unconverged_well_counts:
            worst = sorted(prt_report.unconverged_well_counts.items(), key=lambda item: -item[1])[
                :3
            ]
            worst_text = ", ".join(f"{well} ({count})" for well, count in worst)
            lines.append(f"- Wells with the most convergence warnings: {worst_text}")
        lines.append(f"- Considered clean: {'yes' if prt_report.is_clean else 'no'}")

    if sensitivity_results:
        lines.append("")
        lines.append("## What to try next")
        lines.append("")
        lines.append(
            "Ranked by local sensitivity (how much J moved when this parameter "
            "alone was perturbed, others held fixed):"
        )
        lines.append("")
        lines.append("| Parameter | Swing in J |")
        lines.append("| --- | --- |")
        for result in sensitivity_results[:5]:
            lines.append(f"| {result.parameter} | {result.swing:.4f} |")

    lines.append("")
    return "\n".join(lines)


def write_run_report(
    record: ledger.RunRecord,
    output_path: pathlib.Path | str,
    **kwargs: typing.Any,
) -> pathlib.Path:
    """Render and write a run report; see :func:`render_run_report` for `kwargs`."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_run_report(record, **kwargs), encoding="utf-8")
    return output_path

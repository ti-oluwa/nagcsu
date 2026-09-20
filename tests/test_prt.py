"""Tests for `nagcsu.prt` against the project's real baseline `.PRT`."""

import pathlib

from nagcsu import prt


def test_parse_reads_the_error_summary_block(sample_prt_path: pathlib.Path) -> None:
    report = prt.parse(sample_prt_path)
    assert report.errors == 0
    assert report.bugs == 0
    assert report.warnings > 0
    assert report.problems > 0


def test_parse_finds_every_completed_report_step(sample_prt_path: pathlib.Path) -> None:
    report = prt.parse(sample_prt_path)
    assert report.completed_report_steps == list(range(1, 13))
    assert report.last_simulated_date == "2026-01-01"


def test_parse_tallies_unconverged_wells(sample_prt_path: pathlib.Path) -> None:
    report = prt.parse(sample_prt_path)
    assert sum(report.unconverged_well_counts.values()) > 0
    assert sum(report.unconverged_well_counts.values()) <= report.problems
    assert "EVWRENI" in report.unconverged_well_counts
    assert report.unconverged_well_counts["EVWRENI"] > 0


def test_is_clean_true_for_the_baseline_run_despite_warnings(
    sample_prt_path: pathlib.Path,
) -> None:
    report = prt.parse(sample_prt_path)
    # The baseline run has hundreds of well-convergence warnings but zero
    # fatal errors/bugs and completed every report step, so it counts as
    # clean: "clean" means trustworthy enough to score, not warning-free.
    assert report.is_clean


def test_newton_iteration_summary_is_parsed(sample_prt_path: pathlib.Path) -> None:
    report = prt.parse(sample_prt_path)
    assert report.newton_iterations is not None
    assert report.newton_iterations.total > 0
    assert 0 <= report.newton_iterations.wasted_percent <= 100

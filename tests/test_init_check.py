"""Tests for `nagcsu.init_check`."""

import pathlib

import numpy as np
import pytest
import resfo

from nagcsu import init_check


def write_unrst(path: pathlib.Path, *, swat_by_seqnum: list[list[float]]) -> None:
    """Write a minimal `.UNRST` fixture with one `SWAT` array per `SEQNUM`."""
    contents: list[tuple[str, np.ndarray]] = []
    for seqnum, swat in enumerate(swat_by_seqnum):
        contents.append(("SEQNUM  ", np.array([seqnum], dtype=np.int32)))
        contents.append(("SWAT    ", np.array(swat, dtype=np.float32)))
    resfo.write(path, contents)


def test_read_initial_water_saturation_returns_the_first_swat_array(
    tmp_path: pathlib.Path,
) -> None:
    case_basename = tmp_path / "CASE"
    write_unrst(
        case_basename.with_suffix(".UNRST"),
        swat_by_seqnum=[[0.13, 0.14, 0.15, 0.20], [0.30, 0.31, 0.32, 0.40]],
    )

    swat = init_check.read_initial_water_saturation(case_basename)

    assert swat.tolist() == pytest.approx([0.13, 0.14, 0.15, 0.20], abs=1e-6)


def test_read_initial_water_saturation_raises_when_unrst_missing(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        init_check.read_initial_water_saturation(tmp_path / "MISSING")


def test_read_initial_water_saturation_raises_when_no_swat_array(tmp_path: pathlib.Path) -> None:
    case_basename = tmp_path / "CASE"
    resfo.write(case_basename.with_suffix(".UNRST"), [("SEQNUM  ", np.array([0], dtype=np.int32))])

    with pytest.raises(ValueError, match="No SWAT array"):
        init_check.read_initial_water_saturation(case_basename)


def test_check_initial_water_saturation_is_plausible_near_connate(tmp_path: pathlib.Path) -> None:
    case_basename = tmp_path / "CASE"
    write_unrst(case_basename.with_suffix(".UNRST"), swat_by_seqnum=[[0.13, 0.14, 0.15, 0.16]])

    report = init_check.check_initial_water_saturation(
        case_basename, expected_water_saturation=0.13, tolerance=0.05
    )

    assert report.cell_count == 4
    assert report.mean_water_saturation == pytest.approx(0.145, abs=1e-6)
    assert report.is_plausible


def test_check_initial_water_saturation_flags_an_already_watered_out_start(
    tmp_path: pathlib.Path,
) -> None:
    # This is the shape of the bug this check exists to catch: SWAT
    # initialized near (1 - Sorw) instead of near Swc, everywhere.
    case_basename = tmp_path / "CASE"
    write_unrst(case_basename.with_suffix(".UNRST"), swat_by_seqnum=[[0.88, 0.88, 0.88, 0.88]])

    report = init_check.check_initial_water_saturation(
        case_basename, expected_water_saturation=0.13, tolerance=0.05
    )

    assert not report.is_plausible

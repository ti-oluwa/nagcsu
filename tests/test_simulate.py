"""Tests for `nagcsu.simulate`, especially the Docker-wrapper-safe
`cwd`/relative-path handling `run()` does: a wrapper script that only
mounts its own working directory into a container needs to be launched
from a directory that actually contains both the deck and the output
directory, with both passed as relative paths.
"""

import pathlib
import stat
import unittest.mock

import pytest

from nagcsu import simulate


def test_common_working_directory_finds_shared_ancestor(tmp_path) -> None:
    deck = tmp_path / "Data" / "deck.DATA"
    output_dir = tmp_path / "runs" / "run_0001"
    assert simulate.get_common_working_directory([deck, output_dir]) == tmp_path


def test_get_common_working_directory_falls_back_when_no_ancestor_is_shared() -> None:
    # Different drive roots on Windows, or otherwise unrelated paths:
    # commonpath raises ValueError, so fall back to the first path's parent.
    with unittest.mock.patch("os.path.commonpath", side_effect=ValueError):
        result = simulate.get_common_working_directory([
            pathlib.Path("/a/b/deck.DATA"),
            pathlib.Path("/c/d/output"),
        ])
    assert result == pathlib.Path("/a/b")


def test_format_extra_mounts_env_uses_semicolons_on_windows() -> None:
    with unittest.mock.patch("platform.system", return_value="Windows"):
        result = simulate.format_extra_mounts_env(["C:\\data\\a=/data/a", "C:\\data\\b"])
    assert result == "C:\\data\\a=/data/a;C:\\data\\b"


def test_format_extra_mounts_env_uses_colons_elsewhere() -> None:
    with unittest.mock.patch("platform.system", return_value="Linux"):
        result = simulate.format_extra_mounts_env(["/data/shared", "/data/pvt=/mnt/pvt"])
    assert result == "/data/shared:/data/pvt=/mnt/pvt"


def test_text_output_io_close_leaves_destinations_open() -> None:
    destination = simulate.io.StringIO()
    output = simulate.TextOutputIO([destination])
    output.write("captured")

    output.close()

    assert output.closed
    assert destination.getvalue() == "captured"
    assert not destination.closed
    with pytest.raises(ValueError, match="closed file"):
        output.write("after close")


@pytest.fixture
def fake_flow_executable(tmp_path) -> pathlib.Path:
    """An executable script standing in for `flow`: reports its cwd,
    argv and `OPM_FLOW_EXTRA_MOUNTS`, then writes a minimal
    `.UNSMRY`/`.PRT` pair so `simulate.run` sees a normal exit.
    """
    script = tmp_path / "fake_flow"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, pathlib\n"
        "print('CWD:' + os.getcwd())\n"
        "print('FLOW STDERR', file=sys.stderr)\n"
        "print('ARGS:' + '|'.join(sys.argv[1:]))\n"
        "print('MOUNTS:' + os.environ.get('OPM_FLOW_EXTRA_MOUNTS', '<unset>'))\n"
        "for arg in sys.argv[1:]:\n"
        "    if arg.startswith('--output-dir='):\n"
        "        out = pathlib.Path(arg.split('=', 1)[1])\n"
        "        out.mkdir(parents=True, exist_ok=True)\n"
        "        (out / 'FAKECASE.UNSMRY').touch()\n"
        "        (out / 'FAKECASE.PRT').touch()\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_run_launches_flow_from_the_common_directory_with_relative_paths(
    tmp_path, fake_flow_executable: pathlib.Path
) -> None:
    deck_path = tmp_path / "Data" / "deck.DATA"
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    deck_path.write_text("dummy deck")
    output_dir = tmp_path / "runs" / "run_0001"

    result = simulate.run(
        deck_path,
        output_dir,
        flow_executable=str(fake_flow_executable),
        extra_mounts=["/data/shared"],
    )

    assert f"CWD:{tmp_path.resolve()}" in result.stdout
    args_line = next(line for line in result.stdout.splitlines() if line.startswith("ARGS:"))
    args = args_line[len("ARGS:") :].split("|")
    assert args[0] in ("Data/deck.DATA", "Data\\deck.DATA")
    assert args[1] in ("--output-dir=runs/run_0001", "--output-dir=runs\\run_0001")
    assert "MOUNTS:/data/shared" in result.stdout
    assert result.case_basename is not None
    assert result.returncode == 0


def test_run_omits_extra_mounts_env_when_none_given(
    tmp_path, fake_flow_executable: pathlib.Path
) -> None:
    deck_path = tmp_path / "deck.DATA"
    deck_path.write_text("dummy deck")
    output_dir = tmp_path / "runs" / "run_0001"

    result = simulate.run(deck_path, output_dir, flow_executable=str(fake_flow_executable))

    assert "MOUNTS:<unset>" in result.stdout


def test_run_captures_and_forwards_both_output_streams(
    tmp_path, fake_flow_executable: pathlib.Path, capsys
) -> None:
    deck_path = tmp_path / "deck.DATA"
    deck_path.write_text("dummy deck")
    output_dir = tmp_path / "runs" / "run_0001"

    result = simulate.run(deck_path, output_dir, flow_executable=str(fake_flow_executable))
    console_output = capsys.readouterr()

    assert "CWD:" in result.stdout
    assert "FLOW STDERR" in result.stderr
    assert "CWD:" in console_output.out
    assert "FLOW STDERR" in console_output.err


def get_flow_args(result: simulate.RunResult) -> list[str]:
    args_line = next(line for line in result.stdout.splitlines() if line.startswith("ARGS:"))
    return args_line[len("ARGS:") :].split("|")


def test_run_passes_the_default_thread_count(tmp_path, fake_flow_executable: pathlib.Path) -> None:
    deck_path = tmp_path / "deck.DATA"
    deck_path.write_text("dummy deck")

    result = simulate.run(
        deck_path, tmp_path / "runs" / "run_0001", flow_executable=str(fake_flow_executable)
    )

    assert "--threads-per-process=8" in get_flow_args(result)


def test_run_uses_the_requested_thread_count_and_appends_extra_args(
    tmp_path, fake_flow_executable: pathlib.Path
) -> None:
    deck_path = tmp_path / "deck.DATA"
    deck_path.write_text("dummy deck")

    result = simulate.run(
        deck_path,
        tmp_path / "runs" / "run_0001",
        flow_executable=str(fake_flow_executable),
        threads_per_process=4,
        extra_args=["--enable-tuning=true", "--solver-max-time-step-in-days=30"],
    )
    args = get_flow_args(result)

    assert "--threads-per-process=4" in args
    assert "--threads-per-process=8" not in args
    assert args[-2:] == ["--enable-tuning=true", "--solver-max-time-step-in-days=30"]


def test_run_omits_the_thread_option_when_threads_is_none(
    tmp_path, fake_flow_executable: pathlib.Path
) -> None:
    deck_path = tmp_path / "deck.DATA"
    deck_path.write_text("dummy deck")

    result = simulate.run(
        deck_path,
        tmp_path / "runs" / "run_0001",
        flow_executable=str(fake_flow_executable),
        threads_per_process=None,
    )

    assert not any(arg.startswith("--threads-per-process") for arg in get_flow_args(result))


def test_run_lets_an_explicit_thread_flag_in_extra_args_win(
    tmp_path, fake_flow_executable: pathlib.Path
) -> None:
    deck_path = tmp_path / "deck.DATA"
    deck_path.write_text("dummy deck")

    result = simulate.run(
        deck_path,
        tmp_path / "runs" / "run_0001",
        flow_executable=str(fake_flow_executable),
        threads_per_process=8,
        extra_args=["--threads-per-process=2"],
    )
    thread_args = [arg for arg in get_flow_args(result) if arg.startswith("--threads-per-process")]

    assert thread_args == ["--threads-per-process=2"]

"""Tests for `nagcsu.config`."""

import pathlib

import pytest

from nagcsu import config


def test_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        config.load(tmp_path / "nagcsu.yaml")


def test_save_and_load_round_trip(tmp_path) -> None:
    original = config.ProjectConfig(root=tmp_path)
    config_path = tmp_path / "nagcsu.yaml"
    config.save(original, config_path)

    loaded = config.load(config_path)
    assert loaded.deck_path == original.deck_path
    assert loaded.wells == original.wells
    assert loaded.objective.weights == original.objective.weights
    assert loaded.objective.target_j == pytest.approx(original.objective.target_j)


def test_validate_rejects_weights_not_summing_to_one() -> None:
    bad = config.ProjectConfig(
        objective=config.ObjectiveConfig(weights={"pressure": 0.5, "watercut": 0.2, "gor": 0.1})
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_resolved_path_joins_relative_paths_onto_root(tmp_path) -> None:
    project_config = config.ProjectConfig(root=tmp_path)
    resolved = project_config.get_resolved_path(pathlib.Path("Data/deck.DATA"))
    assert resolved == (tmp_path / "Data/deck.DATA").resolve()


def test_resolved_path_leaves_absolute_paths_unchanged(tmp_path) -> None:
    project_config = config.ProjectConfig(root=tmp_path)
    absolute = tmp_path.resolve() / "elsewhere" / "deck.DATA"
    assert project_config.get_resolved_path(absolute) == absolute


def test_save_and_load_round_trips_extra_mounts_and_history_fields(tmp_path) -> None:
    original = config.ProjectConfig(
        root=tmp_path,
        extra_mounts=["/data/shared", "/data/pvt=/mnt/pvt"],
        history=config.HistoryConfig(
            file_format="csv",
            well_column="Field",
            date_column="Date",
        ),
    )
    config_path = tmp_path / "nagcsu.yaml"
    config.save(original, config_path)

    loaded = config.load(config_path)
    assert loaded.extra_mounts == ["/data/shared", "/data/pvt=/mnt/pvt"]
    assert loaded.history.file_format == "csv"
    assert loaded.history.well_column == "Field"
    assert loaded.history.date_column == "Date"


def test_extra_mounts_defaults_to_an_empty_list(tmp_path) -> None:
    config_path = tmp_path / "nagcsu.yaml"
    config.save(config.ProjectConfig(root=tmp_path), config_path)
    loaded = config.load(config_path)
    assert loaded.extra_mounts == []


def test_history_well_column_and_file_format_default_to_none(tmp_path) -> None:
    config_path = tmp_path / "nagcsu.yaml"
    config.save(config.ProjectConfig(root=tmp_path), config_path)
    loaded = config.load(config_path)
    assert loaded.history.well_column is None
    assert loaded.history.file_format is None


def test_save_and_load_round_trips_threads_and_extra_args(tmp_path) -> None:
    original = config.ProjectConfig(
        root=tmp_path,
        threads_per_process=4,
        extra_args=["--enable-tuning=true", "--solver-max-time-step-in-days=30"],
    )
    config_path = tmp_path / "nagcsu.yaml"
    config.save(original, config_path)

    loaded = config.load(config_path)

    assert loaded.threads_per_process == 4
    assert loaded.extra_args == ["--enable-tuning=true", "--solver-max-time-step-in-days=30"]


def test_threads_and_extra_args_default_to_eight_threads_and_no_extra_args(tmp_path) -> None:
    config_path = tmp_path / "nagcsu.yaml"
    config_path.write_text("deck_path: Data/deck.DATA\n")

    loaded = config.load(config_path)

    assert loaded.threads_per_process == 8
    assert loaded.extra_args == []


def test_threads_per_process_null_in_yaml_means_use_flows_default(tmp_path) -> None:
    config_path = tmp_path / "nagcsu.yaml"
    config_path.write_text("threads_per_process: null\n")

    assert config.load(config_path).threads_per_process is None


def test_load_rejects_a_thread_count_below_one(tmp_path) -> None:
    config_path = tmp_path / "nagcsu.yaml"
    config_path.write_text("threads_per_process: 0\n")

    with pytest.raises(ValueError, match="threads_per_process"):
        config.load(config_path)

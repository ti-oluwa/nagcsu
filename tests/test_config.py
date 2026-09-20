"""Tests for `nagcsu.config`."""

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
    bad = config.ProjectConfig(objective=config.ObjectiveConfig(weights={"pressure": 0.5, "watercut": 0.2, "gor": 0.1}))
    with pytest.raises(ValueError):
        bad.validate()


def test_resolved_path_joins_relative_paths_onto_root(tmp_path) -> None:
    project_config = config.ProjectConfig(root=tmp_path)
    resolved = project_config.resolved_path("Data/deck.DATA")
    assert resolved == (tmp_path / "Data/deck.DATA").resolve()


def test_resolved_path_leaves_absolute_paths_unchanged(tmp_path) -> None:
    project_config = config.ProjectConfig(root=tmp_path)
    absolute = tmp_path.resolve() / "elsewhere" / "deck.DATA"
    assert project_config.resolved_path(absolute) == absolute

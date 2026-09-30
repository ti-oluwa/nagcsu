"""Out-of-bounds ranges and starting a search from a previous run."""

import click
import click.testing
import pytest
import yaml

from nagcsu import config as config_module
from nagcsu import ledger, parameters, ranges, reporting
from nagcsu.cli import context
from nagcsu.cli.app import cli
from nagcsu.cli.commands import match as match_module
from nagcsu.deck import Deck


def _record(run_id, *, j=1.0, state=None, parameter=None, value=1.0, error=None):
    return ledger.RunRecord(
        run_id=run_id,
        created_at="t",
        parameter_state=state if state is not None else {},
        group="aquifer" if parameter else None,
        strategy=None,
        j=j,
        vector_nrmse=None,
        prt_is_clean=True,
        note="",
        simulation_error=error,
        tuned_parameters=[parameter] if parameter else [],
        tuned_values={parameter: value} if parameter else {},
    )


# ------------------------------------------------------------ physical limits


def test_every_parameter_has_limits_that_contain_its_bounds_and_default() -> None:
    assert set(parameters.PHYSICAL_LIMITS) == set(parameters.PARAMETERS)
    for name, spec in parameters.PARAMETERS.items():
        low, high = parameters.get_physical_limits(name)
        assert low <= spec.bounds[0] < spec.bounds[1] <= high, name
        assert low <= spec.default <= high, name


def test_bound_violation_reports_the_side() -> None:
    low, high = parameters.PARAMETERS["aquifer.radius"].bounds
    assert parameters.get_bound_violation("aquifer.radius", low - 1) == "below"
    assert parameters.get_bound_violation("aquifer.radius", high + 1) == "above"
    assert parameters.get_bound_violation("aquifer.radius", (low + high) / 2) is None


def test_range_may_go_past_registered_bounds_but_not_physical_limits() -> None:
    _, high = parameters.PARAMETERS["aquifer.radius"].bounds
    assert (
        match_module.parse_range_options([f"aquifer.radius=10000:{high * 2:.0f}"])[
            "aquifer.radius"
        ][1]
        == high * 2
    )
    with pytest.raises(click.BadParameter):
        match_module.parse_range_options(["aquifer.radius=10000:99999999"])
    with pytest.raises(click.BadParameter):
        match_module.parse_range_options(["sgof.sorg=-0.1:0.3"])


def test_start_and_sweep_values_follow_physical_limits_not_bounds() -> None:
    _, high = parameters.PARAMETERS["aquifer.radius"].bounds
    assert match_module.parse_start_options([f"aquifer.radius={high * 1.5:.0f}"])
    with pytest.raises(click.BadParameter):
        match_module.parse_start_options(["aquifer.radius=0"])
    match_module.check_values_against_limits("aquifer.radius", [high * 1.2])
    with pytest.raises(click.BadParameter):
        match_module.check_values_against_limits("aquifer.radius", [-5.0])


def test_range_suggestions_extend_past_the_registered_bound_when_the_best_is_there() -> None:
    low, high = parameters.PARAMETERS["aquifer.radius"].bounds
    step = (high - low) / 6
    points = {
        low + i * step: 1.0 - 0.15 * i for i in range(7)
    }  # best at the registered upper bound
    records = [
        _record(f"r{i}", j=j, parameter="aquifer.radius", value=v)
        for i, (v, j) in enumerate(points.items())
    ]
    (suggestion,) = ranges.suggest_ranges(records)
    assert suggestion.status == "open_high"
    assert suggestion.high > high
    assert "past the registered bounds" in suggestion.note
    assert suggestion.high <= parameters.get_physical_limits("aquifer.radius")[1]


def test_widen_bounds_for_start_values_keeps_explicit_ranges_untouched() -> None:
    spaces = {"aquifer": {"aquifer.radius": (8000.0, 30000.0), "aquifer.thickness": (20.0, 200.0)}}
    start = {"aquifer.radius": 45000.0, "aquifer.thickness": 77.0}
    widened = match_module.widen_bounds_for_start_values(spaces, start, explicit_ranges={})
    assert widened["aquifer"]["aquifer.radius"] == (8000.0, 45000.0)
    assert widened["aquifer"]["aquifer.thickness"] == (20.0, 200.0)
    kept = match_module.widen_bounds_for_start_values(
        spaces, start, explicit_ranges={"aquifer.radius": (8000.0, 30000.0)}
    )
    assert kept["aquifer"]["aquifer.radius"] == (8000.0, 30000.0)


def test_state_rows_flag_values_beyond_the_registered_bounds() -> None:
    _, high = parameters.PARAMETERS["aquifer.radius"].bounds
    state = {**parameters.default_state(), "aquifer.radius": high * 1.3}
    row = next(r for r in reporting.get_state_rows(state) if r.name == "aquifer.radius")
    assert row.beyond == "above" and row.pinned is None


# ------------------------------------------------------------ run ids


def test_unique_run_id_never_overwrites_an_earlier_session(tmp_path) -> None:
    (tmp_path / "auto_final").mkdir()
    assert match_module.unique_run_id("auto_final", ["x"], output_root=tmp_path) == "auto_final2"
    assert (
        match_module.unique_run_id(
            "auto_final", ["auto_final", "auto_final2"], output_root=tmp_path
        )
        == "auto_final3"
    )
    assert match_module.unique_run_id("fresh_final", [], output_root=tmp_path) == "fresh_final"


# ------------------------------------------------------------ baseline


@pytest.fixture
def project(tmp_path, sample_deck: Deck):
    project_config = config_module.ProjectConfig(
        deck_path=sample_deck.path,
        output_root=tmp_path / "runs",
        ledger_path=tmp_path / "runs" / "ledger.json",
        root=tmp_path,
    )
    config_path = tmp_path / "nagcsu.yaml"
    config_module.save(project_config, config_path)
    ledger_path = tmp_path / "runs" / "ledger.json"
    good = {**parameters.default_state(), "aquifer.radius": 12000.0}
    better = {**parameters.default_state(), "aquifer.radius": 9500.0, "sgof.sorg": 0.25}
    for record in (
        _record("run_0000", j=0.9, state=parameters.default_state()),
        _record("run_0001", j=0.5, state=good),
        _record("run_0002", j=0.3, state=better),
        _record("run_0003", j=None, state=good, error="boom"),
    ):
        ledger.append(ledger_path, record)
    return project_config, config_path, Deck.load(sample_deck.path), tmp_path


def test_baseline_default_is_the_registered_defaults(project) -> None:
    project_config, _, deck, _ = project
    baseline = context.resolve_baseline(project_config, deck)
    assert baseline.state == parameters.default_state()
    assert baseline.run_id is None and baseline.ledger_label() == "default"
    assert baseline.deck_source == "project"


def test_baseline_best_latest_and_run_id(project) -> None:
    project_config, _, deck, _ = project
    best = context.resolve_baseline(project_config, deck, baseline_raw="best")
    assert best.run_id == "run_0002" and best.state["aquifer.radius"] == 9500.0 and best.j == 0.3
    latest = context.resolve_baseline(project_config, deck, baseline_raw="latest")
    assert latest.run_id == "run_0002"  # run_0003 failed, so it is not usable
    chosen = context.resolve_baseline(project_config, deck, baseline_raw="run_0001")
    assert chosen.state["aquifer.radius"] == 12000.0 and chosen.ledger_label() == "run:run_0001"


def test_baseline_unknown_run_suggests_close_matches(project) -> None:
    project_config, _, deck, _ = project
    with pytest.raises(click.ClickException) as error:
        context.resolve_baseline(project_config, deck, baseline_raw="run_0022")
    assert "run_0002" in str(error.value.message)


def test_baseline_from_config_and_override(project) -> None:
    import dataclasses

    project_config, _, deck, _ = project
    configured = dataclasses.replace(project_config, baseline="best")
    assert context.resolve_baseline(configured, deck).run_id == "run_0002"
    assert context.resolve_baseline(configured, deck, baseline_raw="default").run_id is None


def test_baseline_from_a_parameters_snapshot_file(project, tmp_path) -> None:
    project_config, _, deck, _ = project
    snapshot = tmp_path / "parameters.yaml"
    snapshot.write_text(yaml.safe_dump({"sgof.sorg": 0.22}))
    baseline = context.resolve_baseline(project_config, deck, baseline_raw=str(snapshot))
    assert baseline.state["sgof.sorg"] == 0.22
    assert baseline.state["aquifer.radius"] == parameters.PARAMETERS["aquifer.radius"].default
    snapshot.write_text(yaml.safe_dump({"nonsense": 1}))
    with pytest.raises(click.ClickException):
        context.resolve_baseline(project_config, deck, baseline_raw=str(snapshot))


def test_baseline_deck_auto_uses_the_run_deck_when_it_exists(project) -> None:
    project_config, _, deck, tmp_path = project
    run_dir = tmp_path / "runs" / "run_0002"
    run_dir.mkdir(parents=True)
    (run_dir / project_config.deck_path.name).write_text(deck.path.read_text())
    used = context.resolve_baseline(project_config, deck, baseline_raw="best")
    assert used.deck_source == "run:run_0002"
    forced = context.resolve_baseline(
        project_config, deck, baseline_raw="best", base_deck_raw="project"
    )
    assert forced.deck_source == "project"
    missing = context.resolve_baseline(project_config, deck, baseline_raw="run_0001")
    assert missing.deck_source == "project"  # that run's deck is not on disk
    with pytest.raises(click.ClickException):
        context.resolve_baseline(
            project_config, deck, baseline_raw="best", base_deck_raw="/nope.DATA"
        )


def test_config_round_trips_the_baseline_and_rejects_an_empty_one(project) -> None:
    import dataclasses

    project_config, config_path, _, _ = project
    config_module.save(dataclasses.replace(project_config, baseline="best"), config_path)
    assert config_module.load(config_path).baseline == "best"
    with pytest.raises(ValueError):
        dataclasses.replace(project_config, baseline=" ").validate()


def test_sweep_starts_from_the_baseline_state_and_records_it(project, monkeypatch) -> None:
    _, config_path, _, _tmp_path = project
    seen: dict = {}

    def fake_make_evaluate(config, base_deck, *, run_id_prefix, on_outcome):
        seen["deck"] = base_deck

        def evaluate(state):
            seen.setdefault("states", []).append(dict(state))
            return 1.0 / (1.0 + state["sgof.oil_exponent"])

        return evaluate

    monkeypatch.setattr(match_module.pipeline, "make_evaluate", fake_make_evaluate)
    result = click.testing.CliRunner().invoke(
        cli,
        [
            "--config",
            str(config_path),
            "match",
            "sweep",
            "--param",
            "sgof.oil_exponent",
            "--values",
            "2,3.5",
            "--baseline",
            "best",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Baseline: best -> run run_0002" in result.output
    assert all(
        state["aquifer.radius"] == 9500.0 and state["sgof.sorg"] == 0.25
        for state in seen["states"]
    )
    assert [state["sgof.oil_exponent"] for state in seen["states"]] == [2.0, 3.5]

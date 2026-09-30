"""Per-well scoring, well selection, reporting and plots."""

import dataclasses

import click
import pandas
import pytest

from nagcsu import config as config_module
from nagcsu import history, ledger, objective, plotting, reporting
from nagcsu.cli import context
from nagcsu.exceptions import HistoryAlignmentError

DECK_WELLS = ("AFIESERE", "OLOMORO")


def _long_table() -> pandas.DataFrame:
    dates = pandas.date_range("2000-01-01", periods=4, freq="MS")
    rows = []
    for name, start in (("Afiesere", 0), ("Olomoro-Oleh", 1)):
        for index, date in enumerate(dates):
            if index < start:
                continue  # phased start: no rows before the well begins
            rows.append({
                "Field": name,
                "Date": date,
                "Oil_Rate_STBD": 100.0,
                "Gas_Rate_MMSCFD": 80.0 + index,
                "Water_Rate_STBD": 10.0 * (index + 1),
                "Reservoir_Pressure_psia": 2500.0 - 10 * index,
            })
    return pandas.DataFrame(rows)


def _frames() -> tuple[pandas.DataFrame, pandas.DataFrame]:
    raw = _long_table()
    mapping = history.detect_long_format_columns(list(raw.columns))
    assert mapping is not None
    observed = history.load_long_format(
        raw,
        date_column="Date",
        mapping=mapping,
        path=pandas.io.common.Path("x.xlsx"),
        wells=list(DECK_WELLS),
    )
    simulated = observed.copy()
    return simulated, observed


def test_match_well_name_handles_deck_spelling() -> None:
    assert history.match_well_name("Olomoro-Oleh", ["OLOMORO", "KOKORI"]) == "OLOMORO"
    assert history.match_well_name("Unknown", ["OLOMORO"]) is None


def test_long_format_adds_per_well_columns_with_nan_before_start() -> None:
    _, observed = _frames()
    assert {"WWCT:AFIESERE", "WGOR:AFIESERE", "WWCT:OLOMORO", "WGOR:OLOMORO"} <= set(
        observed.columns
    )
    assert observed["WWCT:OLOMORO"].isna().iloc[0]
    assert observed["WWCT:AFIESERE"].iloc[0] == pytest.approx(10.0 / 110.0)


def test_field_only_scoring_is_unchanged_and_wells_are_diagnostic() -> None:
    simulated, observed = _frames()
    simulated["WWCT:AFIESERE"] += 0.1
    result = objective.score(
        simulated, observed, weights={"pressure": 0.5, "watercut": 0.35, "gor": 0.15}
    )
    assert result.j == pytest.approx(0.0)
    assert result.scored_wells == ()
    assert result.well_scores["WWCT:AFIESERE"].nrmse > 0


def test_selected_wells_feed_j_through_the_aggregate_weights() -> None:
    simulated, observed = _frames()
    simulated["WWCT:AFIESERE"] += 0.1
    weights = {
        "pressure": 0.4,
        "watercut": 0.3,
        "gor": 0.1,
        "wells_watercut": 0.2,
        "wells_gor": 0.0,
    }
    only_afiesere = objective.score(simulated, observed, weights=weights, wells=["AFIESERE"])
    both = objective.score(simulated, observed, weights=weights, wells=["AFIESERE", "OLOMORO"])
    assert only_afiesere.j > 0
    assert both.vector_scores["wells_watercut"].nrmse == pytest.approx(
        only_afiesere.vector_scores["wells_watercut"].nrmse / 2
    )
    assert both.j == pytest.approx(only_afiesere.j / 2)


def test_per_well_weight_without_wells_is_an_error() -> None:
    simulated, observed = _frames()
    with pytest.raises(HistoryAlignmentError):
        objective.score(
            simulated, observed, weights={"pressure": 0.5, "watercut": 0.3, "wells_watercut": 0.2}
        )


def test_selected_well_missing_from_data_is_an_error() -> None:
    simulated, observed = _frames()
    observed = observed.drop(columns=["WWCT:OLOMORO"])
    with pytest.raises(HistoryAlignmentError):
        objective.score(
            simulated,
            observed,
            weights={"pressure": 0.5, "watercut": 0.3, "wells_watercut": 0.2},
            wells=["OLOMORO"],
        )


def test_config_rejects_unknown_wells_and_orphan_weights() -> None:
    config = config_module.ProjectConfig()
    config.objective.wells = ("NOPE",)
    with pytest.raises(ValueError):
        config.validate()
    config = config_module.ProjectConfig()
    config.objective.weights = {**config.objective.weights, "wells_watercut": 0.1}
    with pytest.raises(ValueError):
        config.validate()


def test_apply_objective_overrides_resolves_wells_and_guards_zero_weights() -> None:
    config = config_module.ProjectConfig()
    with pytest.raises(click.ClickException):
        context.apply_objective_overrides(config, wells_raw="all")
    updated = context.apply_objective_overrides(
        config,
        wells_raw="afiesere,Olomoro",
        weights_raw="pressure=0.4,watercut=0.3,gor=0.1,wells_watercut=0.15,wells_gor=0.05",
    )
    assert updated.objective.wells == ("AFIESERE", "OLOMORO")
    assert config.objective.wells == ()
    assert context.apply_objective_overrides(config, wells_raw="none").objective.wells == ()
    with pytest.raises(click.BadParameter):
        context.resolve_wells(config, "nowhere")


def _record() -> ledger.RunRecord:
    return ledger.RunRecord(
        run_id="r1",
        created_at="t",
        parameter_state={},
        group=None,
        strategy=None,
        j=0.2,
        vector_nrmse={"pressure": 0.1},
        prt_is_clean=True,
        note="",
        well_nrmse={"WWCT:AFIESERE": 0.3, "WGOR:AFIESERE": 0.1, "WWCT:OLOMORO": 0.05},
        scored_wells=["AFIESERE"],
    )


def test_report_lists_per_well_rows_and_can_hide_them() -> None:
    rows = reporting.get_well_rows(_record())
    assert [row.well for row in rows] == ["AFIESERE", "OLOMORO"]
    assert rows[0].in_objective and not rows[1].in_objective
    assert "## Per-well match" in reporting.render_run_report(_record())
    assert "## Per-well match" not in reporting.render_run_report(_record(), show_wells=False)


def test_plot_wells_match_builds_one_row_per_well() -> None:
    simulated, observed = _frames()
    figure = plotting.plot_wells_match(simulated, observed, list(DECK_WELLS))
    assert len(figure.data) == 8
    with pytest.raises(ValueError):
        plotting.plot_wells_match(simulated, observed, [])


def test_well_table_renders_and_is_none_without_data() -> None:
    from nagcsu.cli import display

    table = display.well_table(_record())
    assert table is not None and table.row_count == 2
    display.console.print(table)
    bare = dataclasses.replace(_record(), well_nrmse=None)
    assert display.well_table(bare) is None

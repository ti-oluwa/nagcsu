"""The table key: every referenced term is defined, and it appears where expected."""

import click.testing
import pytest

from nagcsu import config as config_module
from nagcsu import glossary, ledger, parameters, reporting
from nagcsu.cli import display
from nagcsu.cli.app import cli
from nagcsu.deck import Deck

TERM_GROUPS = {
    name: value
    for name, value in vars(glossary).items()
    if name.isupper()
    and not name.startswith("_")
    and name != "ENTRIES"
    and isinstance(value, tuple)
}


@pytest.mark.parametrize("name", sorted(TERM_GROUPS))
def test_every_term_in_every_group_is_defined(name: str) -> None:
    missing = [term for term in TERM_GROUPS[name] if term not in glossary.GLOSSARY]
    assert not missing, f"{name} references undefined terms {missing}"


def test_every_entry_says_what_it_is_and_what_it_tells_you() -> None:
    for entry in glossary.GLOSSARY.values():
        assert entry.meaning.strip() and entry.impact.strip(), entry.term
        assert "\u2014" not in entry.meaning + entry.impact  # no em dashes


def test_lookup_drops_duplicates_and_unknown_terms_and_keeps_order() -> None:
    entries = glossary.lookup(["Swing", "nonsense", "J", "Swing"])
    assert [entry.term for entry in entries] == ["Swing", "J"]
    assert glossary.render_markdown(["nonsense"]) == []
    assert glossary.render_markdown(["J"])[0] == "## Key"


def _record() -> ledger.RunRecord:
    return ledger.RunRecord(
        run_id="r1",
        created_at="t",
        parameter_state=parameters.default_state(),
        group=None,
        strategy=None,
        j=0.2,
        vector_nrmse={"pressure": 0.1, "watercut": 0.3},
        prt_is_clean=True,
        note="",
        objective_weights={"pressure": 0.5, "watercut": 0.35},
    )


def test_markdown_report_ends_with_a_key_covering_only_sections_present() -> None:
    text = reporting.render_run_report(_record())
    assert "## Key" in text
    assert "| NRMSE |" in text and "| Default |" in text
    assert "| Mean rank |" not in text  # no group ranking in this report


def test_terminal_key_can_be_switched_off(capsys) -> None:
    display.set_key_enabled(True)
    display.print_key(("J",))
    assert "Key" in capsys.readouterr().out
    display.set_key_enabled(False)
    try:
        display.print_key(("J",))
        assert capsys.readouterr().out == ""
    finally:
        display.set_key_enabled(True)


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
    ledger.append(tmp_path / "runs" / "ledger.json", _record())
    return config_path


def test_report_show_prints_a_key_and_no_key_flag_hides_it(project) -> None:
    runner = click.testing.CliRunner()
    shown = runner.invoke(cli, ["--config", str(project), "report", "show", "r1"])
    assert shown.exit_code == 0, shown.output
    assert "Key" in shown.output and "What it tells you" in shown.output
    hidden = runner.invoke(cli, ["--config", str(project), "--no-key", "report", "show", "r1"])
    assert hidden.exit_code == 0, hidden.output
    assert "What it tells you" not in hidden.output
    listed = runner.invoke(cli, ["--config", str(project), "report", "list"])
    assert "What it tells you" in listed.output

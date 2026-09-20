"""Project configuration for a nagcsu working directory.

A project config is a small YAML file (`nagcsu.yaml` by default) that
records where the deck, history workbook and run outputs live, plus the
objective weights and tuning target, so every CLI command can be run as
`nagcsu <command> ...` without repeating `--deck`, `--history` and
similar paths on every invocation.
"""

import dataclasses
import pathlib
import typing

import yaml

from nagcsu import constants

DEFAULT_CONFIG_FILENAME: typing.Final[str] = "nagcsu.yaml"
"""Filename a bare `nagcsu <command>` looks for in the current directory."""


@dataclasses.dataclass(slots=True)
class ObjectiveConfig:
    """Weights and stopping target for the combined mismatch score J."""

    weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(constants.DEFAULT_OBJECTIVE_WEIGHTS)
    )
    """NRMSE weight per scored vector, keyed by "pressure", "watercut"
    and "gor". Must sum to 1.0; :meth:`ProjectConfig.validate` checks this.
    """

    target_j: float = constants.DEFAULT_TARGET_J
    """J value at or below which tuning should stop (Stage C.4)."""


@dataclasses.dataclass(slots=True)
class HistoryConfig:
    """Where the synthetic production history lives and how to read it."""

    path: pathlib.Path = pathlib.Path("Data/NigerDelta Synthetic Production History.xlsx")
    """Path to the workbook holding the observed pressure/water-cut/GOR
    history, relative to the project root unless given as an absolute
    path.
    """

    sheet_name: str | int = 0
    """Worksheet to read, passed straight through to `pandas.read_excel`."""

    date_column: str = "DATE"
    """Name of the column holding the report date for each history row."""

    column_map: dict[str, str] | None = None
    """Optional explicit mapping from a res2df summary vector name (for
    example "FPR" or "WWCT:AFIESERE") to the workbook's column name for
    that same quantity. Leave unset to use
    :func:`nagcsu.history.guess_column_map`, which matches the
    DATE,FPR,WWCT_<WELL>,WGOR_<WELL> layout Stage B.2 of the Execution
    Plan builds the workbook in.
    """


@dataclasses.dataclass(slots=True)
class ProjectConfig:
    """Top level configuration for a single nagcsu working directory."""

    deck_path: pathlib.Path = pathlib.Path("Data/NigerDelta UGH1 Composite Field.DATA")
    """Path to the OPM Flow `.DATA` deck this project tunes."""

    output_root: pathlib.Path = pathlib.Path("runs")
    """Directory each simulation run gets its own numbered subdirectory
    under. Created on first use if it does not already exist.
    """

    ledger_path: pathlib.Path = pathlib.Path("runs/ledger.json")
    """Path to the JSON run ledger (see :mod:`nagcsu.ledger`)."""

    flow_executable: str = "flow"
    """Name or path of the OPM Flow executable to invoke for each run."""

    wells: tuple[str, ...] = constants.PRODUCER_WELLS
    """Producer well names to score and report on."""

    objective: ObjectiveConfig = dataclasses.field(default_factory=ObjectiveConfig)
    """Objective weighting and stopping target."""

    history: HistoryConfig = dataclasses.field(default_factory=HistoryConfig)
    """Observed history location and column mapping."""

    root: pathlib.Path = pathlib.Path(".")
    """Directory the config file was loaded from. Relative paths in every
    other field are resolved against this when :meth:`resolved_path` is
    called, so the project can be run from any working directory.
    """

    def resolved_path(self, path: pathlib.Path) -> pathlib.Path:
        """Return `path` resolved against this project's root directory.

        Absolute paths are returned unchanged; relative paths are joined
        onto :attr:`root`.
        """
        path = pathlib.Path(path)
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    def validate(self) -> None:
        """Raise `ValueError` if the config is internally inconsistent.

        Checks that the objective weights sum to 1.0 (within floating
        point tolerance) and that at least one well is configured. Does
        not check that any file on disk actually exists, since a config
        can legitimately be created before the deck is generated.
        """
        weight_total = sum(self.objective.weights.values())
        if abs(weight_total - 1.0) > 1e-6:
            raise ValueError(
                f"Objective weights must sum to 1.0, got {weight_total:.6f} "
                f"from {self.objective.weights!r}"
            )
        if not self.wells:
            raise ValueError("At least one well must be configured to score against")


def load(config_path: pathlib.Path | str = DEFAULT_CONFIG_FILENAME) -> ProjectConfig:
    """Load a `ProjectConfig` from a YAML file.

    Missing keys fall back to the field defaults above, so a project can
    start with a two-line YAML file and only add sections it wants to
    override.

    :raises FileNotFoundError: if `config_path` does not exist.
    :raises ValueError: if the loaded config fails :meth:`ProjectConfig.validate`.
    """
    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No project config found at {config_path}. Run `nagcsu init` first, "
            f"or pass --config to point at an existing one."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    objective_raw = raw.get("objective", {})
    objective = ObjectiveConfig(
        weights=dict(objective_raw.get("weights", constants.DEFAULT_OBJECTIVE_WEIGHTS)),
        target_j=float(objective_raw.get("target_j", constants.DEFAULT_TARGET_J)),
    )

    history_raw = raw.get("history", {})
    history = HistoryConfig(
        path=pathlib.Path(
            history_raw.get("path", "Data/NigerDelta Synthetic Production History.xlsx")
        ),
        sheet_name=history_raw.get("sheet_name", 0),
        date_column=history_raw.get("date_column", "DATE"),
        column_map=history_raw.get("column_map"),
    )

    config = ProjectConfig(
        deck_path=pathlib.Path(raw.get("deck_path", "Data/NigerDelta UGH1 Composite Field.DATA")),
        output_root=pathlib.Path(raw.get("output_root", "runs")),
        ledger_path=pathlib.Path(raw.get("ledger_path", "runs/ledger.json")),
        flow_executable=raw.get("flow_executable", "flow"),
        wells=tuple(raw.get("wells", constants.PRODUCER_WELLS)),
        objective=objective,
        history=history,
        root=config_path.resolve().parent,
    )
    config.validate()
    return config


def save(config: ProjectConfig, config_path: pathlib.Path | str = DEFAULT_CONFIG_FILENAME) -> None:
    """Write `config` out as YAML at `config_path`.

    Path fields are written relative to :attr:`ProjectConfig.root` where
    possible, so the file stays portable if the project directory is
    moved or cloned elsewhere.
    """
    config_path = pathlib.Path(config_path)
    payload = {
        "deck_path": str(config.deck_path),
        "output_root": str(config.output_root),
        "ledger_path": str(config.ledger_path),
        "flow_executable": config.flow_executable,
        "wells": list(config.wells),
        "objective": {
            "weights": config.objective.weights,
            "target_j": config.objective.target_j,
        },
        "history": {
            "path": str(config.history.path),
            "sheet_name": config.history.sheet_name,
            "date_column": config.history.date_column,
            "column_map": config.history.column_map,
        },
    }
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

"""Shared plumbing every `nagcsu` subcommand uses."""

import dataclasses
import difflib
import pathlib
import typing

import click
import yaml

from nagcsu import config, history, ledger, parameters
from nagcsu.deck import Deck

WELLS_HELP = (
    "Wells whose water cut and GOR count toward J: 'none' (field totals only), 'all', "
    "or a comma-separated list such as AFIESERE,KOKORI. Needs non-zero `wells_watercut` / "
    "`wells_gor` weights (nagcsu.yaml `objective.weights` or --weights). Defaults to "
    "`objective.wells` in nagcsu.yaml."
)


def wells_option(function: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    """Add the shared `--wells` option to a command."""
    return click.option("--wells", "wells_raw", default=None, help=WELLS_HELP)(function)


def weights_option(function: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    """Add the shared `--weights` option to a command."""
    return click.option(
        "--weights",
        "weights_raw",
        default=None,
        help=(
            "Override objective weights for this command only, e.g. "
            "'pressure=0.5,watercut=0.3,gor=0.1,wells_watercut=0.07,wells_gor=0.03'. "
            "Vectors not listed keep their nagcsu.yaml weight; nagcsu.yaml is never modified."
        ),
    )(function)


def parse_weight_overrides(raw: str) -> dict[str, float]:
    """Parse a `--weights` value into `{vector_name: weight}`.

    :param raw: Comma-separated `name=value` pairs, for example
        `"pressure=1,watercut=0,gor=0"`.
    :raises click.BadParameter: if a pair is malformed or a weight is
        not a valid float.
    """
    overrides: dict[str, float] = {}
    for pair in raw.split(","):
        name, _, value = pair.partition("=")
        name = name.strip()
        if not name or not value:
            raise click.BadParameter(
                f"Malformed weight override {pair!r}; expected 'name=value', e.g. 'gor=0'."
            )
        try:
            overrides[name] = float(value)
        except ValueError as error:
            raise click.BadParameter(f"Weight for {name!r} is not a number: {value!r}") from error
    return overrides


def resolve_wells(project_config: config.ProjectConfig, raw: str) -> tuple[str, ...]:
    """Turn a `--wells` value into deck well names.

    :raises click.BadParameter: for a name that matches no configured well.
    """
    text = raw.strip().lower()
    if text in ("none", "field", ""):
        return ()
    if text == "all":
        return tuple(project_config.wells)

    chosen: list[str] = []
    for name in raw.split(","):
        matched = history.match_well_name(name.strip(), project_config.wells)
        if matched is None:
            raise click.BadParameter(
                f"Unknown well {name.strip()!r}. Configured wells: {list(project_config.wells)}"
            )
        if matched not in chosen:
            chosen.append(matched)
    return tuple(chosen)


def apply_objective_overrides(
    project_config: config.ProjectConfig,
    *,
    wells_raw: str | None = None,
    weights_raw: str | None = None,
) -> config.ProjectConfig:
    """Return `project_config` with `--wells` / `--weights` applied for this command only.

    :raises click.ClickException: if wells are selected but both per-well
        weights are zero, since those wells would silently not count.
    """
    objective = project_config.objective
    if weights_raw:
        objective = dataclasses.replace(
            objective, weights={**objective.weights, **parse_weight_overrides(weights_raw)}
        )
    if wells_raw is not None:
        objective = dataclasses.replace(objective, wells=resolve_wells(project_config, wells_raw))
    well_weight = sum(objective.weights.get(name, 0.0) for name in ("wells_watercut", "wells_gor"))
    if objective.wells and well_weight <= 0:
        raise click.ClickException(
            f"Wells {list(objective.wells)} are selected but wells_watercut and wells_gor have "
            f"zero weight, so they would not affect J. Add e.g. "
            f"--weights pressure=0.5,watercut=0.3,gor=0.1,wells_watercut=0.07,wells_gor=0.03 "
            f"(or set them in nagcsu.yaml), or use --wells none."
        )
    if well_weight > 0 and not objective.wells:
        raise click.ClickException(
            "wells_watercut / wells_gor have weight but no wells are selected; "
            "pass --wells all (or a list), or set their weights to 0."
        )
    return dataclasses.replace(project_config, objective=objective)


def load(ctx: click.Context) -> tuple[config.ProjectConfig, Deck]:
    """Load the project config and its base deck for the current invocation.

    :raises click.ClickException: if the config file or the deck it
        points at cannot be found.
    """
    config_path = ctx.obj["config_path"]
    try:
        project_config = config.load(config_path)
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    deck_path = project_config.get_resolved_path(project_config.deck_path)
    try:
        base_deck = Deck.load(deck_path)
    except FileNotFoundError as error:
        raise click.ClickException(f"Deck not found at {deck_path}: {error}") from error

    return project_config, base_deck


def parse_param_options(pairs: tuple[str, ...]) -> dict[str, float]:
    """Parse repeated `--param NAME=VALUE` strings into a state dict.

    :raises click.BadParameter: if a pair is not `NAME=VALUE`, if `NAME`
        is not a registered parameter, or if `VALUE` is not a float.
    """
    state: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"Expected NAME=VALUE, got {pair!r}")
        name, _, raw_value = pair.partition("=")
        name = name.strip()
        if name not in parameters.PARAMETERS:
            valid_names = ", ".join(sorted(parameters.PARAMETERS))
            raise click.BadParameter(f"Unknown parameter {name!r}. Valid names: {valid_names}")
        try:
            state[name] = float(raw_value)
        except ValueError as error:
            raise click.BadParameter(f"{name}={raw_value!r} is not a number") from error
    return state


def echo_outcome_header(record: ledger.RunRecord) -> None:
    """Print a one-line summary of a run's score and health, consistently.

    Prints the simulation-failure message instead, when `record`
    represents a run whose simulation never produced output, since `j`
    and `prt_is_clean` are meaningless for that run.
    """
    if record.simulation_error:
        click.echo(f"{record.run_id}: simulation failed: {record.simulation_error}")
        return
    j_text = f"J={record.j:.4f}" if record.j is not None else "J=not scored"
    clean_text = (
        "clean"
        if record.prt_is_clean
        else ("NEEDS ATTENTION" if record.prt_is_clean is not None else "not checked")
    )
    click.echo(f"{record.run_id}: {j_text}, run health: {clean_text}")


def warn_if_every_trial_failed(best_j: float, *, command: str) -> None:
    """Raise a clear error if a search's best trial never actually scored.

    A completely flat `float("inf")` objective (every trial's simulation
    failed) is not a calibration result worth trusting or writing out as
    a "best" state; this stops `match sweep/random/auto` from silently
    treating a broken OPM Flow setup as a successful search.

    :raises click.ClickException: if `best_j` is infinite.
    """
    if best_j == float("inf"):
        raise click.ClickException(
            f"Every trial in this `{command}` run failed to simulate; there is no usable "
            f"best result. Check `flow_executable` in the project config and that OPM Flow "
            f"runs on this deck at all (try a plain `nagcsu run` first)."
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Baseline:
    """The parameter state and deck a command starts from."""

    state: dict[str, float]
    """Full parameter state, every registered parameter present."""

    deck: Deck
    """Deck the state is patched into."""

    spec: str
    """What was asked for: "default", "best", "latest", a run ID or a file path."""

    run_id: str | None
    """Run the state came from, or `None` for defaults or a snapshot file."""

    j: float | None
    """That run's J, if it had one."""

    deck_source: str
    """"project", "run:<run_id>" or a path: where `deck` was loaded from."""

    def ledger_label(self) -> str:
        """Short text stored on each run record as `RunRecord.baseline`."""
        if self.run_id:
            return f"run:{self.run_id}"
        return "default" if self.spec == "default" else f"file:{self.spec}"


def baseline_options(
    function: typing.Callable[..., typing.Any],
) -> typing.Callable[..., typing.Any]:
    """Add the shared `--baseline` and `--base-deck` options to a command."""
    function = click.option(
        "--base-deck",
        "base_deck_raw",
        default=None,
        help=(
            "Deck to patch parameters into: 'auto' (the baseline run's own deck when it "
            "still exists, else the project deck), 'project', or a path to a .DATA file."
        ),
    )(function)
    return click.option(
        "--baseline",
        "baseline_raw",
        default=None,
        help=(
            "Where to start from: 'default' (registered defaults), 'best' (lowest-J run), "
            "'latest', a run ID, or a parameters.yaml path. Defaults to `baseline` in "
            "nagcsu.yaml, which defaults to 'default'. --param values still apply on top."
        ),
    )(function)


def resolve_baseline(
    project_config: config.ProjectConfig,
    project_deck: Deck,
    *,
    baseline_raw: str | None = None,
    base_deck_raw: str | None = None,
) -> Baseline:
    """Work out the state and deck a command should start from.

    :param project_deck: The deck loaded from `deck_path`.
    :param baseline_raw: The `--baseline` value, or `None` to use the
        project config's `baseline`.
    :param base_deck_raw: The `--base-deck` value, or `None` for 'auto'.
    :raises click.ClickException: if the named run or file does not exist
        or has no usable state.
    """
    spec = (baseline_raw if baseline_raw is not None else project_config.baseline).strip()
    defaults = parameters.default_state()
    run_id: str | None = None
    j: float | None = None
    state = dict(defaults)

    if spec.lower() != "default":
        if spec.lower().endswith((".yaml", ".yml")):
            snapshot_path = pathlib.Path(spec)
            if not snapshot_path.is_file():
                raise click.ClickException(f"Baseline file not found: {spec}")
            loaded = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
            unknown = [name for name in loaded if name not in parameters.PARAMETERS]
            if unknown:
                raise click.ClickException(f"{spec} has unknown parameters: {unknown}")
            state.update({name: float(value) for name, value in loaded.items()})
        else:
            records = ledger.load(project_config.get_resolved_path(project_config.ledger_path))
            usable = [r for r in records if r.parameter_state and not r.simulation_error]
            if spec.lower() == "best":
                record = ledger.get_best_record(usable)
                if record is None:
                    raise click.ClickException("--baseline best: no scored run in the ledger yet.")
            elif spec.lower() == "latest":
                if not usable:
                    raise click.ClickException(
                        "--baseline latest: the ledger has no usable run yet."
                    )
                record = usable[-1]
            else:
                matches = [r for r in records if r.run_id == spec]
                if not matches:
                    close = difflib.get_close_matches(spec, [r.run_id for r in records], n=3)
                    hint = f" Did you mean: {', '.join(close)}?" if close else ""
                    raise click.ClickException(f"--baseline: no run {spec!r} in the ledger.{hint}")
                record = matches[0]
                if not record.parameter_state:
                    raise click.ClickException(f"Run {spec!r} has no recorded parameter state.")

            run_id, j = record.run_id, record.j
            state.update({
                n: float(v)
                for n, v in record.parameter_state.items()
                if n in parameters.PARAMETERS
            })

    deck, deck_source = project_deck, "project"
    wanted = (base_deck_raw or "auto").strip()
    if wanted.lower() == "project":
        pass
    elif wanted.lower() == "auto":
        if run_id:
            run_deck = (
                project_config.get_resolved_path(project_config.output_root)
                / run_id
                / project_config.deck_path.name
            )
            if run_deck.is_file():
                deck, deck_source = Deck.load(run_deck), f"run:{run_id}"
            else:
                click.echo(f"Note: run {run_id}'s deck is gone; using the project deck.", err=True)
    else:
        try:
            deck, deck_source = Deck.load(wanted), wanted
        except FileNotFoundError as error:
            raise click.ClickException(f"--base-deck not found: {wanted}") from error
    return Baseline(state=state, deck=deck, spec=spec, run_id=run_id, j=j, deck_source=deck_source)


def widen_bounds_to_include(bounds: tuple[float, float], value: float) -> tuple[float, float]:
    """Stretch `bounds` so `value` (a baseline outside the recommended range) sits inside."""
    return min(bounds[0], value), max(bounds[1], value)

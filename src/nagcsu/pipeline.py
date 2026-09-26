"""Wiring one run together: patch the deck, simulate, parse, score.

Every CLI command that needs to turn a parameter state into a J value,
whether that is `nagcsu run` doing it once or `nagcsu match auto` doing
it hundreds of times through `nagcsu.algorithms`, goes through
`execute_run` so the deck-patch-simulate-score sequence is only
written once.
"""

import dataclasses
import itertools
import pathlib
import typing

from nagcsu import history, ledger, objective, parameters, prt, simulate, summary
from nagcsu.config import ProjectConfig
from nagcsu.deck import Deck
from nagcsu.exceptions import SimulationError


@dataclasses.dataclass(frozen=True, slots=True)
class RunOutcome:
    """Everything produced by one `execute_run` call."""

    run_id: str
    """Identifier this run was executed under."""

    output_dir: pathlib.Path
    """Directory this run's deck and OPM Flow output were written to."""

    deck_path: pathlib.Path
    """Path to the patched deck that was actually simulated."""

    resolved_state: dict[str, float]
    """Full parameter state used, with every omitted parameter filled
    in from its default (see `nagcsu.parameters.resolve_state`).
    """

    prt_report: prt.PrtReport | None
    """Parsed `.PRT` health, or `None` if no `.PRT` was produced."""

    objective_result: objective.ObjectiveResult | None
    """Score against the observed history, or `None` if `score=False`
    was passed to `execute_run`, the simulation failed outright
    (see `simulation_error`), or no summary output was produced.
    """

    simulation_error: str | None
    """Message from `nagcsu.simulate.run` if it raised `SimulationError`
    (OPM Flow could not be launched, or exited nonzero with no summary
    output), or `None` if the simulation ran without that failure. A
    parameter state that makes OPM Flow crash outright is a normal
    outcome for a search strategy to hit; this field is how the caller
    tells that apart from a state that scored badly.
    """


def execute_run(
    config: ProjectConfig,
    base_deck: Deck,
    state: dict[str, float],
    *,
    run_id: str,
    score: bool = True,
) -> RunOutcome:
    """Patch, simulate and (optionally) score one parameter state.

    Never raises for a simulation that fails to run (see
    `RunOutcome.simulation_error`): a parameter state that makes OPM
    Flow crash outright is a normal, expected outcome for a search
    strategy to encounter, not a reason to abort the whole search.

    :param base_deck: The pristine deck to patch from; see the
        reproducibility note on `nagcsu.parameters.apply_state`
        for why this should never be a deck from a previous run.
    :param state: Parameter state to apply; missing parameters fall
        back to their defaults.
    :param run_id: Subdirectory name (under `config.output_root`) this
        run's deck and OPM Flow output are written to.
    :param score: Whether to load the run's summary output and score it
        against the configured observed history. Set to `False` for a
        quick sanity run where scoring is not needed.
    """
    resolved_state = parameters.resolve_state(state)
    patched_deck = parameters.apply_state(base_deck, resolved_state)

    output_dir = config.get_resolved_path(config.output_root) / run_id
    deck_path = output_dir / base_deck.path.name
    patched_deck.save(deck_path)

    try:
        run_result = simulate.run(
            deck_path,
            output_dir,
            flow_executable=config.flow_executable,
            extra_mounts=config.extra_mounts or None,
        )
    except SimulationError as error:
        return RunOutcome(
            run_id=run_id,
            output_dir=output_dir,
            deck_path=deck_path,
            resolved_state=resolved_state,
            prt_report=None,
            objective_result=None,
            simulation_error=str(error),
        )

    prt_report = None
    if run_result.prt_path is not None and run_result.prt_path.exists():
        prt_report = prt.parse(run_result.prt_path)

    objective_result = None
    if score and run_result.case_basename is not None:
        simulated_frame = summary.load_summary(run_result.case_basename, wells=list(config.wells))
        observed_frame = history.load_observed_history(
            config.get_resolved_path(config.history.path),
            file_format=config.history.file_format,
            sheet_name=config.history.sheet_name,
            date_column=config.history.date_column,
            well_column=config.history.well_column,
            column_map=config.history.column_map,
            wells=list(config.wells),
        )
        objective_result = objective.score(
            simulated_frame,
            observed_frame,
            weights=config.objective.weights,
            date_column=history.OUTPUT_DATE_COLUMN,
        )

    return RunOutcome(
        run_id=run_id,
        output_dir=output_dir,
        deck_path=deck_path,
        resolved_state=resolved_state,
        prt_report=prt_report,
        objective_result=objective_result,
        simulation_error=None,
    )


def make_evaluate(
    config: ProjectConfig,
    base_deck: Deck,
    *,
    run_id_prefix: str = "eval",
    on_outcome: typing.Callable[[RunOutcome], None] | None = None,
) -> typing.Callable[[dict[str, float]], float]:
    """Build an `evaluate(state) -> J` callback for `nagcsu.algorithms`.

    Each call runs a full `execute_run` under a fresh, incrementing
    run ID (`<run_id_prefix>_00000`, `<run_id_prefix>_00001`, ...), so a
    search strategy's hundreds of trials each get their own output
    directory rather than overwriting one another.

    :param on_outcome: Called with each trial's `RunOutcome`, for example
        to append it to the run ledger. Optional; omit for a
        throwaway search where every trial's output can be discarded
        once the search returns.
    :returns: A function returning `float("inf")` for a state whose run
        produced no score, so a failed or unscoreable trial is never
        mistaken for the best one by a minimizing search strategy.
    """
    counter = itertools.count()

    def evaluate(state: dict[str, float]) -> float:
        run_id = f"{run_id_prefix}_{next(counter):05d}"
        outcome = execute_run(config, base_deck, state, run_id=run_id, score=True)
        if on_outcome is not None:
            on_outcome(outcome)
        if outcome.objective_result is None:
            return float("inf")
        return outcome.objective_result.j

    return evaluate


def to_run_record(
    outcome: RunOutcome, *, group: str | None, strategy: str | None, note: str
) -> ledger.RunRecord:
    """Build a `ledger.RunRecord` from a `RunOutcome`.

    The single place every CLI command turns a run's outcome into its
    logged record, so a fix here (for example, surfacing
    `simulation_error`) reaches every command that logs a run instead of
    needing the same fix repeated in each one.
    """
    return ledger.RunRecord(
        run_id=outcome.run_id,
        created_at=ledger.timestamp_now(),
        parameter_state=outcome.resolved_state,
        group=group,
        strategy=strategy,
        j=outcome.objective_result.j if outcome.objective_result else None,
        vector_nrmse=(
            {
                name: vector_score.nrmse
                for name, vector_score in outcome.objective_result.vector_scores.items()
            }
            if outcome.objective_result
            else None
        ),
        prt_is_clean=outcome.prt_report.is_clean if outcome.prt_report else None,
        note=note,
        simulation_error=outcome.simulation_error,
    )

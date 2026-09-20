# nagcsu CLI: architecture and design notes

This document explains how the `nagcsu` package is put together and why,
so a change six months from now can be made in the right module instead
of wherever is convenient. It was written against the real deck and
sample run already in `Data/`, not against the Execution Plan document
alone, and it calls out the two or three places where the real output
disagreed with that document's assumptions.

## Goals

- One CLI, `nagcsu`, covering: run the deck, sweep or randomize a
  parameter, auto-tune a full history match, rank parameters by
  sensitivity, and report on any of it.
- Every simulation is reproducible from a logged parameter state alone.
  No command edits a deck in place; every run patches a fresh copy of
  the pristine base deck.
- Tunable parameters follow the tuning priority groups from Stage D.1 of
  the Phase 2 Execution Plan (aquifer, then permeability multiplier,
  then SGOF shape, and so on), and `match auto` changes one group at a
  time per Stage D.3, stopping as soon as J reaches its target per
  Stage C.4 rather than continuing to chase a smaller number.

## Two things the real data changed from the plan

The Execution Plan was written before a real OPM Flow run existed to
check it against. Building against the actual `Data/` output turned up
two places worth flagging explicitly, because a script written to the
plan's assumptions instead of the real format would appear to work and
then silently do nothing:

1. **No `.DATA`-file split into `INCLUDE` files.** Stage 3.2's folder
   layout assumes `base/include/aquifer.inc`, `relperm.inc`, and so on.
   The actual deck is one monolithic `.DATA` file. `nagcsu.deck` and
   `nagcsu.parameters` patch the monolithic file directly with
   exactly-one-match regexes rather than assuming an include-file split
   exists. `match auto`'s "write the calibrated value to an include
   file" is implemented as writing a full patched deck plus a small
   `parameters.yaml` snapshot instead, since there is no single `.inc`
   file per parameter to write into.

2. **The `.PRT` file has no "MATERIAL BALANCE" line.** Stage A's
   `prt_check.py` greps for one, and even says to "adapt to your actual
   keyword layout." OPM Flow 2026.04's real `.PRT` output has no such
   line; it prints an `Error summary:` block (Warnings/Info/Errors/Bugs/
   Problems counts) and marks each well-convergence failure as
   `Warning: Inner well iterations failed for well <NAME> Treat the well
   as unconverged.`, paired with a `Problem: [...] Error when inverting
   local well equations for well <NAME>` line. `nagcsu.prt` is built
   against that real format; see `tests/test_prt.py`, which runs against
   the actual baseline `.PRT` in `Data/` (0 errors, 351 warnings, 12/12
   report steps completed, and it correctly identifies EVWRENI as the
   well with by far the most convergence warnings in that run).

A third, smaller gotcha worth knowing: OPM Flow does not necessarily
write its output under the same basename as the input `.DATA` file's
case. The baseline run's output files are all upper case
(`NIGERDELTA UGH1 COMPOSITE FIELD.PRT`) even though the input deck is
mixed case (`NigerDelta UGH1 Composite Field.DATA`). `nagcsu.simulate`
never assumes a basename; it globs the output directory for whatever
`.UNSMRY` file actually appeared.

## Module map

```
src/nagcsu/
  constants.py       Well names, default objective weights, tuning group order
  exceptions.py       NagcsuError and its subtypes
  config.py            ProjectConfig / nagcsu.yaml load and save
  deck.py               Deck text wrapper: exactly-one-match patch primitives
  corey.py               Corey relperm formulas, verified against the shipped
                          SGOF/SWOF tables to ~1e-5
  parameters.py            The tunable parameter registry and apply_state()
  simulate.py                subprocess wrapper for `flow`, output-basename discovery
  prt.py                      .PRT parsing, built against the real output format
  summary.py                   res2df/resfo-based tidy summary extraction
  history.py                    Observed history workbook loader
  objective.py                   NRMSE per vector, combined weighted J
  ledger.py                       JSON record of every run, for report and match auto
  pipeline.py                      execute_run() / make_evaluate(): the one place
                                    deck-patch -> simulate -> parse -> score is wired
  reporting.py                      Markdown snapshot report rendering
  algorithms/
    __init__.py         Trial / SearchResult shared types
    grid.py               Grid search over one or more parameters
    random_search.py       Uniform random search within bounds
    coordinate_descent.py   The Stage D.1/D.3 one-group-at-a-time auto-tune driver
    sensitivity.py           One-at-a-time local sensitivity ranking
  cli/
    __init__.py    Top-level `nagcsu` click group, assembles every subcommand
    _context.py     Shared config loading and --param NAME=VALUE parsing
    init_cmd.py      `nagcsu init`
    run_cmd.py        `nagcsu run`
    match_cmd.py       `nagcsu match` (sweep, random, auto, list-parameters)
    sensitivity_cmd.py  `nagcsu sensitivity run`
    report_cmd.py         `nagcsu report` (list, show)
```

Each algorithm module is a self-contained entry point (`search(...)` or
`run(...)`) that takes a plain `evaluate(state) -> J` callback, built by
`nagcsu.pipeline.make_evaluate`. This keeps every search strategy
testable against a cheap synthetic objective (see `tests/test_algorithms.py`)
without needing OPM Flow installed, which is also why the test suite for
everything except `simulate.py`/`summary.py`'s actual OPM Flow
invocation runs in this sandbox without OPM Flow being available at all.

## The parameter model

`nagcsu.parameters.PARAMETERS` is a flat registry of every tunable
parameter, each tagged with the tuning priority group it belongs to
(`nagcsu.constants.TUNING_PRIORITY_ORDER`):

| Group | Priority | Parameters |
| --- | --- | --- |
| `aquifer` | 1 | radius, permeability, thickness, encroachment angle (AQUCT) |
| `permeability_multiplier` | 2 | areal PERMX/PERMY contrast (MULTIPLY) |
| `sgof_shape` | 3 | Sorg, oil-in-gas Corey exponent (literature assumptions) |
| `sgof_endpoints` | 4 | Sgc, krg_max, gas Corey exponent (anchor-derived, touch last) |
| `swof_endpoints` | 5 | krw_max, Sorw, water and oil Corey exponents (anchor-derived, touch last) |
| `rock_and_porosity` | 6 | rock compressibility, uniform porosity multiplier |

The plan's Stage D.1 table lists five groups; this registry splits SGOF
into a "shape" group (the two literature-assumed values, tune first) and
an "endpoints" group (the three anchor-derived values, tune only if
shape alone doesn't close the gap), mirroring how SWOF's endpoints are
already handled. This seemed a more faithful reading of "touch this
only if 1-3 don't close the water-cut gap" than leaving SGOF as a single
group.

`nagcsu.parameters.apply_state(base_deck, state)` always patches from a
*pristine* base deck, never from a previously patched one. The SGOF and
SWOF tables are fully regenerated from the Corey formulas each time
(keeping only the saturation and capillary-pressure columns from the
existing table); the porosity multiplier scales whatever value is in
`base_deck`, not whatever the deck currently holds. Applying the same
state twice from the same pristine base always produces byte-identical
output (`tests/test_parameters.py::test_apply_state_is_reproducible_from_the_pristine_base`).
This is why every run in `pipeline.execute_run` reloads the base deck
fresh rather than reusing a `Deck` object across runs.

## The objective

`nagcsu.objective.score()` implements Stage C exactly: NRMSE per vector
(pressure, water cut, GOR), combined into `J = w_p*NRMSE_p +
w_wc*NRMSE_wc + w_gor*NRMSE_gor` with the Stage C.2 starting weights
(0.50 / 0.35 / 0.15). Only `FPR`, `FWCT`, `FGOR` are ever scored, never
a rate vector, since the deck's `WCONPROD` blocks prescribe rates as an
input rather than something OPM Flow predicts.

## `match auto`: the auto-tune driver

`nagcsu match auto` is `nagcsu.algorithms.coordinate_descent.search()`
wired to the CLI. For each tuning group in priority order:

1. Skip the group entirely if J is already at or below the target.
2. Otherwise, run up to `--passes-per-group` full cycles through the
   group's parameters. Each pass re-optimizes one parameter at a time
   with `scipy.optimize.minimize_scalar(method="bounded")`, holding
   every other parameter at its current best value, and updates the
   running best state whenever a trial improves on it.
3. Move to the next group once the passes are done or the target is hit.

Every trial (not just the group's final best) is logged to the run
ledger, so `nagcsu report show <a trial run id>` can be inspected even
though the CLI only prints the per-group summary and the final state.
The final state is re-run once more under its own `auto_final` run ID so
its `.PRT` gets parsed and its deck and `parameters.yaml` snapshot get
written to a stable, predictable location.

This is a coordinate descent, not a global optimizer. It will not
escape a bad starting region by itself, which is exactly why the tuning
priority order matters and why it should always start from the deck's
shipped defaults (`nagcsu.parameters.default_state()`) rather than an
arbitrary state.

## What "an include file" became

The original ask was for auto mode to "write it to an include file."
Since the shipped deck has no `INCLUDE` structure to write into, `match
auto` instead writes:

- The full patched deck for the final state, at
  `runs/<run_id>/<deck filename>`, ready to run standalone.
- `runs/<run_id>/parameters.yaml`, a flat mapping of every parameter to
  its resolved value. This is what a future refactor into `INCLUDE`
  files (splitting the deck the way Stage 3.2 originally envisioned)
  would read its per-parameter values from.

## Bugs found and fixed after the first build

A code-review pass after the initial build turned up a few real
correctness issues worth recording, since the symptoms would only have
shown up during an actual multi-trial tuning session, not from reading
any single function in isolation:

- **A failed simulation used to crash the whole search.**
  `nagcsu.simulate.run` correctly raises `SimulationError` when OPM Flow
  exits nonzero with no summary output, but `nagcsu.pipeline.execute_run`
  did not catch it, so the first parameter combination that made OPM
  Flow crash outright (not uncommon for auto-tune's coordinate descent,
  which explores the full bounds of every parameter) would kill a
  `match auto`, `match sweep`, `match random` or `sensitivity run`
  session partway through, discarding every trial already run. This
  directly contradicted `pipeline.make_evaluate`'s own docstring, which
  already promised graceful `float("inf")` handling for "a state whose
  run produced no score." `execute_run` now catches `SimulationError`
  and returns a `RunOutcome` with `simulation_error` set instead of
  raising; every CLI command surfaces that message and logs it to the
  ledger rather than aborting. `match sweep/random/auto` also now raise
  a clear error up front if literally every trial failed (a flat
  `float("inf")` objective is never a usable calibration result), rather
  than quietly writing out a meaningless "best" state.
- **`nagcsu sensitivity run` was reconstructing run IDs by hand instead
  of using the ones actually assigned.** It built each ledger record
  from `enumerate(trials)` after the fact, assuming that loop's index
  would always line up with the counter inside
  `pipeline.make_evaluate`'s closure. The two happened to stay in sync
  for the exact call pattern `sensitivity.run` uses, but the records it
  produced were missing `vector_nrmse`, `prt_is_clean`, and any
  simulation failure entirely, unlike every other `match` subcommand.
  It now wires `on_outcome` the same way `match sweep/random/auto` do,
  logging the real `RunOutcome` for each trial instead of a
  reconstruction of it.
- **`swof.oil_exponent` was hardcoded to `4.0`** instead of exposed as a
  tunable parameter, even though the equivalent SGOF parameter
  (`sgof.oil_exponent`) is tunable and both are literally the same
  Corey-model role in their respective tables. Stage D.1's table only
  names Krw_max/Sorw/nw for the SWOF group, which is why it was left
  out originally, but permanently fixing a real relative-permeability
  shape parameter meant `match auto`'s `swof_endpoints` group could
  never fully close a water-cut gap that this exponent, not the other
  three, was actually responsible for. It is now `swof.oil_exponent`,
  in the same `swof_endpoints` group, default `4.0` (matching the
  anchor) so nothing about the shipped deck's behavior changes until it
  is actually tuned.
- **`history.load_observed_history` did not normalize its `DATE` column**
  the way `summary.load_summary` normalizes its own, even though
  `objective.score` merges the two frames on an exact date match. A
  workbook whose dates come back from `pandas.read_excel` with a
  different time-of-day component, or as strings rather than a parsed
  datetime, could merge to nothing (an empty, `HistoryAlignmentError`
  result) or silently line up incorrectly. Both frames now go through
  the same `pandas.to_datetime(...).dt.normalize()` step.
- **`simulate.run`'s `extra_args` parameter had the same
  `list[str] = ()` type/default mismatch already fixed in
  `summary.load_summary` during the previous round**, missed the first
  time because ruff's mutable-default check (`B006`) only flags a
  literal mutable default, not a type/default mismatch with an
  immutable one; nothing in the lint config catches this class of bug,
  only reading the signature against its own type hint does.

## Known gaps, worth knowing before relying on this

- **`history.py`'s column mapping is a guess.** The production-history
  workbook's actual columns could not be read while building this (it's
  a binary `.xlsx`, and the tool used to browse the repo could not
  return its content). `guess_column_map` matches the
  `DATE,FPR,WWCT_<WELL>,WGOR_<WELL>` layout Stage B.2 of the Execution
  Plan documents building the workbook in, tolerant of `_`, `:` or `-`
  as the separator and either case. If the real workbook doesn't match,
  set `history.column_map` explicitly in `nagcsu.yaml` (see
  `nagcsu.config.HistoryConfig.column_map`); `tests/test_history.py`
  shows the explicit-mapping path working.
- **`summary.py`'s res2df/resfo calls are API-verified, not run-verified.**
  `res2df.summary.df()`'s signature and `resfo.lazy_read()`'s entry API
  were both checked directly against the installed packages, but no real
  `.UNSMRY`/`.SMSPEC` pair was available to run them against end to end
  (same binary-file limitation as above). Run `nagcsu run` once against
  a real deck as the first thing to check before relying on `match auto`
  for a long session.
- **The MULTIPLY and PORO patches are shaped to the current deck.** They
  match the exact box layout (`1 10`, `21 30`, five `1 30 1 30 <k> <k>`
  layers) the shipped deck uses. If the deck's grid box structure ever
  changes, `nagcsu.parameters.apply_permeability_multiplier` and
  `apply_rock_and_porosity` will raise `DeckPatchError` (wrong match
  count) rather than silently patching the wrong thing, which is the
  intended failure mode, but the patterns themselves will need updating.

## Everything that was end-to-end tested versus what wasn't

Tested against the real files in `Data/`: deck patching (all six
parameter groups), Corey formula correctness, `.PRT` parsing, the full
CLI (`init`, `run`, `match sweep/random/auto`, `sensitivity run`,
`report list/show`) with `simulate.run`/`summary.load_summary`/
`history.load_observed_history` stubbed out to avoid needing OPM Flow or
the real workbook in this environment.

Not run against a real OPM Flow binary or the real history workbook,
since neither was available while this was built: the actual
`subprocess.run([flow_executable, ...])` call in `simulate.py`, and
`history.py`'s column guessing against the real workbook's actual
column names. Both are small, isolated pieces; running `nagcsu run`
once for real is the fastest way to confirm both at once.

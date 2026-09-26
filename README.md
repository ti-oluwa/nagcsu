# nagcsu

A command line tool for history matching and provisional storage-cycling
scheduling against the N/AGCSU Phase 2 UGH-1 composite sector model. It
wraps OPM Flow, [res2df](https://github.com/equinor/res2df) and
[resfo](https://github.com/equinor/resfo) so a full tuning session is a
handful of `nagcsu` commands instead of hand-editing deck files and
eyeballing plots.

See `docs/ARCHITECTURE.md` for how this is put together and, more
importantly, the two or three places the real deck and `.PRT` output
disagreed with the Phase 2 Execution Plan's assumptions.

## Requirements

- Python 3.10+
- OPM Flow available on your `PATH` as `flow` (or point `--flow-executable`
  / `nagcsu.yaml`'s `flow_executable` at it). Not installed by this
  package; nagcsu only shells out to it, and works the same whether
  that's a native binary or a Docker-wrapped one (see `docs/ARCHITECTURE.md`).

## Install

As a standalone tool, without setting up a virtual environment:

```sh
uvx --from git+https://github.com/ti-oluwa/nagcsu nagcsu --help
# or
pipx install git+https://github.com/ti-oluwa/nagcsu
nagcsu --help
```

For local development, from a clone:

```sh
uv sync
uv run nagcsu --help
```

or, without `uv`:

```sh
pip install -e .
nagcsu --help
```

## Quick start

From the repository root, where `Data/` already has the deck and
observed history file (`.xlsx`/`.xls` or `.csv`, one row per date or
one row per well per date; both are auto-detected, see
`docs/ARCHITECTURE.md`):

```sh
nagcsu init                                   # writes nagcsu.yaml
nagcsu run                                    # baseline run, logs J to the ledger
nagcsu match list-parameters                  # see every tunable parameter, grouped
nagcsu sensitivity run --group aquifer        # what's worth tuning first
nagcsu match sweep --param aquifer.radius --values 12000,14000,16137,18000,20000
nagcsu match auto --report runs/summary.md    # full auto-tune, one group at a time
nagcsu report list                            # every run logged so far, best first
nagcsu report show best                       # snapshot of the best run found
```

## Commands

- `nagcsu init` - write a `nagcsu.yaml` project config in the current directory.
- `nagcsu run [--param NAME=VALUE ...]` - run the deck once, score it, log it.
- `nagcsu match list-parameters` - list every tunable parameter, its group, bounds and default.
- `nagcsu match sweep --param NAME --values v1,v2,...` - run every value of one parameter.
- `nagcsu match random --param NAME [--param NAME2 ...] --trials N` - random search within bounds.
- `nagcsu match auto [--target-j J] [--groups g1,g2,...]` - auto-tune one group at a time until J reaches its target (Stage D.1/D.3/C.4).
- `nagcsu sensitivity run [--group NAME]` - rank parameters by local effect on J.
- `nagcsu report list` / `nagcsu report show <run_id|latest|best>` - inspect logged runs.

Every command reads `nagcsu.yaml` by default; pass `--config path/to/file.yaml`
to use a different one.

## Parameter names

Every parameter is a dotted `<group-ish>.<name>`, for example
`aquifer.radius` or `sgof.sorg`. Run `nagcsu match list-parameters` for
the full list with bounds, defaults and a one-line description of what
each one controls; it's generated from the same registry every other
command uses, so it never drifts out of date.

## Observed history file format

`history.path` in `nagcsu.yaml` can point at either an Excel file
(`.xlsx`/`.xls`) or a CSV file; the extension decides which, or set
`history.file_format: csv`/`excel` explicitly if it doesn't match the
file's real content. Two layouts are auto-detected:

- **Wide**: one row per date, `DATE,FPR,WWCT_<WELL>,WGOR_<WELL>,...`.
- **Long**: one row per well per date (a typical monthly production
  export), with a well-identifier column (`Field`, `Well`, and similar)
  and per-well rate/pressure columns. Field totals are computed from
  summed rates across wells, not averaged per-well ratios.

If your file's columns aren't recognized, set `history.well_column`
(long format) or `history.column_map` (wide format) explicitly in
`nagcsu.yaml`. See "The observed history file" in `docs/ARCHITECTURE.md`
for exactly what's matched and the one unit assumption (gas/oil rate
units matching OPM Flow's own GOR convention) worth checking by hand.

## Running OPM Flow through Docker

If `flow` on your system is a Docker-wrapped script rather than a
native binary, nothing extra is needed: `nagcsu` always launches it
from the directory containing both the deck and that run's output
folder, and passes both as relative paths, which is what such a
wrapper's own directory-mounting needs to see them. If your deck has an
`INCLUDE` reaching outside the project directory, list the extra host
paths it needs under `extra_mounts` in `nagcsu.yaml`:

```yaml
extra_mounts:
  - /data/shared          # mounted at the same path inside the container
  - /data/pvt=/mnt/pvt    # mounted at a different path inside the container
```

See "Running OPM Flow through Docker" in `docs/ARCHITECTURE.md` for how
this is implemented and why it matters.

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format .
```

The test suite runs against the real sample deck and `.PRT` file already
in `Data/`, not synthetic fixtures, and does not require OPM Flow to be
installed: `simulate.py`'s actual `flow` invocation is the one piece not
covered by the test suite for that reason (see "Known gaps" in
`docs/ARCHITECTURE.md`).

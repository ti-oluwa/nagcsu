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
  package; nagcsu only shells out to it.

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
observed history workbook:

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

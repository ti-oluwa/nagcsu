"""The `nagcsu` command line tool.

Wires together every subcommand: `nagcsu init`, `nagcsu run`,
`nagcsu match` (`sweep`, `random`, `auto`, `list-parameters`),
`nagcsu sensitivity run`, and `nagcsu report` (`list`, `show`). The
click group and its `main()` entry point live in `nagcsu.cli.app`,
re-exported here so the `nagcsu.cli:main` console script in
`pyproject.toml` and `python -m nagcsu.cli` both keep working.
"""

from nagcsu.cli.app import cli, main

__all__ = ["cli", "main"]

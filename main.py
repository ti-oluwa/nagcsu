"""Backward-compatible script entry point.

The real CLI lives in `nagcsu.cli`, installed as the `nagcsu` console
script (see `pyproject.toml`). This file exists so `uv run main.py` and
`python main.py` keep working exactly as they did before this CLI was
built, for anyone with muscle memory for the original `uv init` layout.
"""

from nagcsu.cli import main

if __name__ == "__main__":
    main()

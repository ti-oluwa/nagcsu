"""Loading the observed production/pressure history for scoring.

Stage B.2 of the Phase 2 Execution Plan has the workbook built with a
`DATE,FPR,WWCT_<WELL>,WGOR_<WELL>,...` column layout. Since the workbook
itself was not available to verify column-by-column while building this
module, :func:`guess_column_map` matches that documented layout with a
tolerant regex (accepting `WWCT_WELL`, `WWCT:WELL` or `WWCT-WELL`, any
case) rather than a single hardcoded name, and a project can always
override individual mappings via `history.column_map` in `nagcsu.yaml`
(see :class:`nagcsu.config.HistoryConfig`) if a workbook does not match.
"""

import pathlib
import re

import pandas

_WELL_VECTOR_PATTERN = re.compile(r"^(?P<vector>WWCT|WGOR|WBHP)[_:\-](?P<well>.+)$", re.IGNORECASE)


def guess_column_map(columns: list[str]) -> dict[str, str]:
    """Guess a res2df-vector-to-workbook-column mapping from column names.

    Matches a field-total pressure column named (case-insensitively)
    `FPR`, and any per-well water-cut/GOR/BHP column shaped like
    `WWCT_AFIESERE`, `WWCT:AFIESERE` or `WWCT-AFIESERE`, mapping it to
    the equivalent res2df vector name `WWCT:AFIESERE`.

    :returns: Mapping from res2df vector name (for example `"FPR"` or
        `"WWCT:AFIESERE"`) to the workbook column name it was matched from.
    """
    mapping: dict[str, str] = {}
    for column in columns:
        if column.strip().upper() == "FPR":
            mapping["FPR"] = column
            continue
        match = _WELL_VECTOR_PATTERN.match(column.strip())
        if match:
            vector = match["vector"].upper()
            well = match["well"].upper()
            mapping[f"{vector}:{well}"] = column
    return mapping


def load_observed_history(
    path: pathlib.Path | str,
    *,
    sheet_name: str | int = 0,
    date_column: str = "DATE",
    column_map: dict[str, str] | None = None,
    wells: list[str] | None = None,
) -> pandas.DataFrame:
    """Load the observed history workbook into a field-total scoring frame.

    Produces a frame with a `date_column` plus one column named after
    each entry in :data:`nagcsu.objective.SCORED_FIELD_VECTORS` (`FPR`,
    `FWCT`, `FGOR`), so it can be passed straight to
    :func:`nagcsu.objective.score` alongside a res2df-produced simulated
    frame. Field-total water cut and GOR are computed as simple
    per-well averages across `wells` when the workbook has no field-total
    column of its own; this is a coarse stand-in and Stage B.3 of the
    Execution Plan's ResInsight overlay should be treated as the source
    of truth for any well showing an unusually large residual.

    :param column_map: Explicit res2df-vector-name to workbook-column
        mapping. Falls back to :func:`guess_column_map` for anything
        not given explicitly.
    :param wells: Well names to average over for field-total water cut
        and GOR when no field-total column exists.
    :raises FileNotFoundError: if `path` does not exist.
    :raises KeyError: if `date_column`, `FPR`, or every per-well WWCT/WGOR
        column for `wells` is missing after mapping.
    """
    path = pathlib.Path(path)
    raw = pandas.read_excel(path, sheet_name=sheet_name)
    guessed = guess_column_map(list(raw.columns))
    resolved_map = {**guessed, **(column_map or {})}

    if date_column not in raw.columns:
        raise KeyError(
            f"Date column {date_column!r} not found in {path} (columns: {list(raw.columns)})"
        )

    frame = pandas.DataFrame({date_column: raw[date_column]})

    if "FPR" not in resolved_map:
        raise KeyError(f"No field pressure column found or mapped in {path}")
    frame["FPR"] = raw[resolved_map["FPR"]]

    wells = wells or []
    frame["FWCT"] = _field_average(raw, resolved_map, "WWCT", wells, path)
    frame["FGOR"] = _field_average(raw, resolved_map, "WGOR", wells, path)
    return frame


def _field_average(
    raw: pandas.DataFrame,
    resolved_map: dict[str, str],
    vector_prefix: str,
    wells: list[str],
    path: pathlib.Path,
) -> pandas.Series:
    """Average a per-well vector across `wells` into a field-total series."""
    field_key = f"F{vector_prefix[1:]}"  # WWCT -> FWCT, WGOR -> FGOR
    if field_key in resolved_map:
        return raw[resolved_map[field_key]]

    well_columns = [
        resolved_map[f"{vector_prefix}:{well.upper()}"]
        for well in wells
        if f"{vector_prefix}:{well.upper()}" in resolved_map
    ]
    if not well_columns:
        raise KeyError(
            f"No {field_key!r} or per-well {vector_prefix}:<WELL> columns found or "
            f"mapped in {path} for wells {list(wells)}"
        )
    return raw[well_columns].mean(axis=1)

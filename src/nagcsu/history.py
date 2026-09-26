"""Loading the observed production/pressure history for scoring.

Two file shapes are supported, auto-detected from the column names
(see `detect_long_format_columns`):

- **Wide**: one row per date, with a `DATE,FPR,WWCT_<WELL>,WGOR_<WELL>,...`
  layout. This is what Stage B.2 of the Phase 2 Execution Plan describes
  building.
- **Long**: one row per well per date (the shape of a typical monthly
  production export), identified by a well-name column (`Field`, `Well`,
  and similar) alongside per-well rate and pressure columns
  (`Oil_Rate_STB`, `Water_Rate_...`, `Reservoir_Pressure`, and so on).
  Field-total water cut and GOR are computed from summed rates across
  wells at each date, the same way OPM Flow's own FWCT/FGOR summary
  vectors are defined, rather than averaging each well's own ratio.

Both `.xlsx`/`.xls` and `.csv` files are accepted, dispatched by file
extension (`nagcsu.config.HistoryConfig.file_format` overrides this if a
file's extension does not match its real content, for example a `.txt`
export that is actually CSV).

Column names are matched with tolerant regexes rather than hardcoded,
since the exact wording varies by data source; a file whose columns
still are not recognized can be pointed at explicitly with
`nagcsu.config.HistoryConfig.column_map` and `.well_column`.
"""

import pathlib
import re
import typing

import pandas

WELL_VECTOR_PATTERN = re.compile(r"^(?P<vector>WWCT|WGOR|WBHP)[_:\-](?P<well>.+)$", re.IGNORECASE)
DATE_COLUMN_PATTERN = re.compile(
    r"^(date|report[_\s-]?date|prod(?:uction)?[_\s-]?date)$", re.IGNORECASE
)
WELL_ID_COLUMN_PATTERN = re.compile(
    r"^(field|well|well[_\s-]?name|well[_\s-]?id|uwi|api)$", re.IGNORECASE
)
OIL_RATE_PATTERN = re.compile(r"oil.*rate", re.IGNORECASE)
GAS_RATE_PATTERN = re.compile(r"gas.*rate", re.IGNORECASE)
WATER_RATE_PATTERN = re.compile(r"water.*rate", re.IGNORECASE)
PRESSURE_PATTERN = re.compile(r"pressure", re.IGNORECASE)
WATER_CUT_PATTERN = re.compile(r"water.*cut", re.IGNORECASE)
GOR_PATTERN = re.compile(r"(?<![a-z])gor(?![a-z])|gas.*oil.*ratio", re.IGNORECASE)
"""Matches "GOR" as a standalone word-ish token, including when it is
separated from the rest of the column name by an underscore (`Solution_GOR`)
rather than whitespace, which `\\bgor\\b` misses since `_` counts as a word
character in regex and so is not itself a word boundary.
"""

CSV_EXTENSIONS: frozenset[str] = frozenset({".csv", ".tsv", ".txt"})
EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls", ".xlsm"})

OUTPUT_DATE_COLUMN: str = "DATE"
"""The date column name every `load_observed_history` result uses,
regardless of what the source file's own date column is called (that
name is only ever the `date_column` parameter, used for reading). Kept
fixed because `nagcsu.summary.load_summary` always produces a `DATE`
column too, and `nagcsu.objective.score` merges the two frames on an
exact match of that column name; a history file whose own date column
happens to be called anything else (`Date`, `REPORT_DATE`, ...) must
not leak that name into the frame `objective.score` sees, or the merge
silently finds nothing in common between the two frames.
"""


def detect_file_format(path: pathlib.Path, file_format: str | None = None) -> str:
    """Return `"csv"` or `"excel"` for `path`.

    :param file_format: Explicit override (`"csv"` or `"excel"`); used
        as-is, case-insensitively, without looking at `path` at all.
    :raises ValueError: if `file_format` is given and is neither `"csv"`
        nor `"excel"`, or if neither is given and `path`'s extension is
        not recognized.
    """
    if file_format is not None:
        normalized = file_format.strip().lower()
        if normalized not in ("csv", "excel"):
            raise ValueError(f"file_format must be 'csv' or 'excel', got {file_format!r}")
        return normalized

    suffix = path.suffix.lower()
    if suffix in CSV_EXTENSIONS:
        return "csv"
    if suffix in EXCEL_EXTENSIONS:
        return "excel"
    raise ValueError(
        f"Could not tell whether {path} is CSV or Excel from its extension {suffix!r}. "
        f"Set history.file_format to 'csv' or 'excel' in nagcsu.yaml to be explicit."
    )


def read_raw_table(
    path: pathlib.Path,
    *,
    sheet_name: str | int = 0,
    file_format: str | None = None,
) -> pandas.DataFrame:
    """Read `path` into a raw `DataFrame`, dispatching on `detect_file_format`.

    :raises FileNotFoundError: if `path` does not exist.
    """
    resolved_format = detect_file_format(path, file_format)
    if resolved_format == "csv":
        return pandas.read_csv(path)
    return pandas.read_excel(path, sheet_name=sheet_name)


def peek_columns(
    path: pathlib.Path,
    *,
    sheet_name: str | int = 0,
    file_format: str | None = None,
) -> list[str]:
    """Read only `path`'s header row, without loading any data rows.

    Used by `nagcsu init` to guess `date_column` from a history file
    that already exists at init time, without paying for a full read of
    a potentially large file.

    :raises FileNotFoundError: if `path` does not exist.
    """
    resolved_format = detect_file_format(path, file_format)
    if resolved_format == "csv":
        return list(pandas.read_csv(path, nrows=0).columns)
    return list(pandas.read_excel(path, sheet_name=sheet_name, nrows=0).columns)


def detect_date_column(columns: list[str]) -> str | None:
    """Guess which column holds the report date, by name.

    Matches `Date`, `Report_Date`, `Production Date`, and similar,
    case-insensitively, tolerant of `_`, `-` or a space as the word
    separator.

    :returns: The matching column name, or `None` if no column name
        looked date-like.
    """
    return next((column for column in columns if DATE_COLUMN_PATTERN.match(column.strip())), None)


def guess_column_map(columns: list[str]) -> dict[str, str]:
    """Guess a res2df-vector-to-workbook-column mapping for a wide-format table.

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
        match = WELL_VECTOR_PATTERN.match(column.strip())
        if match:
            vector = match["vector"].upper()
            well = match["well"].upper()
            mapping[f"{vector}:{well}"] = column
    return mapping


def detect_long_format_columns(columns: list[str]) -> dict[str, str] | None:
    """Guess a long-format (one row per well per date) column mapping.

    :returns: A mapping with a `"well"` key plus whichever of
        `"oil_rate"`, `"gas_rate"`, `"water_rate"`, `"pressure"`,
        `"water_cut"` and `"gor"` were found, or `None` if no
        well-identifier column (`Field`, `Well`, and similar) was found
        at all, meaning this table is not in long format.
    """
    well_column = next(
        (column for column in columns if WELL_ID_COLUMN_PATTERN.match(column.strip())), None
    )
    if well_column is None:
        return None

    mapping = {"well": well_column}
    pattern_by_key = {
        "oil_rate": OIL_RATE_PATTERN,
        "gas_rate": GAS_RATE_PATTERN,
        "water_rate": WATER_RATE_PATTERN,
        "pressure": PRESSURE_PATTERN,
        "water_cut": WATER_CUT_PATTERN,
        "gor": GOR_PATTERN,
    }
    for column in columns:
        if column == well_column:
            continue
        for key, pattern in pattern_by_key.items():
            if key not in mapping and pattern.search(column):
                mapping[key] = column
    return mapping


def load_observed_history(
    path: pathlib.Path | str,
    *,
    sheet_name: str | int = 0,
    date_column: str = "DATE",
    column_map: dict[str, str] | None = None,
    well_column: str | None = None,
    file_format: str | None = None,
    wells: list[str] | None = None,
) -> pandas.DataFrame:
    """Load the observed history file into a field-total scoring frame.

    Produces a frame with a `DATE` column (always named exactly that,
    regardless of what `date_column` names in the source file) plus one
    column named after each entry in `nagcsu.objective.SCORED_FIELD_VECTORS`
    (`FPR`, `FWCT`, `FGOR`), so it can be passed straight to
    `nagcsu.objective.score` alongside a res2df-produced simulated
    frame, which always uses the same `DATE` name. The date values
    themselves are parsed and normalized to midnight the same way
    `nagcsu.summary.load_summary` normalizes its own `DATE` column,
    since `objective.score` merges the two frames on an exact date
    match.

    Auto-detects whether `path` is long format (one row per well per
    date; see `detect_long_format_columns`) or wide format (one row per
    date, `WWCT_<WELL>`-style columns; see `guess_column_map`), unless
    `well_column` forces long format explicitly.

    :param date_column: Name of the date column in the source file
        itself. The returned frame's date column is always named `DATE`
        regardless of this (see above), so this only affects how
        `path` is read, not the result's shape.
    :param column_map: Explicit res2df-vector-name to workbook-column
        mapping for a wide-format table. Ignored for a long-format table.
    :param well_column: Explicit well-identifier column name, forcing
        long-format handling and skipping auto-detection. Set this if
        `detect_long_format_columns` does not recognize your well
        column's name.
    :param file_format: `"csv"` or `"excel"`, overriding the guess
        `detect_file_format` makes from `path`'s extension.
    :param wells: Well names to average over for field-total water cut
        and GOR in a wide-format table with no field-total column of
        its own. Unused for a long-format table, which aggregates every
        well present in the data regardless of this list.
    :raises FileNotFoundError: if `path` does not exist.
    :raises KeyError: if `date_column` is missing, or if neither format
        could find enough columns to compute FPR/FWCT/FGOR.
    """
    path = pathlib.Path(path)
    raw = read_raw_table(path, sheet_name=sheet_name, file_format=file_format)

    if date_column not in raw.columns:
        raise KeyError(
            f"Date column {date_column!r} not found in {path} (columns: {list(raw.columns)})"
        )

    long_format = detect_long_format_columns(list(raw.columns))
    if well_column is not None:
        long_format = {**(long_format or {}), "well": well_column}

    if long_format is not None:
        return load_long_format(raw, date_column=date_column, mapping=long_format, path=path)
    return load_wide_format(
        raw, date_column=date_column, column_map=column_map, wells=wells or [], path=path
    )


def load_long_format(
    raw: pandas.DataFrame, *, date_column: str, mapping: dict[str, str], path: pathlib.Path
) -> pandas.DataFrame:
    """Aggregate a one-row-per-well-per-date table into field totals.

    Water cut and GOR are computed from summed rates across every well
    reporting on a given date (`FWCT = sum(water_rate) / sum(oil_rate +
    water_rate)`, `FGOR = sum(gas_rate) / sum(oil_rate)`), matching how
    OPM Flow's own FWCT/FGOR summary vectors are defined, rather than
    averaging each well's own ratio. Falls back to averaging a
    precomputed `water_cut`/`gor` column only when the underlying rate
    columns were not found; a fallback `water_cut` column that looks
    like it is in percent (values above 1.5) is divided by 100, since
    OPM's FWCT is a 0-1 fraction.
    """
    frame = raw.copy()
    frame[date_column] = pandas.to_datetime(frame[date_column]).dt.normalize()
    grouped = frame.groupby(date_column, as_index=False)

    result = pandas.DataFrame({date_column: sorted(frame[date_column].unique())})

    if "pressure" not in mapping:
        raise KeyError(
            f"No pressure column found in {path} (looked for a column name containing "
            f"'pressure'); set history.column_map or check the file's headers."
        )
    pressure_by_date = (
        grouped[mapping["pressure"]].mean().set_index(date_column)[mapping["pressure"]]
    )
    result["FPR"] = result[date_column].map(pressure_by_date)

    if "oil_rate" in mapping and "water_rate" in mapping:
        oil_sum = grouped[mapping["oil_rate"]].sum().set_index(date_column)[mapping["oil_rate"]]
        water_sum = (
            grouped[mapping["water_rate"]].sum().set_index(date_column)[mapping["water_rate"]]
        )
        liquid_sum = oil_sum + water_sum
        fwct_by_date = (water_sum / liquid_sum.replace(0, pandas.NA)).fillna(0.0)
        result["FWCT"] = result[date_column].map(fwct_by_date)
    elif "water_cut" in mapping:
        watercut_by_date = (
            grouped[mapping["water_cut"]].mean().set_index(date_column)[mapping["water_cut"]]
        )
        if watercut_by_date.max() > 1.5:
            watercut_by_date = watercut_by_date / 100.0
        result["FWCT"] = result[date_column].map(watercut_by_date)
    else:
        raise KeyError(
            f"No oil/water rate columns or a water-cut column found in {path}; cannot "
            f"compute field water cut."
        )

    if "gas_rate" in mapping and "oil_rate" in mapping:
        oil_sum = grouped[mapping["oil_rate"]].sum().set_index(date_column)[mapping["oil_rate"]]
        gas_sum = grouped[mapping["gas_rate"]].sum().set_index(date_column)[mapping["gas_rate"]]
        fgor_by_date = (gas_sum / oil_sum.replace(0, pandas.NA)).fillna(0.0)
        result["FGOR"] = result[date_column].map(fgor_by_date)
    elif "gor" in mapping:
        gor_by_date = grouped[mapping["gor"]].mean().set_index(date_column)[mapping["gor"]]
        result["FGOR"] = result[date_column].map(gor_by_date)
    else:
        raise KeyError(
            f"No oil/gas rate columns or a GOR column found in {path}; cannot compute field GOR."
        )

    return result.rename(columns={date_column: OUTPUT_DATE_COLUMN})


def load_wide_format(
    raw: pandas.DataFrame,
    *,
    date_column: str,
    column_map: dict[str, str] | None,
    wells: list[str],
    path: pathlib.Path,
) -> pandas.DataFrame:
    """Read a one-row-per-date, `WWCT_<WELL>`-style table into field totals."""
    guessed = guess_column_map(list(raw.columns))
    resolved_map = {**guessed, **(column_map or {})}

    frame = pandas.DataFrame({date_column: pandas.to_datetime(raw[date_column]).dt.normalize()})

    if "FPR" not in resolved_map:
        raise KeyError(f"No field pressure column found or mapped in {path}")
    frame["FPR"] = raw[resolved_map["FPR"]]

    wells = wells or []
    frame["FWCT"] = get_field_average(raw, resolved_map, "WWCT", wells, path)
    frame["FGOR"] = get_field_average(raw, resolved_map, "WGOR", wells, path)
    return frame.rename(columns={date_column: OUTPUT_DATE_COLUMN})


def get_field_average(
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
    return typing.cast(pandas.Series, raw[well_columns].mean(axis=1))

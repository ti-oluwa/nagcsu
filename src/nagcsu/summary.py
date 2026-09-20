"""Reading OPM Flow summary output into a tidy scoring frame.

Uses res2df rather than reading `resfo`'s raw keyword/array pairs
directly, since `res2df.summary.df` already does the report-step to
tidy-DataFrame work Stage B.1 of the Execution Plan describes doing by
hand. `resfo` is still useful directly for a quick existence/shape check
without paying res2df's parsing cost; see :func:`peek_vectors`.
"""

import pathlib

import pandas
import resfo
from res2df import ResdataFiles, summary

from nagcsu.objective import SCORED_FIELD_VECTORS


def load_summary(
    case_basename: pathlib.Path | str, *, wells: list[str] | None = None
) -> pandas.DataFrame:
    """Load a run's summary output into a frame ready for `objective.score`.

    :param case_basename: Path to the run's output files, without
        extension, for example the `case_basename` on a
        :class:`nagcsu.simulate.RunResult`.
    :param wells: Well names to also pull `WWCT:<well>` and `WGOR:<well>`
        for, in case a caller wants a per-well breakdown alongside the
        field totals. The returned frame always has the field-total
        columns `FPR`, `FWCT`, `FGOR` regardless of `wells`.
    :returns: A frame with a `DATE` column plus `FPR`, `FWCT`, `FGOR`
        and, for each well in `wells`, `WWCT:<well>` and `WGOR:<well>`.
    """
    case_basename = pathlib.Path(case_basename)
    resdata_files = ResdataFiles(str(case_basename))
    well_vectors = [
        vector for well in (wells or []) for vector in (f"WWCT:{well}", f"WGOR:{well}")
    ]
    column_keys = list(SCORED_FIELD_VECTORS.values()) + well_vectors
    raw = summary.df(resdata_files, column_keys=column_keys)

    frame = raw.reset_index()
    frame["DATE"] = pandas.to_datetime(frame["DATE"]).dt.normalize()
    return frame


def peek_vectors(case_basename: pathlib.Path | str) -> list[str]:
    """List the summary vector names present in a run's `.SMSPEC`, cheaply.

    Reads only the `.SMSPEC` header via `resfo`, without decoding the
    full `.UNSMRY` time series, so this is safe to call just to check
    whether a run produced the vectors a caller needs before paying for
    a full :func:`load_summary`.
    """
    case_basename = pathlib.Path(case_basename)
    smspec_path = case_basename.with_suffix(".SMSPEC")
    keywords: list[str] = []
    for entry in resfo.lazy_read(smspec_path):
        if entry.read_keyword().strip() == "KEYWORDS":
            keywords = [value.strip() for value in entry.read_array()]
            break
    return keywords

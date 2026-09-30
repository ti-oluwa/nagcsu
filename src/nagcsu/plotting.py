"""Interactive plots: how a tuning session converged, and how a run's
simulated curves compare to the observed history.

Two kinds of plot, both built with `plotly` and saved as a single
self-contained HTML file (or a static image, if `kaleido` is installed):

- `convergence_figure`: J and each scored vector's NRMSE across a
  sequence of ledger trials, so a stalled or wasted search is visible
  at a glance instead of read off a wall of ledger JSON.
- `history_match_figure`: simulated vs observed pressure, water cut and
  GOR for one run, the same three vectors `nagcsu.objective.score`
  compares, so a mismatch is something you can see rather than infer
  from three NRMSE numbers.
"""

import pathlib
import typing

import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nagcsu import ledger, objective

VECTOR_TITLES: dict[str, str] = {
    "pressure": "Pressure (psia)",
    "watercut": "Water cut (fraction)",
    "gor": "GOR (Mscf/STB)",
}
"""Axis titles for each entry in `nagcsu.objective.SCORED_FIELD_VECTORS`."""


def plot_convergence(
    records: list[ledger.RunRecord],
    *,
    title: str = "Tuning convergence",
) -> go.Figure:
    """Plot J and each scored vector's NRMSE across a sequence of trials.

    One row per scored vector's NRMSE, plus a top row for the combined
    J, all sharing a "trial" x-axis (the position of each record in
    `records`, not a timestamp) but each with its own y-axis, since a
    single runaway vector's NRMSE (GOR after a bubble-point crossing is
    the usual case) would otherwise flatten the others to invisible
    lines on a shared scale.

    :param records: Ledger records in the order they should be plotted,
        typically `nagcsu.ledger.load(...)` already filtered to one
        strategy or group with `nagcsu.ledger.filter_records`. Records
        with no score (a failed trial, `j is None`) are skipped rather
        than breaking the trial axis.
    :param title: Overall figure title.
    :raises ValueError: if `records` has no scored entries to plot.
    """
    scored = [record for record in records if record.j is not None]
    if not scored:
        raise ValueError("No scored records to plot (every record has j=None).")

    vector_names = list(objective.SCORED_FIELD_VECTORS)
    figure = make_subplots(
        rows=1 + len(vector_names),
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Combined J", *(VECTOR_TITLES.get(name, name) for name in vector_names)],
        vertical_spacing=0.06,
    )

    trial_index = list(range(len(scored)))
    hover_text = [
        f"{record.run_id}<br>strategy={record.strategy}<br>group={record.group}<br>"
        f"{record.note}".strip()
        for record in scored
    ]

    figure.add_trace(
        go.Scatter(
            x=trial_index,
            y=[record.j for record in scored],
            mode="lines+markers",
            name="J",
            text=hover_text,
            hovertemplate="trial %{x}<br>J=%{y:.4g}<br>%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    for row, vector_name in enumerate(vector_names, start=2):
        y_values = [
            record.vector_nrmse.get(vector_name) if record.vector_nrmse else None
            for record in scored
        ]
        figure.add_trace(
            go.Scatter(
                x=trial_index,
                y=y_values,
                mode="lines+markers",
                name=f"NRMSE({vector_name})",
                text=hover_text,
                hovertemplate=f"trial %{{x}}<br>{vector_name} NRMSE=%{{y:.4g}}<br>%{{text}}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    figure.update_xaxes(title_text="Trial", row=1 + len(vector_names), col=1)
    figure.update_layout(title=title, showlegend=False, height=280 * (1 + len(vector_names)))
    return figure


def plot_match(
    simulated: pandas.DataFrame,
    observed: pandas.DataFrame,
    *,
    title: str = "Simulated vs observed",
    date_column: str = "DATE",
) -> go.Figure:
    """Plot simulated vs observed pressure, water cut and GOR over time.

    One row per entry in `nagcsu.objective.SCORED_FIELD_VECTORS`, each
    with a simulated and an observed trace sharing a date x-axis. Both
    frames are used exactly as `nagcsu.objective.score` reads them, so
    this shows precisely what the NRMSE numbers were computed from.

    :param simulated: A `nagcsu.summary.load_summary` frame.
    :param observed: A `nagcsu.pipeline.load_observed_history` frame.
    """
    vector_names = list(objective.SCORED_FIELD_VECTORS)
    figure = make_subplots(
        rows=len(vector_names),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[VECTOR_TITLES.get(name, name) for name in vector_names],
        vertical_spacing=0.08,
    )

    for row, name in enumerate(vector_names, start=1):
        column = objective.SCORED_FIELD_VECTORS[name]
        if column in simulated.columns:
            figure.add_trace(
                go.Scatter(
                    x=simulated[date_column],
                    y=simulated[column],
                    mode="lines",
                    name="Simulated",
                    legendgroup="simulated",
                    showlegend=(row == 1),
                    line={"color": "#1f77b4"},
                ),
                row=row,
                col=1,
            )
        if column in observed.columns:
            figure.add_trace(
                go.Scatter(
                    x=observed[date_column],
                    y=observed[column],
                    mode="markers",
                    name="Observed",
                    legendgroup="observed",
                    showlegend=(row == 1),
                    marker={"color": "#d62728", "size": 5},
                ),
                row=row,
                col=1,
            )

    figure.update_xaxes(title_text="Date", row=len(vector_names), col=1)
    figure.update_layout(title=title, height=280 * len(vector_names))
    return figure


def plot_wells_match(
    simulated: pandas.DataFrame,
    observed: pandas.DataFrame,
    wells: typing.Sequence[str],
    *,
    title: str = "Per-well simulated vs observed",
    date_column: str = "DATE",
) -> go.Figure:
    """Plot each well's simulated and observed water cut and GOR.

    One row per well, water cut on the left and GOR on the right. A well
    with no history or no simulated column for a vector leaves that panel
    empty rather than failing, so a partly covered history still plots.

    :param simulated: A `nagcsu.summary.load_summary` frame loaded with `wells`.
    :param observed: A `nagcsu.pipeline.load_observed_history` frame.
    :param wells: Deck well names to plot, in order.
    :raises ValueError: if `wells` is empty.
    """
    if not wells:
        raise ValueError("No wells to plot.")
    titles = [
        text
        for well in wells
        for text in (f"{well}: water cut (fraction)", f"{well}: GOR (Mscf/STB)")
    ]
    figure = make_subplots(
        rows=len(wells), cols=2, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.04
    )
    for row, well in enumerate(wells, start=1):
        for col, prefix in enumerate(("WWCT", "WGOR"), start=1):
            column = f"{prefix}:{well}"
            if column in simulated.columns:
                figure.add_trace(
                    go.Scatter(
                        x=simulated[date_column],
                        y=simulated[column],
                        mode="lines",
                        name="Simulated",
                        legendgroup="simulated",
                        showlegend=(row == 1 and col == 1),
                        line={"color": "#1f77b4"},
                    ),
                    row=row,
                    col=col,
                )
            if column in observed.columns:
                figure.add_trace(
                    go.Scatter(
                        x=observed[date_column],
                        y=observed[column],
                        mode="markers",
                        name="Observed",
                        legendgroup="observed",
                        showlegend=(row == 1 and col == 1),
                        marker={"color": "#d62728", "size": 4},
                    ),
                    row=row,
                    col=col,
                )
    figure.update_layout(title=title, height=230 * len(wells))
    return figure


def save_figure(figure: go.Figure, output_path: pathlib.Path | str) -> pathlib.Path:
    """Save a figure as a self-contained interactive HTML, or a static image.

    Dispatches on `output_path`'s suffix: `.html` (or no suffix, which
    gets `.html` added) writes an interactive file with `plotly.js`
    embedded, viewable offline with no server and no network access.
    Any other suffix (`.png`, `.svg`, `.pdf`, ...) writes a static image
    at 2x resolution via `kaleido`, which is not installed by default;
    a plain `ImportError` from that path is re-raised with a one-line
    install hint rather than a wall of plotly's own traceback.
    """
    output_path = pathlib.Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".html":
        figure.write_html(output_path, include_plotlyjs=True, full_html=True)
        return output_path

    try:
        figure.write_image(output_path, scale=2)
    except (ValueError, RuntimeError) as error:
        message = str(error).lower()
        if "kaleido" not in message and "chrome" not in message:
            raise
        raise ImportError(
            "Static image export needs the optional 'kaleido' package, and kaleido "
            "itself needs Chrome: pip install kaleido, then run `plotly_get_chrome` "
            "if prompted. Use an .html output path to skip both."
        ) from error
    return output_path

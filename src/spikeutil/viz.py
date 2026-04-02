import numpy as np
import pandas as pd
import plotly.graph_objects as go

from spikeutil.core import spikes_as_df


def plot_chip(analyzer, chip_width, chip_height, cell_type=None, colormap=None):
    fig = go.Figure()

    probe = analyzer.get_probe().to_dataframe()
    trace1 = go.Scatter(
        x=probe["x"],
        y=probe["y"],
        mode="markers",
        marker=dict(color="black", size=1),
        name="channels",
    )
    fig.add_trace(trace1)

    unit_pos = analyzer.get_extension("unit_locations").data["unit_locations"]
    if cell_type is None:
        cell_type = np.array(["Unit"] * len(unit_pos))
    for t in np.unique(cell_type):
        marker = None
        if colormap is not None:
            marker = dict(color=colormap[t])
        trace = go.Scatter(
            mode="markers",
            x=unit_pos[cell_type == t, 0],
            y=unit_pos[cell_type == t, 1],
            name=t,
            hovertext=analyzer.unit_ids,
            marker=marker,
        )
        fig.add_trace(trace)

    fig.update_xaxes(range=[0, chip_width])
    fig.update_yaxes(
        range=[0, chip_height],
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        xaxis_title="X pos. (μm)",
        yaxis_title="Y pos. (μm)",
    )
    return fig


def plot_spikes(analyzer, t_max=60, color=None):
    sorting = analyzer.sorting
    spikes = spikes_as_df(sorting)
    spikes = spikes[spikes["time"] <= t_max]
    if color is not None:
        color = spikes["unit_index"].map(lambda i: color[i])

    x_pos = analyzer.get_extension("unit_locations").get_data()[:, 0]
    order = np.argsort(np.argsort(-x_pos))
    y = [order[i] for i in spikes["unit_index"]]

    fig = go.Figure()
    trace = go.Scatter(
        x=spikes["time"],
        y=y,
        mode="markers",
        marker=dict(size=2, color="black"),
        marker_color=color,
    )
    fig.add_trace(trace)
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Unit (X pos. rank)",
    )
    return fig

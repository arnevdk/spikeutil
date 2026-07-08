import numpy as np
import plotly.graph_objects as go

from spikeutil.core import spikes_as_df


def mea_traces(analyzer, cell_type=None, colormap=None, legend=True):
    traces = []

    probe = analyzer.get_probe().to_dataframe()
    trace1 = go.Scattergl(
        x=probe["x"],
        y=probe["y"],
        mode="markers",
        marker=dict(color="black", size=1),
        name="channels",
        legendgroup="channels",
        text=probe['contact_ids'],
        showlegend=legend,
    )
    traces.append(trace1)

    unit_pos = analyzer.get_extension("unit_locations").data["unit_locations"]
    if cell_type is None:
        cell_type = np.array(["Unit"] * len(unit_pos))
    for t in np.unique(cell_type):
        marker = None
        if colormap is not None:
            marker = dict(color=colormap[t])
        trace = go.Scattergl(
            mode="markers",
            x=unit_pos[cell_type == t, 0],
            y=unit_pos[cell_type == t, 1],
            name=t,
            text=analyzer.unit_ids,
            marker=marker,
            legendgroup=t,
            showlegend=legend,
        )
        traces.append(trace)

    # fig.update_xaxes(range=[0, chip_width])
    # fig.update_yaxes(
    #    range=[0, chip_height],
    #    scaleanchor="x",
    #    scaleratio=1,
    # )
    # fig.update_layout(
    #    xaxis_title="X pos. (μm)",
    #    yaxis_title="Y pos. (μm)",
    # )
    return traces


def spike_raster_traces(analyzer, order=None, t_max=None, cell_type=None, colormap=None, legend=True):
    sorting = analyzer.sorting
    spikes = spikes_as_df(sorting)
    if t_max is not None:
        spikes = spikes[spikes["time"] <= t_max]

    if cell_type is None:
        cell_type = ["spikes"]
        colormap = {"spikes": "black"}

    if order is None:
        order = np.arange(len(analyzer.unit_ids))
    y = np.array([order[i] for i in spikes["unit_index"]])

    traces = []
    for ct in np.unique(cell_type):
        ct_idc = cell_type[spikes["unit_index"]] == ct
        trace = go.Scattergl(
            x=spikes["time"].iloc[ct_idc],
            y=y[ct_idc],
            mode="markers",
            marker=dict(size=2, color=colormap[ct]),
            legendgroup=ct,
            showlegend=legend,
            name=ct,
        )
        traces.append(trace)
    return traces

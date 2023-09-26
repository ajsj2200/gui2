
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.io as pio
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler, FigureWidgetResampler
import numpy as np
from plotly.colors import qualitative

pio.renderers.default = "plotly_mimetype+notebook"


def visualing(
    data, scaler='minmax', port=1234, height=600, width=1000,
    show=False, verbose=True, widget=False
):
    # data가 numpy array일 경우 dataframe으로 변환
    if type(data) == np.ndarray:
        data = pd.DataFrame(data)
    df = data.copy()
    if scaler == "standard":
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=data.columns)

    elif scaler == "minmax":
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=data.columns)

    if widget:
        fig = FigureWidgetResampler(go.Figure())
    else:
        fig = FigureResampler(go.Figure())

    # Available color palette
    colors = qualitative.Plotly

    # Ensure there are enough colors by repeating the palette
    num_columns = len(data.columns)
    colors = colors * (num_columns // len(colors)) + \
        colors[:num_columns % len(colors)]

    if verbose:
        for idx, c in enumerate(tqdm(data.columns)):
            fig.add_trace(
                go.Scattergl(name=c, line=dict(color=colors[idx])),
                hf_x=data[c].index,
                hf_y=df[c],
            )
    else:
        for idx, c in enumerate(data.columns):
            fig.add_trace(
                go.Scattergl(name=c, line=dict(color=colors[idx])),
                hf_x=data[c].index,
                hf_y=df[c],
            )

    fig.update_layout(
        height=height,
        width=width,
    )

    if show:
        fig.show(
            mode="inline",
            port=port,
            config={
                "modeBarButtonsToAdd": [
                    "drawline",
                    "drawopenpath",
                    "drawclosedpath",
                    "drawcircle",
                    "drawrect",
                    "eraseshape",
                ]
            },
        )
    else:
        return fig


def visualing_subplots(
    data,
    scaler='minmax',
    shared_xaxes=False,
    port=5555,
    horizontal_spacing=0.05,
    vertical_spacing=0.05,
    height=600,
    width=1000,
    show=False,
):
    fig = FigureWidgetResampler(
        make_subplots(
            rows=len(data),
            cols=1,
            shared_xaxes=shared_xaxes,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )
    )
    for i in range(len(data)):
        if type(data[i]) == np.ndarray:
            data[i] = pd.DataFrame(data[i])
        df = data[i].copy()
        if scaler == "standard":
            scale = StandardScaler()
            df = pd.DataFrame(
                scale.fit_transform(df), columns=df.columns, index=df.index
            )
        elif scaler == "minmax":
            scale = MinMaxScaler()
            df = pd.DataFrame(
                scale.fit_transform(df), columns=df.columns, index=df.index
            )
        for c in tqdm(df.columns[:]):
            fig.add_trace(
                go.Scattergl(
                    name=c,
                ),
                hf_x=df[c].index,
                hf_y=df[c],
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        height=height,
        width=width,
    )
    if show == True:
        fig.show(
            mode="inline",
            port=port,
            config={
                "modeBarButtonsToAdd": [
                    "drawline",
                    "drawopenpath",
                    "drawclosedpath",
                    "drawcircle",
                    "drawrect",
                    "eraseshape",
                ]
            },
        )
    else:
        return fig


def visualing_subplots_group(
    data,
    scaler='minmax',
    shared_xaxes=False,
    port=5555,
    horizontal_spacing=0.05,
    vertical_spacing=0.05,
    height=600,
    width=1000,
    show=False,
):
    fig = FigureWidgetResampler(
        make_subplots(
            rows=len(data),
            cols=1,
            shared_xaxes=shared_xaxes,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )
    )
    for i in range(len(data)):
        if type(data[i]) == np.ndarray:
            data[i] = pd.DataFrame(data[i])
        df = data[i].copy()
        if scaler == "standard":
            scale = StandardScaler()
            df = pd.DataFrame(
                scale.fit_transform(df), columns=df.columns, index=df.index
            )
        elif scaler == "minmax":
            scale = MinMaxScaler()
            df = pd.DataFrame(
                scale.fit_transform(df), columns=df.columns, index=df.index
            )
        for c in tqdm(df.columns[:]):
            fig.add_trace(
                go.Scattergl(name=c, legendgroup=c),
                hf_x=df[c].index,
                hf_y=df[c],
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        height=height,
        width=width,
    )
    if show == True:
        fig.show(
            mode="inline",
            port=port,
            config={
                "modeBarButtonsToAdd": [
                    "drawline",
                    "drawopenpath",
                    "drawclosedpath",
                    "drawcircle",
                    "drawrect",
                    "eraseshape",
                ]
            },
        )
    else:
        return fig


def visualing_subplots_noresample(
    data,
    scaler=None,
    shared_xaxes=False,
    port=5555,
    horizontal_spacing=0.05,
    vertical_spacing=0.05,
    height=600,
    width=1800,
    show=True,
):
    fig = make_subplots(
        rows=len(data),
        cols=1,
        shared_xaxes=shared_xaxes,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    for i in range(len(data)):
        if type(data[i]) == np.ndarray:
            data[i] = pd.DataFrame(data[i])
        df = data[i].copy()
        if scaler == "standard":
            scale = StandardScaler()
            df = pd.DataFrame(
                scale.fit_transform(df), columns=df.columns, index=df.index
            )
        elif scaler == "minmax":
            scale = MinMaxScaler()
            df = pd.DataFrame(
                scale.fit_transform(df), columns=df.columns, index=df.index
            )
        for c in tqdm(df.columns[:]):
            fig.add_trace(
                go.Scattergl(
                    x=df[c].index,
                    y=df[c],
                    name=c,
                ),
                hf_x=df[c].index,
                hf_y=df[c],
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        height=height,
        width=width,
    )
    if show == True:
        fig.show(
            mode="inline",
            port=port,
        )
    else:
        return fig

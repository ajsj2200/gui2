from typing import List

import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler, FigureWidgetResampler

device = 'cuda'

def read_interpolate_filter_scale_select(filename: str) -> pd.DataFrame:
    """
    Read a csv file, interpolate missing values, apply a Savitzky-Golay filter,
    scale the data using MinMaxScaler, and select specific columns.

    Args:
        filename (str): path to the csv file

    Returns:
        pd.DataFrame: a processed dataframe with selected columns
    """
    # Read the csv file and interpolate missing values
    df = pd.read_csv(filename, encoding="euc_kr").iloc[:, 1:8].interpolate()

    # Apply a Savitzky-Golay filter to each column
    for col_idx in tqdm(range(df.shape[1])):
        df.iloc[:, col_idx] = savgol_filter(
            df.iloc[:, col_idx], window_length=201, polyorder=2
        )

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    df.iloc[:, :] = scaler.fit_transform(df.values)

    # Replace a specific range of values in column 1
    df.iloc[277289:277289+651, 1] = df.iloc[11926:12577, 1].values

    # Select specific columns
    df = df.iloc[:300000, [0, 1, 3, 4, 5, 6]]

    return df

from sklearn.preprocessing import StandardScaler, MinMaxScaler
def visualing(data, scaler=None, port=1234, height=600, width=1800, show=True):
    df = data.copy()
    if scaler == "standard":
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=data.columns)

    elif scaler == "minmax":
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=data.columns)
    fig = FigureResampler(go.Figure())

    for c in tqdm(data.columns[:]):
        fig.add_trace(
            go.Scatter(
                x=data[c].index,
                y=df[c],
                name=c,
            )
        )
    fig.update_layout(
        height=height,
        width=width,
    )
    if show == True:
        fig.show_dash(
            mode="inline_persistent",
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
    scaler=None,
    shared_xaxes=False,
    port=5555,
    horizontal_spacing=0.05,
    vertical_spacing=0.05,
    height=600,
    width=1800,
    show=True,
):
    fig = FigureResampler(
        make_subplots(
            rows=len(data),
            cols=1,
            shared_xaxes=shared_xaxes,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )
    )
    for i in range(len(data)):
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
                go.Scatter(
                    x=df[c].index,
                    y=df[c],
                    name=c,
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        height=height,
        width=width,
    )
    if show == True:
        fig.show_dash(
            mode="inline_persistent",
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
    scaler=None,
    shared_xaxes=False,
    port=5555,
    horizontal_spacing=0.05,
    vertical_spacing=0.05,
    height=600,
    width=1800,
    show=True,
):
    fig = FigureResampler(
        make_subplots(
            rows=len(data),
            cols=1,
            shared_xaxes=shared_xaxes,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )
    )
    for i in range(len(data)):
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
                go.Scatter(x=df[c].index, y=df[c], name=c, legendgroup=c),
                row=i + 1,
                col=1,
            )

    fig.update_layout(
        height=height,
        width=width,
    )
    if show == True:
        fig.show_dash(
            mode="inline_persistent",
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
                go.Scatter(
                    x=df[c].index,
                    y=df[c],
                    name=c,
                ),
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
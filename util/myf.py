import tensorflow as tf
from tqdm import tqdm
from jax import jit, vmap, pmap
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.io as pio
from typing import List

import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler, FigureWidgetResampler
import numpy as np

device = "cuda"

pio.renderers.default = "plotly_mimetype+notebook"


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
    df = pd.read_csv(filename, encoding="euc_kr")
    df.index = pd.to_datetime(df["일시"], format="%Y-%m-%d %I:%M:%S %p")
    df.drop("일시", axis=1, inplace=True)
    df = df.interpolate().iloc[:, :7]
    df = df["2021-04-12 9:23:23":"2021-04-17 16:32:09"]

    # Apply a Savitzky-Golay filter to each column
    for col_idx in tqdm(range(df.shape[1])):
        df.iloc[:, col_idx] = savgol_filter(
            df.iloc[:, col_idx], window_length=200, polyorder=2
        )

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    df.iloc[:, :] = scaler.fit_transform(df.values)

    # Replace a specific range of values in column 1
    # df.iloc[277289:277289+651, 1] = df.iloc[11926:12577, 1].values

    # Select specific columns
    df = df.iloc[:, [0, 1, 3, 4, 5, 6]]
    df = df.resample("1S").mean()

    return df


def visualing(
    data, scaler='minmax', port=1234, height=600, width=1000, show=False, verbose=True
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
    fig = FigureWidgetResampler(go.Figure())

    if verbose:
        for c in tqdm(data.columns[:]):
            fig.add_trace(
                go.Scattergl(name=c),
                hf_x=data[c].index,
                hf_y=df[c],
            )
    else:
        for c in data.columns[:]:
            fig.add_trace(
                go.Scattergl(name=c),
                hf_x=data[c].index,
                hf_y=df[c],
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


@jit
def mse(Q, T):
    return jnp.mean((Q - T) ** 2)


@jit
def mae(Q, T):
    return jnp.mean(jnp.abs(Q - T))


def compute_distance_for_shard_mse(batch_data_shard, data):
    batch_size = batch_data_shard.shape[0]
    distance_matrix_batch = jnp.zeros((batch_size, len(data)))
    for i in range(batch_size):
        query = batch_data_shard[i]
        vmap_mse = vmap(mse, in_axes=(None, 0))
        distance_matrix_batch = distance_matrix_batch.at[i].set(
            vmap_mse(query, data))
    return distance_matrix_batch


def compute_distance_for_shard_mae(batch_data_shard, data):
    batch_size = batch_data_shard.shape[0]
    distance_matrix_batch = jnp.zeros((batch_size, len(data)))
    for i in range(batch_size):
        query = batch_data_shard[i]
        vmap_mae = vmap(mae, in_axes=(None, 0))
        distance_matrix_batch = distance_matrix_batch.at[i].set(
            vmap_mae(query, data))
    return distance_matrix_batch


def compute_distance_matrix(data, metric="mse", batch_size=64):
    # pmap을 사용하여 함수를 병렬화합니다.

    if metric == "mse":
        parallel_compute_distance = pmap(
            compute_distance_for_shard_mse, in_axes=(0, None))
    elif metric == "mae":
        parallel_compute_distance = pmap(
            compute_distance_for_shard_mae, in_axes=(0, None))

    # TensorFlow tf.data를 사용하여 Dataset과 DataLoader를 구성합니다.
    batch_size_per_gpu = batch_size
    total_batch_size = batch_size_per_gpu * 3  # 3 GPUs
    dataset = (
        tf.data.Dataset.from_tensor_slices(data)
        .batch(total_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    dataloader = iter(dataset)

    # 결과를 저장할 행렬을 초기화합니다.
    distance_matrix = np.zeros((len(data), len(data)))

    # 배치별로 거리 계산을 실행합니다.
    for i in tqdm(range(len(dataset))):
        batch = next(dataloader).numpy()

        # Determine shard size and pad the batch if necessary
        shard_size = batch.shape[0] // 3
        remainder = batch.shape[0] % 3
        if remainder:
            padding_rows_needed = 3 - remainder
            padding = np.zeros((padding_rows_needed, batch.shape[1]))
            batch = np.vstack([batch, padding])

        shards = jnp.stack(np.array_split(batch, 3))

        distance_batches = parallel_compute_distance(shards, jnp.array(data))
        distance_batch = np.concatenate(distance_batches, axis=0)
        # Remove the padded rows from the distance matrix
        if remainder:
            distance_batch = distance_batch[: -shard_size + remainder]
        start_idx = i * total_batch_size
        end_idx = start_idx + distance_batch.shape[0]
        distance_matrix[start_idx:end_idx] = distance_batch

    return distance_matrix

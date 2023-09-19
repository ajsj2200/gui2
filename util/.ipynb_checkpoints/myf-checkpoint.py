from typing import List

import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import torch
from torch_geometric.data import Data, Batch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler, FigureWidgetResampler

device = 'cuda'
import plotly.io as pio
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
    df = pd.read_csv(filename, encoding="euc_kr").iloc[:, 1:8].interpolate()

    # Apply a Savitzky-Golay filter to each column
    for col_idx in tqdm(range(df.shape[1])):
        df.iloc[:, col_idx] = savgol_filter(
            df.iloc[:, col_idx], window_length=200, polyorder=2
        )

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    df.iloc[:, :] = scaler.fit_transform(df.values)

    # Replace a specific range of values in column 1
    df.iloc[277289:277289+651, 1] = df.iloc[11926:12577, 1].values

    # Select specific columns
    df = df.iloc[:300000, [0, 1, 3, 4, 5, 6]]

    return df

def get_k_neighbors(input_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute the k-nearest neighbors for an input matrix.

    Args:
        input_matrix (torch.Tensor): input matrix to compute neighbors for
        k (int): number of neighbors to compute

    Returns:
        torch.Tensor: a tensor of shape (2, num_neighbors) containing indices of the k-nearest neighbors
    """
    cosine_similarity = (
        torch.mm(input_matrix, input_matrix.t())
        / torch.norm(input_matrix, dim=1).reshape(-1, 1)
        / torch.norm(input_matrix, dim=1)
    )
    values, indices = torch.topk(cosine_similarity, k=k)
    node_indices = (
        torch.arange(0, cosine_similarity.shape[0], dtype=torch.long)
        .repeat(k, 1)
        .to(device)
    )
    return torch.stack([node_indices.flatten(), indices.flatten()], axis=1).t()


def get_neighbors(input_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity between rows of an input matrix.

    Args:
        input_matrix (torch.Tensor): input matrix to compute cosine similarity for

    Returns:
        torch.Tensor: a tensor of shape (num_rows, num_rows) containing cosine similarities
    """
    cosine_similarity = (
        torch.mm(input_matrix, input_matrix.t())
        / torch.norm(input_matrix, dim=1).reshape(-1, 1)
        / torch.norm(input_matrix, dim=1)
    )
    return cosine_similarity

def get_k_matrix(cosine_similarity: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return a k x 2 matrix containing the indices of the top k values in the cosine_similarity matrix.
    
    Args:
    - cosine_similarity: a square matrix of cosine similarities between nodes
    - k: the number of top values to retrieve
    
    Returns:
    - a k x 2 matrix containing the node indices and the corresponding indices of the top k values
    """
    values, indices = torch.topk(cosine_similarity, k=k)
    node_indices = torch.arange(0, cosine_similarity.shape[0], dtype=torch.long).repeat(k, 1).to(device)
    return torch.stack([node_indices.flatten(), indices.flatten()], axis=1).t()


def get_k_neighbors_batch(input_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns the top k neighbors for each node in the input matrix based on cosine similarity.

    Args:
        input_matrix (torch.Tensor): A tensor of shape (batch_size, num_nodes, num_features) representing
                                      the input matrix.
        k (int): The number of neighbors to return for each node.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_nodes * k, 2) containing the indices of the
                      top k neighbors for each node.
    """
    batch_size, num_nodes, num_features = input_matrix.shape
    
    # Compute cosine similarity between all pairs of nodes
    cosine_similarity = torch.bmm(input_matrix, input_matrix.permute(0, 2, 1)) \
                        / torch.norm(input_matrix, dim=2).reshape(batch_size, num_nodes, 1) \
                        / torch.norm(input_matrix, dim=2).reshape(batch_size, 1, num_nodes)
    
    # Get the indices of the top k neighbors for each node
    values, indices = torch.topk(cosine_similarity, k=k, dim=2)
    
    # Create a tensor of the node indices and their corresponding neighbor indices
    node_indices = torch.arange(0, num_nodes, dtype=torch.long) \
                   .repeat(batch_size, k, 1) \
                   .permute(0, 2, 1) \
                    .to('cuda')
    values = torch.stack([node_indices.flatten(), indices.flatten()], axis=1)
    values = values.reshape((batch_size, num_nodes * k, 2))
    
    return values

def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the number of trainable parameters in the given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count the parameters of.

    Returns:
        The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
def visualing(data, scaler=None, port=1234, height=600, width=1800, show=False):
    df = data.copy()
    if scaler == "standard":
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=data.columns)

    elif scaler == "minmax":
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=data.columns)
    fig = FigureWidgetResampler(go.Figure())

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
    scaler=None,
    shared_xaxes=False,
    port=5555,
    horizontal_spacing=0.05,
    vertical_spacing=0.05,
    height=600,
    width=1800,
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
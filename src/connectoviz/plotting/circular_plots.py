# src/connectoviz/plotting/circular_plots.py

from typing import Union, List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from connectoviz.core.connectome import Connectome
from connectoviz.visualization.circular_layout import compute_layout
from connectoviz.visualization.ring_tracks import add_ring_tracks
from connectoviz.visualization.styling import apply_styling


def plot_circular_connectome(
    con_mat: Union[str, list, "np.ndarray", "pd.DataFrame"],
    node_metadata: Union[str, Dict[str, List], "pd.DataFrame"],
    tracks: Optional[List[str]] = None,
    group_by: Optional[List[str]] = None,
    atlas: Optional[str] = None,
    cmap: str = "viridis",
    node_size: int = 100,
    figsize: tuple = (10, 10),
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    One-liner function to plot circular connectome with multi-track metadata rings.

    Parameters
    ----------
    con_mat : array-like or path
        2D connectivity matrix or filepath to it.
    node_metadata : dict, DataFrame, or path
        Metadata containing node info like lobe, hemisphere, etc.
    tracks : list of str
        Metadata columns to plot as outer rings.
    group_by : list of str
        Metadata columns to define grouping (e.g., ["lobe", "hemisphere"])
    atlas : str
        Optional name of the brain atlas.
    cmap : str
        Colormap to use.
    node_size : int
        Size of nodes.
    figsize : tuple
        Size of the figure.
    title : str
        Title of the figure.
    show : bool
        Whether to display the figure immediately.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure.
    """

    # Step 1: Construct and validate connectome
    connectome = Connectome(con_mat=con_mat, node_metadata=node_metadata, atlas=atlas)

    # Step 2: Create figure and compute layout
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    node_positions = compute_layout(connectome, group_by=group_by)

    # Step 3: Draw connectivity edges (basic placeholder, refined later)
    for i, pos_i in enumerate(node_positions):
        for j, pos_j in enumerate(node_positions):
            if i < j and connectome.con_mat[i, j] != 0:
                ax.plot([pos_i, pos_j], [1, 1], color="gray", alpha=0.3)

    # Step 4: Add ring tracks for metadata visualization
    if tracks:
        add_ring_tracks(ax, connectome, node_positions, tracks, cmap)

    # Step 5: Styling and aesthetics
    apply_styling(ax, title=title, node_size=node_size)

    if show:
        plt.show()

    return fig

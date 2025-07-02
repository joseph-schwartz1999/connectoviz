# connectoviz/plotting/circular_plots.py

from typing import Optional, List, Union
import numpy as np
import pandas as pd

from connectoviz.core.connectome import Connectome
from connectoviz.visualization.circular_plot_builder import CircularPlotBuilder


def plot_circular_connectome(
    con_mat: np.ndarray,
    atlas: pd.DataFrame,
    metadata_df: pd.DataFrame,
    hemispheric_par: bool = False,
    group_by: Optional[str] = None,
    tracks: Optional[List[str]] = None,
    index_mapping: Optional[Union[dict, pd.DataFrame]] = None,
    weights: Optional[np.ndarray] = None,
    # Styling kwargs
    cmap: str = "coolwarm",
    gap: float = 2.0,
    figsize: tuple = (10, 10),
    start_angle: float = 0.0,
    edge_threshold: Optional[float] = None,
    show_labels: bool = True,
    **kwargs,
):
    """
    Main API to generate a circular connectome plot.

    Parameters
    ----------
    con_mat : np.ndarray
        Square connectivity matrix (n x n).
    atlas : pd.DataFrame
        Atlas reference table.
    metadata_df : pd.DataFrame
        Node-level metadata (same length as con_mat).
    hemispheric_par : bool
        Whether to use symmetrical hemisphere-based layout.
    group_by : str, optional
        Metadata column to group nodes by.
    tracks : list of str, optional
        Metadata columns to draw as concentric rings.
    index_mapping : dict or pd.DataFrame, optional
        Optional remapping of node indices to labels.
    weights : np.ndarray, optional
        Matrix of same shape as con_mat to apply as a mask.
    cmap : str
        Colormap name for edges and tracks.
    gap : float
        Angular gap between groups.
    figsize : tuple
        Size of the figure.
    start_angle : float
        Starting angle in degrees.
    edge_threshold : float, optional
        Threshold to mask weak edges.
    show_labels : bool
        Whether to annotate nodes.
    kwargs : dict
        Other styling overrides.
    """

    # Step 1: Construct the Connectome
    connectome = Connectome.from_inputs(
        con_mat=con_mat, atlas=atlas, node_metadata=metadata_df, mapping=index_mapping
    )

    # Step 2: Apply weights/mask if provided
    if weights is not None:
        connectome.apply_mask(weights)

    # Step 3: Apply hemisphere reordering if requested
    if hemispheric_par:
        connectome.reorder_by_hemisphere()

    # Step 4: Validate track and group_by fields
    if tracks:
        missing = [t for t in tracks if t not in connectome.node_metadata.columns]
        if missing:
            raise ValueError(f"Tracks not found in metadata: {missing}")

    if group_by and group_by not in connectome.node_metadata.columns:
        raise ValueError(f"group_by column '{group_by}' not found in metadata.")

    # Step 5: Build and render the circular plot
    builder = CircularPlotBuilder(
        connectome=connectome,
        tracks=tracks,
        group_by=[group_by] if group_by else [],
        cmap=cmap,
        gap=gap,
        figsize=figsize,
        start_angle=start_angle,
        edge_threshold=edge_threshold,
        show_labels=show_labels,
        **kwargs,
    )

    fig = builder.build()
    return fig

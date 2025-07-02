import numpy as np
from typing import List, Dict
import pandas as pd


def compute_layout(
    node_order: List[str],
    group_by: List[str],
    metadata: pd.DataFrame,
    gap: float = 2.0,
    start_angle: float = 0.0,
) -> Dict[str, float]:
    """
    Compute angular positions for each node in the circular layout.

    Parameters
    ----------
    node_order : list of str
        Ordered list of node indices.
    group_by : list of str
        Metadata columns to define groupings.
    metadata : pd.DataFrame
        Metadata with node group info.
    gap : float
        Angular gap (in degrees) between groups.
    start_angle : float
        Starting angle in degrees.

    Returns
    -------
    dict
        Mapping from node index to angular position (in radians).
    """
    total_nodes = len(node_order)
    node_positions = {}

    # Group nodes by group_by columns
    if group_by:
        grouped = metadata.loc[node_order].groupby(group_by, sort=False).groups
    else:
        grouped = {"all": node_order}

    # Compute total gap space
    n_groups = len(grouped)
    total_gap_deg = gap * n_groups
    total_arc_deg = 360.0 - total_gap_deg
    angle_per_node = total_arc_deg / total_nodes
    current_angle = start_angle

    for group_nodes in grouped.values():
        for node in group_nodes:
            node_positions[node] = np.deg2rad(current_angle)
            current_angle += angle_per_node
        current_angle += gap  # Add group gap

    return node_positions

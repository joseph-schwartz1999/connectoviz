import pandas as pd
import numpy as np
from random import randint
import networkx as nx
import matplotlib.pyplot as plt
import math
from matplotlib.path import Path as MplPath
import matplotlib.patches as patches
from pathlib import Path


def load_data(
    connectivity_matrix_path,
    atlas_path,
    grouping_name="Lobe",
    label="Label",
    roi_names="ROIname",
    hemisphere="Hemi",
    left_symbol="L",
    right_symbol="R",
    metadata=None,
    display_node_names: bool = False,
    display_group_names: bool = False,
):
    """
    Now returns:
      connectivity_matrix, groups, metadata_map,
      row_names_map, display_node_names, display_group_names

    Modified to allow atlases that contain ONLY left and right (no 'else'),
    while still supporting a third 'else' hemisphere if present.
    """
    conn = pd.read_csv(connectivity_matrix_path, header=None).values
    atlas = pd.read_csv(atlas_path)

    # basic shape & column checks
    n, m = conn.shape
    num_rois = atlas[label].max()
    if n != m or n != num_rois:
        raise ValueError("Connectivity matrix size must match atlas labels.")

    for col in (grouping_name, label, roi_names, hemisphere, metadata):
        if col not in atlas.columns:
            raise ValueError(f"Atlas missing required column '{col}'")

    # build metadata and name maps
    metadata_map = dict(zip(atlas[label] - 1, atlas[metadata]))
    row_names_map = dict(zip(atlas[label] - 1, atlas[roi_names]))

    # enforce L/R/else
    atlas = atlas.copy()
    atlas[hemisphere] = atlas[hemisphere].apply(
        lambda x: x if x in (left_symbol, right_symbol) else "else"
    )
    groups_hemi = atlas.groupby(hemisphere)

    # helper: get group‐DataFrame or empty
    def _get(side):
        return (
            groups_hemi.get_group(side)
            if side in groups_hemi.groups
            else atlas.iloc[0:0]
        )

    # build the three hemisphere‐based dictionaries
    left_df = _get(left_symbol)
    right_df = _get(right_symbol)
    else_df = _get("else")

    groups = [
        create_dictionary(left_df, grouping_name, label, roi_names),
        create_dictionary(right_df, grouping_name, label, roi_names),
        create_dictionary(else_df, grouping_name, label, roi_names),
    ]

    return (
        conn,
        groups,
        metadata_map,
        metadata,
        row_names_map,
        display_node_names,
        display_group_names,
    )


def create_dictionary(grouped_by_hemisphere, grouping_name, label, roi_names):
    """
    This function groups the ROIs according to a grouping variable within hemisphere.

    Parameters
    ----------
    grouped_by_hemisphere: DataFrame
        Part of the matrix that related to specific hemisphere.
    grouping_name: string
        The name of variable by which we will group ROIs in the graph.
        the name of grouping variaible must be
    label: string
        Name of column in the atlas that contains the numbers (labels) of the ROIs
    roi_names: string
        Name of column in the atlas that contains the ROIs names.
        These names will be presented on the circular graph

    Returns
    -------
    groups: Dictionary<string,List<(int, string)>
        Dictionary of groups of ROIs, divided by the grouping variable.
        The keys are the groups names. The values are lists of tuples, each tuple represents a ROI in the group.
        Each tuple contains the index of a ROI in the connectivity matrix (starting from zero) and the ROI name.
        for example:  {"Frontal lobe": [(0, precentral gyrus), (1, SFG), (2, MFG), (3, IFG)}

    """

    grouped_atlas = grouped_by_hemisphere.groupby([grouping_name])
    groups_names = list(grouped_atlas.groups.keys())
    groups = {}
    for group in groups_names:
        group_df = grouped_atlas.get_group(group)
        groups[group] = list(zip(group_df[label] - 1, group_df[roi_names]))
    return groups


def normalize_and_set_threshold(connectivity_matrix, threshold=0.5):
    """
    This function gets a connectivity matrix and normalize its values between 0 to 1.
    After normalization, the function zero the matrix values that are lower than  the threshold

    Parameters
    ----------
    connectivity_matrix: np.ndarray
        n X n matrix where n is the number of ROIs in  the atlas.
        Each cell in the matrix describes the connection between each pair of ROIs.

    threshold: float
        A float between 0 to 1 by. Values lower than threshold are set to zero.

    Returns
    -------
    filtered_matrix: np.ndarray
        connecitivty matrix after thresholding and normalization
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("Threshold value must be between 0-1!")

    normalized_connectivity_matrix = (
        connectivity_matrix - np.min(connectivity_matrix)
    ) / (np.max(connectivity_matrix) - np.min(connectivity_matrix))

    filtered_matrix = normalized_connectivity_matrix
    filtered_matrix[normalized_connectivity_matrix < threshold] = 0

    return filtered_matrix


def rotate_node_by_count(g, count):
    """
    This function gets a nx graph and rotate it by count.
    Parameters
    ----------
    g: networkx.Graph

    count: int
        by how many points the graph should be rotated
    """
    values = [g.nodes()[node]["sort"] for node in g.nodes]
    print(count)
    values.sort()
    values = values[-count:]
    for node in g.nodes():
        if g.nodes()[node]["sort"] in values:
            g.nodes()[node]["sort"] += -1000


def add_padding(g, padding_count, sort_value):
    """
    This function gets a nx graph and add empty padding values
    ----------
    g: networkx.Graph

    padding_count: int
        how many empty points should be added

    sort_value: int
        running index for sorting

    Returns
    -------
      sort_value: int
        running index for sorting
    """
    for i in range(padding_count):
        node_value = i * randint(1000, 100000000)
        g.add_node(node_value)
        g.nodes()[node_value]["group"] = "_"
        g.nodes()[node_value]["transparent"] = 0
        g.nodes()[node_value]["sort"] = sort_value
        sort_value += 1
    return sort_value


def add_values(g, items, sort_value):
    """
    This function add values to a graph
    ----------
    g: networkx.Graph

    items: dictionary<(int,str>
        dictionary from key to tuple of int and label

    sort_value: int
        running index for sorting

    rotate_nodes: Boolean
        should rotate the list labels

    Returns
    -------
      sort_value: int
        running index for sorting
    """
    for k1, v1 in items:
        for i1 in v1:
            g.nodes()[i1[0]]["group"] = k1
            g.nodes()[i1[0]]["transparent"] = 1
            g.nodes()[i1[0]]["sort"] = sort_value
            sort_value += 1
        sort_value = add_padding(g, 5, sort_value)
    return sort_value


class circular_graph:
    def __init__(
        self,
        filtered_matrix: np.ndarray,
        groups,
        metadata_map,
        metadata_label: str,
        row_names_map,
        display_node_names: bool,
        display_group_names: bool,
    ):
        self.filtered = filtered_matrix
        self.groups = groups
        self.metadata_map = metadata_map
        self.metadata_label = metadata_label
        self.row_names_map = row_names_map
        self.disp_nodes = display_node_names
        self.disp_groups = display_group_names

    def _compute_positions(
        self,
        small_gap_arc: float = 0.05,   # radians between groups
        large_gap_arc: float = 0.3     # radians to leave clear at top
    ):
        """
        Compute positions so that:
          - A large gap of `large_gap_arc` sits centered at 90° (π/2).
          - If there are any 'else' nodes, carve out `else_arc = small_gap_arc` at 270°,
            and shrink each hemi by half of that.
          - Within each hemi, groups get arcs proportional to their node counts,
            separated by fixed `small_gap_arc`.
        """
        left_dict, right_dict, else_dict = self.groups
        group_names = list(left_dict.keys())
        H = len(group_names)

        sg = small_gap_arc
        lg = large_gap_arc

        # 1) see if we have any bottom‐(else) nodes
        all_else = [idx for grp in else_dict for idx, _ in else_dict[grp]]
        n_else  = len(all_else)
        if n_else:
            left_counts  = [len(v) for v in left_dict.values()]
            right_counts = [len(v) for v in right_dict.values()]
            else_counts  = [len(v) for v in else_dict.values()]

            total_left   = sum(left_counts)
            total_right  = sum(right_counts)
            total_else   = sum(else_counts)

            # how many interior small gaps?
            gaps_left  = max(len(left_counts)  - 1, 0)
            gaps_right = max(len(right_counts) - 1, 0)
            gaps_else  = max(len(else_counts)  - 1, 0)

            # total nodes & total small‐gap length
            total_nodes      = total_left + total_right + total_else
            total_small_gaps = small_gap_arc * (gaps_left + gaps_right + gaps_else)

            # 1) compute per-node spacing so that:
            #    2π = large_top_gap + total_small_gaps + per_node_arc * total_nodes
            per_node_arc = (2*math.pi - large_gap_arc - total_small_gaps) / total_nodes

            # 2) turn counts into group‐arcs
            left_arcs  = [per_node_arc * c for c in left_counts]
            right_arcs = [per_node_arc * c for c in right_counts]
            else_arcs  = [per_node_arc * c for c in else_counts]

            else_arc = sum(else_arcs) + small_gap_arc * gaps_else
        else: 
            else_arc = 0.0

        # 2) carve out top gap (lg) and bottom gap (else_arc), split remaining half/half
        hemi_arc = math.pi - lg - (else_arc / 2)

        # 3) compute how much of hemi_arc each group gets
        left_counts = [len(left_dict.get(grp, [])) for grp in group_names]
        right_counts = [len(right_dict.get(grp, [])) for grp in group_names]
        total_left = sum(left_counts)  or 1
        total_right = sum(right_counts) or 1

        avail_arc   = hemi_arc - (H - 1) * sg
        left_arcs   = [avail_arc * (c / total_left)  for c in left_counts]
        right_arcs  = [avail_arc * (c / total_right) for c in right_counts]

        # 4) starting angles for each hemi
        left_start  = math.pi/2 + lg/2
        right_start = math.pi/2 - lg/2

        angles = {}

        # LEFT hemi: CCW
        theta = left_start
        for arc, grp, cnt in zip(left_arcs, group_names, left_counts):
            items = left_dict.get(grp, [])
            if cnt:
                for j, (idx, _) in enumerate(items):
                    frac = (j + 0.5) / cnt
                    angles[idx] = theta + frac * arc
            theta += arc + sg

        # RIGHT hemi: CW
        theta = right_start
        for arc, grp, cnt in zip(right_arcs, group_names, right_counts):
            items = right_dict.get(grp, [])
            if cnt:
                for j, (idx, _) in enumerate(items):
                    frac = (j + 0.5) / cnt
                    angles[idx] = theta - frac * arc
            theta -= arc + sg

        # 5) ELSE group at bottom, spanning else_arc
        if n_else:
            for j, idx in enumerate(all_else):
                frac = (j + 0.5) / n_else
                angles[idx] = 3*math.pi/2 + (frac - 0.5) * else_arc

        # 6) build your position dicts
        base_pos   = {n: (math.cos(a), math.sin(a)) for n, a in angles.items()}
        inner_pos  = base_pos.copy()
        outer_pos  = {n: (1.1*x, 1.1*y) for n,(x,y) in base_pos.items()}
        labels_pos = {n: (1.05*x,1.05*y) for n,(x,y) in base_pos.items()}

        return base_pos, inner_pos, outer_pos, labels_pos, angles
    
    def show_graph(self):
        # --- build graph & attrs (unchanged) ---
        g = nx.from_numpy_array(self.filtered).to_directed()
        nx.set_edge_attributes(
            g,
            {e: w * 3 for e, w in nx.get_edge_attributes(g, "weight").items()},
            "doubled_weight",
        )
        nx.set_node_attributes(g, self.metadata_map, "metadata")

        node_group_map = {}
        for hemi_dict in self.groups:
            for grp_label, items in hemi_dict.items():
                for idx, _ in items:
                    node_group_map[idx] = grp_label
        nx.set_node_attributes(g, node_group_map, "group")

        # --- build symmetric L/R sequence with gaps ---
        base_pos, inner_pos, outer_pos, labels_pos, angles = self._compute_positions()

        # --- prepare color data (unchanged) ---
        meta_vals = [float(g.nodes[n]["metadata"]) for n in g.nodes()]
        grp_vals = [g.nodes[n]["group"] for n in g.nodes()]
        unique_grp = list(dict.fromkeys(grp_vals))
        grp_to_int = {g: i for i, g in enumerate(unique_grp)}
        grp_nums = [grp_to_int[g] for g in grp_vals]

        # --- draw ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.axis("off")

        # metadata ring (outer)
        nc = nx.draw_networkx_nodes(
            g,
            pos=outer_pos,
            node_color=meta_vals,
            cmap=plt.get_cmap("viridis"),
            node_size=10,
            ax=ax,
        )
        fig.colorbar(
            nc, ax=ax, location="right", fraction=0.046, pad=0.04, label=self.metadata_label
        )

        # group ring (inner)
        nx.draw_networkx_nodes(
            g,
            pos=inner_pos,
            node_color=grp_nums,
            cmap=plt.get_cmap("tab20"),
            node_size=10,
            ax=ax,
        )

        # curved edges via Bézier into the center
        cmap = plt.get_cmap("plasma")
        edge_attrs = nx.get_edge_attributes(g, "weight")
        min_w, max_w = min(edge_attrs.values()), max(edge_attrs.values())
        norm = plt.Normalize(vmin=min_w, vmax=max_w)

        for u, v, attr in g.edges(data=True):
            w = attr["weight"]
            ww = attr["doubled_weight"]
            color = cmap(norm(w))

            x1, y1 = inner_pos[u]
            x2, y2 = inner_pos[v]
            # control point at the center (0,0):
            verts = [(x1, y1), (0, 0), (x2, y2)]
            # codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            # path = Path(verts, codes)
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
            path = MplPath(verts, codes)

            patch = patches.PathPatch(
                path, edgecolor=color, linewidth=ww, alpha=0.8, facecolor="none"
            )
            ax.add_patch(patch)

        # add the colorbar for egdes
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(
            sm, ax=ax, location="top", fraction=0.046, pad=0.04, label="Edge weight"
        )

        # add node labels
        if self.disp_nodes:
            nx.draw_networkx_labels(
                g, pos=labels_pos, labels=self.row_names_map, font_size=2.5, ax=ax
            )

        # --- group labels with hemisphere‐specific alignment ---
        if self.disp_groups:
            # self.groups = [left_dict, right_dict, else_dict]
            for side_idx, hemi_dict in enumerate(self.groups):
                for grp_label, items in hemi_dict.items():
                    # centroid angle
                    indices = [idx for idx, _ in items]
                    thetas  = [angles[idx] for idx in indices]
                    mean_sin = sum(math.sin(t) for t in thetas) / len(thetas)
                    mean_cos = sum(math.cos(t) for t in thetas) / len(thetas)
                    mean_theta = math.atan2(mean_sin, mean_cos)
                    # position just outside the node‐ring
                    tx, ty = 1.35 * math.cos(mean_theta), 1.35 * math.sin(mean_theta)
                    # choose horizontal alignment per hemisphere
                    if side_idx == 0:
                        ha = "left"
                    elif side_idx == 1:
                        ha = "right"
                    else:
                        ha = "center"
                    ax.text(tx, ty, grp_label, ha=ha, va="center", fontsize=8)

        plt.show()


# ---------------------------- usage ----------------------------
# Path to the current script
SCRIPT_DIR = Path(__file__).resolve().parent

# Path to the data directory (going up one level from visualization/)
ATLAS_DIR = SCRIPT_DIR.parent / "data" / "atlases" / "available_atlases"
MAT_DIR = SCRIPT_DIR.parent / "data" / "connectomes"

atlas_fname = r"fan2016/MNI152/space-MNI152_atlas-fan2016_res-1mm_dseg.csv"
matrix_fname = r"fan2016.csv"
# Now construct full paths
atlas_path = ATLAS_DIR / atlas_fname

matrix_path = MAT_DIR / matrix_fname
conn, groups, metadata_map, metadata_label, row_names_map, disp_nodes, disp_groups = load_data(
    matrix_path,
    atlas_path,
    grouping_name="Lobe",
    label="Label",
    roi_names="ROIname",
    hemisphere="Hemi",
    metadata="Yeo_7network",
    display_node_names=False,
    display_group_names=True,
)

filtered = normalize_and_set_threshold(conn, threshold=0.1)
bna = circular_graph(
    filtered, groups, metadata_map, metadata_label, row_names_map, display_node_names=disp_nodes, display_group_names=disp_groups
)
bna.show_graph()

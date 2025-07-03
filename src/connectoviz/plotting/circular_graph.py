import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CircularGraphBuilder:
    """
    Builds and plots circular connectivity graphs with optional hemispheric, group, and metadata layering.
    """
    def __init__(
        self,
        connectivity: np.ndarray,
        layout_dfs: dict,
        hemispheric_layout: bool = False,
        group: str = None,
        display_node_name: bool = False,
        display_group_name: bool = False,
        metadata_layers: list = None,
    ):
        """
        Parameters:
        - connectivity: square np.ndarray of connectivity values indexed by node_index.
        - layout_dfs: dict of DataFrames describing node parcels:
            * If hemispheric_layout=True: must include 'L' and 'R' keys, optional others.
            * If hemispheric_layout=False: must include only 'All'.
        - hemispheric_layout: whether to split into hemispheric and other parcels.
        - group: column name for grouping nodes after parcellation.
        - display_node_name: show node labels.
        - display_group_name: show group labels.
        - metadata_layers: list of column names for inner concentric metadata rings.
        """
        self.conn = connectivity
        self.hemi = hemispheric_layout
        self.group_col = group
        self.display_node_name = display_node_name
        self.display_group_name = display_group_name
        self.metadata_layers = metadata_layers or []

        # Validate and store layout dict
        if self.hemi:
            required = {'L', 'R'}
            if not required.issubset(layout_dfs.keys()):
                raise ValueError("layout_dfs must include 'L' and 'R' keys for hemispheric layout.")
        else:
            if list(layout_dfs.keys()) != ['All']:
                raise ValueError("layout_dfs must contain only 'All' key when hemispheric_layout=False.")

        self.layout_dfs = layout_dfs
        # Flatten for metadata/group
        self.layout_df = pd.concat(layout_dfs.values(), ignore_index=True)

        # Placeholders
        self.node_angles = {}
        self.parcel_sectors = None
        self.group_sectors = None
        self.layer_positions = {}

    def compute_layout(self):
        """
        Compute node angles, parcel sectors (if hemispheric), group sectors, and metadata layer radii.
        """
        if self.hemi:
            self._compute_hemispheric_layout()
        else:
            self._compute_single_layout()

        if self.group_col:
            self._apply_grouping()

        if self.metadata_layers:
            self._compute_metadata_layers()

        return {
            'node_angles': self.node_angles,
            'parcel_sectors': self.parcel_sectors,
            'group_sectors': self.group_sectors,
            'layer_positions': self.layer_positions,
        }

    def _compute_single_layout(self):
        """Compute full-circle layout from 'All'."""
        df = self.layout_dfs['All']
        n = len(df)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self.node_angles = dict(zip(df['node_index'], angles))
        self.parcel_sectors = None

    def _compute_hemispheric_layout(self):
        """Compute layout for L/R hemispheres (top) and optional others (bottom)."""
        dfs = self.layout_dfs
        n_L = len(dfs['L'])
        n_R = len(dfs['R'])
        other_keys = [k for k in dfs if k not in ('L', 'R')]
        n_other = sum(len(dfs[k]) for k in other_keys)

        hemi_gap = 0.4
        parcel_gap = 0.05

        # Top half spans
        top_half = np.pi
        half_span = (top_half - hemi_gap) / 2
        mid = np.pi / 2

        # Right sector
        end_R = mid - hemi_gap / 2
        start_R = end_R - half_span
        angles_R = np.linspace(start_R, end_R, n_R, endpoint=False)
        # Left sector
        start_L = mid + hemi_gap / 2
        end_L = start_L + half_span
        angles_L = np.linspace(start_L, end_L, n_L, endpoint=False)

        self.node_angles = {}
        for idx, ang in zip(dfs['R']['node_index'], angles_R):
            self.node_angles[idx] = ang
        for idx, ang in zip(dfs['L']['node_index'], angles_L):
            self.node_angles[idx] = ang

        sectors = {'R': (start_R, end_R), 'L': (start_L, end_L)}

        # Bottom parcels
        if other_keys:
            bottom_start = np.pi + parcel_gap / 2
            bottom_span = np.pi - parcel_gap * (len(other_keys) - 1)
            total = n_other
            current = bottom_start
            for key in other_keys:
                df = dfs[key]
                m = len(df)
                span = bottom_span * (m / total)
                start = current
                end = start + span
                angles = np.linspace(start, end, m, endpoint=False)
                for idx, ang in zip(df['node_index'], angles):
                    self.node_angles[idx] = ang
                sectors[key] = (start, end)
                current = end + parcel_gap

        self.parcel_sectors = sectors
        self.group_sectors = None

    def _apply_grouping(self):
        """Define angular sectors per group with small gaps."""
        df = self.layout_df.sort_values(self.group_col)
        groups = df[self.group_col].unique()
        total = len(df)
        gap = 0.05
        start_angle = gap
        sectors = {}
        for grp in groups:
            members = df[df[self.group_col] == grp]
            span = (2 * np.pi - gap * len(groups)) * (len(members) / total)
            sectors[grp] = (start_angle, start_angle + span)
            start_angle += span + gap
        self.group_sectors = sectors

    def _compute_metadata_layers(self):
        """Compute radial positions for metadata layers inside main circle."""
        base_radius = 1.0
        thickness = 0.1
        for i, col in enumerate(self.metadata_layers, 1):
            self.layer_positions[col] = base_radius - i * thickness

    def plot(self):
        """Render the circular graph with matplotlib."""
        layout = self.compute_layout()
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})

        # Metadata layers
        for col, radius in layout['layer_positions'].items():
            vals = self.layout_df.set_index('node_index')[col]
            angs = [self.node_angles[idx] for idx in vals.index]
            ax.scatter(angs, [radius]*len(angs), s=20)

        # Nodes
        for idx, ang in self.node_angles.items():
            ax.scatter(ang, 1.0, s=50, zorder=3)
            if self.display_node_name:
                name = self.layout_df.set_index('node_index').loc[idx, 'node_name']
                ax.text(ang, 1.05, name, rotation=ang*180/np.pi,
                        ha='center', va='bottom', fontsize=8)

        # Edges
        n = self.conn.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                w = self.conn[i, j]
                if w:
                    a1, a2 = self.node_angles[i], self.node_angles[j]
                    ax.plot([a1, a2], [1.0, 1.0], alpha=min(1, abs(w)), linewidth=abs(w)*2)

        # Hemispheric labels
        if self.hemi and not self.group_col:
            for side in ('L', 'R'):
                if side in layout['parcel_sectors']:
                    s, e = layout['parcel_sectors'][side]
                    mid = (s + e)/2
                    ax.text(mid, 1.2, side, rotation=mid*180/np.pi,
                            ha='center', va='center', fontsize=12, fontweight='bold')

        # Group labels
        if self.display_group_name and self.group_col:
            for grp, (s, e) in layout['group_sectors'].items():
                mid = (s + e)/2
                ax.text(mid, 1.2, grp, rotation=mid*180/np.pi,
                        ha='center', va='center', fontsize=10, fontweight='bold')

        ax.set_axis_off()
        plt.show()


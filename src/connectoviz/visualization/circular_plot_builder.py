# connectoviz/visualization/circular_plot_builder.py

import matplotlib.pyplot as plt
from typing import Optional, List

from connectoviz.core.connectome import Connectome
from connectoviz.visualization.circular_layout import compute_layout, draw_edges
from connectoviz.visualization.ring_tracks import draw_ring_tracks
from connectoviz.visualization.styling import get_colormap


class CircularPlotBuilder:
    def __init__(
        self,
        connectome: Connectome,
        tracks: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None,
        cmap: str = "coolwarm",
        gap: float = 2.0,
        figsize: tuple = (10, 10),
        start_angle: float = 0.0,
        **kwargs,
    ):
        """
        Orchestrates the construction of a circular connectome plot.

        Parameters
        ----------
        connectome : Connectome
            A fully initialized Connectome object.
        tracks : list of str, optional
            Metadata columns to be drawn as concentric rings.
        group_by : list of str, optional
            Metadata columns to define group boundaries/gaps.
        cmap : str, default "coolwarm"
            Colormap to use for edge/track coloring.
        gap : float, default 2.0
            Angular gap (in degrees) between groups.
        figsize : tuple, default (10, 10)
            Matplotlib figure size.
        start_angle : float, default 0.0
            Starting angle (in degrees) for the circular layout.
        kwargs : dict, optional
            Optional styling and configuration parameters:
            - hemispheric : bool
                If True, plot will be split into two hemispheres (default is False).
            - track_colormaps : dict[str, str or Colormap]
                Mapping of track names to colormaps (e.g., {"GM volume": "viridis"}).
            - edge_colormap : str or Colormap
                Colormap for edges (default is "coolwarm").
            - node_colormap: str or Colormap
                Colormap for nodes (default is "viridis").
            - highlight_edges : list[tuple[str, str]]
                Specific edges to highlight (e.g., [("A", "B"), ("C", "D")]).
        """
        self.connectome = connectome
        self.tracks = tracks or []
        self.group_by = group_by or []
        self.cmap_name = cmap
        self.gap = gap
        self.figsize = figsize
        self.start_angle = start_angle
        self.kwargs = kwargs
        self.node_order = None
        self.layout = None
        self.fig = None
        self.ax = None

    def build(self):
        """Main method to build the plot step-by-step.
        Steps of the process:
        -----

        Returns
        -------
        a figure object and an additional axes object.
        """
        self._compute_node_order()
        self._generate_layout()
        self._initialize_canvas()
        self._color_tracks()
        self._draw_tracks()
        self._color_nodes()
        self._draw_nodes()
        self._style_edges()
        self._draw_edges()
        self._finalize()
        return self.fig

    def _compute_node_order(self):
        """Sort node order based on grouping."""
        self.node_order = self.connectome.get_node_order(group_by=self.group_by)

    def _generate_layout(self):
        """Compute angular positions and layout parameters."""
        self.layout = compute_layout(
            node_order=self.node_order,
            group_by=self.group_by,
            metadata=self.connectome.node_metadata,
            gap=self.gap,
            start_angle=self.start_angle,
        )

    def _initialize_canvas(self):
        """Create matplotlib polar plot."""
        self.fig, self.ax = plt.subplots(
            figsize=self.figsize, subplot_kw={"projection": "polar"}
        )
        self.ax.set_axis_off()

    def _color_tracks(self):
        """
        This method prepares and stores a dictionary of color values, 
        but does not draw anything. The result is saved to `self.track_colors`.

        For each track:
            - If the user provided a specific colormap via the `track_colormaps` kwarg, it will be used.
            - Otherwise, a default colormap is used (from `self.cmap_name`).

        Returns
        -------
        None
            The track colors are stored in `self.track_colors`, structured as:
            {
                "track_name_1": {node_id: color, ...},
                "track_name_2": {node_id: color, ...},
                ...
        }

        Notes
        -----
        Expects the `Connectome` class to implement a method:
        get_track_colors(node_order, tracks, cmap_per_track)
        """

        user_cmaps = self.kwargs.get("track_colormaps", {})
        default_cmap = get_colormap(self.cmap_name)

        cmap_per_track = {
            track: get_colormap(user_cmaps.get(track, default_cmap))
            for track in self.tracks
        }

        self.track_colors = self.connectome.get_track_colors(
            node_order=self.node_order,
            tracks=self.tracks,
            cmap=get_colormap(self.cmap_name),
        )
    
    def _draw_tracks(self):
        """Draw concentric metadata rings (if any)."""
        if self.tracks:
            draw_ring_tracks(
                ax=self.ax,
                connectome=self.connectome,
                node_order=self.node_order,
                tracks=self.tracks,
                layout=self.layout,
                track_colors=self.track_colors
            )

    def _color_nodes(self):
        """
        Assigns colors to nodes based on `group_by` metadata.

        Determines how each node should be colored:
        - If the user specifies a `group_by` metadata field (e.g., "network" or "lobe"),
          each node is assigned a color based on its group.
        - All nodes in the same group receive the same color.
        - If no `group_by` is provided, all nodes are colored uniformly.

        The resulting color map is stored internally and used by the `_draw_nodes` method.
        This method does not perform any drawing — it only prepares and stores styling information.

        Returns
        -------
        None
            Stores color mappings in:
            - self.node_colors : dict[str, tuple]
                Maps node IDs to RGBA color tuples.
            - self.group_color_map : dict[str, tuple]
                Maps group names to color tuples (used for legend rendering).
        
        Notes
        -----
        - Assumes `self.node_order` is already defined.
        - Requires access to `self.connectome.node_metadata`.
        - Expects a valid colormap via `self.cmap_name`.
        - If `group_by` includes more than one level, only the first is used for coloring.
        - If a node lacks metadata for the grouping key, it's assigned to the "Unknown" group.
        """
        node_ids = list(self.node_order)
        group_by = self.group_by[0] if self.group_by else None
        node_meta = self.connectome.node_metadata

        # Get group values (e.g., functional network, lobe)
        group_vals = [node_meta[n].get(group_by, "Unknown") for n in node_ids]

        # Build color map
        unique_groups = sorted(set(group_vals))
        cmap = get_colormap(self.cmap_name, n_colors=len(unique_groups), kind="categorical")
        color_map = {group: cmap(i) for i, group in enumerate(unique_groups)}

        # Assign and store node colors
        self.node_colors = {n: color_map[val] for n, val in zip(node_ids, group_vals)}
        self.group_color_map = color_map
    
    def _draw_nodes(self):
        """Draw nodes at their computed positions."""
        self.connectome.draw_nodes(
            ax=self.ax,
            node_order=self.node_order,
            layout=self.layout,
            node_colors=self.node_colors,
            group_color_map=self.group_color_map,
        )

    def _style_edges(self):
        """
        Assigns colors to edges based on connectivity strength and stores them.
        Does not draw — just prepares styling info for the canvas.
        """
        self.edge_colors = self.connectome.get_edge_colors(
            node_order=self.node_order,
            cmap=get_colormap(self.cmap_name),
        )

    def _draw_edges(self):
        """Draw connectivity edges."""
        draw_edges(
            ax=self.ax,
            connectome=self.connectome,
            node_order=self.node_order,
            layout=self.layout,
            cmap=get_colormap(self.cmap_name),
        )

    def _finalize(self):
        """Add title, legend, or any final touches."""
        # Placeholder for future customizations
        pass

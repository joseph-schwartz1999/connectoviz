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
        kwargs : dict
            Reserved for future extensions or styling overrides.
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
        """Main method to build the plot step-by-step."""
        self._compute_node_order()
        self._generate_layout()
        self._initialize_canvas()
        self._draw_tracks()
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

    def _draw_tracks(self):
        """Draw concentric metadata rings (if any)."""
        if self.tracks:
            draw_ring_tracks(
                ax=self.ax,
                connectome=self.connectome,
                node_order=self.node_order,
                tracks=self.tracks,
                layout=self.layout,
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

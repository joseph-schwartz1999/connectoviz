"""Main module."""

#initialize the package
from . import io, utils, viz, config, exceptions
from .io import parsers, readers, writers
from .visualization import plot, plotly, networkx, graphviz
from .utils import decorators, helpers, constants
from .config import settings, logger
from .core.connectome import Connectome
from .plotting.circular_plots import plot_circular_connectome

def plot_connectome_circular(matrix, atlas, metadata=None, mask=None, mapping=None,
                              node_vec=None, label_vec=None, **kwargs):
    """
    Create and plot a circular connectome from matrix and metadata inputs.

    Parameters
    ----------
    matrix : np.ndarray or str
        Connectivity matrix or path to matrix file.
    atlas : pd.DataFrame
        Atlas DataFrame containing region information.
    metadata : pd.DataFrame, optional
        Additional metadata for nodes (e.g., groupings, colors).
    mask : np.ndarray, optional
        Binary mask to apply to the matrix.
    mapping : dict, optional
        Dictionary mapping node index to label.
    node_vec : list of str, optional
        Alternative way to pass mapping nodes.
    label_vec : list of str, optional
        Alternative way to pass mapping labels.
    **kwargs : dict
        Additional keyword arguments passed to the circular plot function.

    Returns
    -------
    matplotlib.figure.Figure
        Circular plot figure.
    """
    connectome = Connectome(
        matrix=matrix,
        atlas=atlas,
        metadata=metadata,
        mask=mask,
        mapping=mapping,
        node_vec=node_vec,
        label_vec=label_vec
    )
    return plot_circular_connectome(connectome, **kwargs)






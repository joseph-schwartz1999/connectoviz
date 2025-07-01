# connectoviz/core/connectome.py

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional
from connectoviz.io.parsers import parse_matrix, parse_metadata
from connectoviz.utils.validation import validate_connectome_inputs


class Connectome:
    """
    Internal data structure to store parsed connectome info.

    Parameters
    ----------
    con_mat : array-like
        Connectivity matrix (square 2D structure).
    node_metadata : dict or DataFrame
        Metadata table (e.g. lobe, hemisphere, etc.).
    atlas : str
        Atlas name (e.g. 'Schaefer100').
    """

    def __init__(
        self,
        con_mat: Union[np.ndarray, pd.DataFrame, List[List[float]], str],
        node_metadata: Union[pd.DataFrame, Dict[str, List], str],
        atlas: Optional[str] = None,
    ):
        self.con_mat: np.ndarray = parse_matrix(con_mat)
        self.node_metadata: pd.DataFrame = parse_metadata(node_metadata)
        self.atlas = atlas

        validate_connectome_inputs(self.con_mat, self.node_metadata)

    def to_numpy(self):
        return self.con_mat.copy()

    def to_dataframe(self):
        return pd.DataFrame(
            self.con_mat,
            index=self.node_metadata.index,
            columns=self.node_metadata.index,
        )

    def summary(self):
        return (
            f"Atlas: {self.atlas or 'unspecified'}\n"
            f"Nodes: {self.con_mat.shape[0]}\n"
            f"Metadata: {list(self.node_metadata.columns)}"
        )

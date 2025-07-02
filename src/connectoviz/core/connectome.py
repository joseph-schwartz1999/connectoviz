# connectoviz/core/connectome.py

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Any
from connectoviz.io.parsers import (
    parse_matrix,
    masking,
    check_metadata,
    check_mapping,
    compare_mapping,
    atlas_check,
)

# next line commented out so pre commit wont kill me
# from connectoviz.utils.validation import validate_connectome_inputs


# 1.07.25- ideas for improvements:
# - add a method to convert the connectome to a networkx graph
# - use @property @setter and so on to encapsulate diffrent funcions and attributes as properties


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
        atlas: Optional[str] = None,
        node_metadata: Union[pd.DataFrame, Dict[str, List], str] = None,
        mapping: Optional[
            Union[Dict[str, str], pd.DataFrame]
        ] = None,  # verse 1 for mapping
        node_vec: Optional[Union[List[str], np.ndarray]] = None,
        label_vec: Optional[Union[List[str], np.ndarray]] = None,  # verse 2 for mapping
        # when no mapping is provided, we can use index_col and label_col to specify the columns in the metadata
        index_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ):
        self.con_mat: np.ndarray = parse_matrix(con_mat)
        self.atlas: pd.DataFrame = self._assess_atlas(atlas)
        self.node_metadata: pd.DataFrame = self._process_metadata(node_metadata)
        self.mapping, self.index_col, self.label_col = self._validate_maps(
            mapping, node_vec, label_vec, index_col, label_col
        )
        # set the merged metadata

    def _assess_atlas(self, atlas: Optional[Any]) -> pd.DataFrame:
        # uncomment when they approve my work
        # bool_check, _, _ = atlas_check(atlas, None, None)
        # for know kust do a block check that is true
        bool_check = True  # remove this line when they approve my work
        # If atlas is a boolean, it means it was validated successfully
        if bool_check:
            return atlas
        else:
            # raise error
            raise ValueError(
                "Invalid atlas provided. Please provide a valid atlas DataFrame or name."
            )

        # If atlas is a DataFrame, we assume it's already validated

    def _process_metadata(self, metadata):
        # Your logic for checking and standardizing metadata
        if metadata is None:
            return None
        # Assuming `atlas` is already loaded and validated

        return check_metadata(metadata, atlas=self.atlas, mapping=self.mapping)

    def _validate_maps(
        self,
        mapping: Optional[Union[Dict[str, str], pd.DataFrame]],
        node_vec: Optional[Union[List[str], np.ndarray]] = None,
        label_vec: Optional[Union[List[str], np.ndarray]] = None,
        index_col: Optional[str] = None,
        label_col: Optional[str] = None,
    ):
        if mapping is not None:
            # Validate mapping structure
            validated_mapping = check_mapping(mapping, node_vec, label_vec)

            # Check mapping values against atlas
            _, label_col_r = compare_mapping(validated_mapping, self.atlas)
            # validate them and the atlas on the way:
            # uncomment when they approve my work
            # _, validated_index_col, validated_label_col = atlas_check(
            #     self.atlas, index_col, label_col_r
            # )
            # for now present the previous form
            validated_index_col, validated_label_col = atlas_check(
                self.atlas, index_col, label_col_r
            )  #

            return mapping, validated_index_col, validated_label_col

    # add a function from_input to create a Connectome from inputs
    # functions assumes con_mat is a square 2D structure, atlas is a DataFrame, and node_metadata is a DataFrame
    # and mapping is an optional dict or DataFrame
    @classmethod
    def from_inputs(
        con_mat: Any,
        atlas: pd.DataFrame,
        node_metadata: Optional[pd.DataFrame],
        mapping: Optional[Union[Dict[str, str], pd.DataFrame]] = None,
    ):
        """
        Create a Connectome instance from input parameters.

        Parameters
        ----------
        con_mat : array-like
            Connectivity matrix (square 2D structure).
        atlas : pd.DataFrame
            Atlas reference table.
        node_metadata : pd.DataFrame
            Node-level metadata (same length as con_mat).
        mapping : dict or DataFrame, optional
            Optional remapping of node indices to labels.

        Returns
        -------
        Connectome
            A Connectome instance.
        """

        # check mapping

        # Validate and process the atlas

        return Connectome(
            con_mat=con_mat,
            atlas=atlas,
            node_metadata=node_metadata,
            mapping=mapping,
        )

    def reorder_by_hemisphere(self):
        """
        Reorder the connectivity matrix by hemisphere.

        This method assumes that the node metadata contains a 'hemisphere' column.
        It will reorder the connectivity matrix such that all left hemisphere nodes
        come before right hemisphere nodes.
        """
        # to continue i need that both my changes will be accepted to the main repo

    def to_numpy(self):
        return self.con_mat.copy()

    def to_dataframe(self):
        # Convert the connectivity matrix to a DataFrame with node metadata as index and columns.
        if self.node_metadata is None or self.node_metadata.empty:
            raise ValueError("Node metadata is not specified or empty.")
        return pd.DataFrame(
            self.con_mat,
            index=self.node_metadata.index,
            columns=self.node_metadata.index,
        )

    def apply_mask(self, mask: Union[np.ndarray, List[bool]]):
        """
        Apply a binary mask to the connectivity matrix.

        Parameters
        ----------
        mask : array-like
            Binary mask to apply to the connectivity matrix.
        """
        self.con_mat = masking(self.con_mat, mask)

    def summary(self):
        return (
            f"Atlas: {self.atlas or 'unspecified'}\n"
            f"Nodes: {self.con_mat.shape[0]}\n"
            f"Metadata: {list(self.node_metadata.columns or "none specified")}\n"
            f"\nMapping: {self.mapping or 'none specified'}\n"
            f"Index Column: {self.index_col or 'none specified'}\n"
            f"Label Column: {self.label_col or 'none specified'}"
        )

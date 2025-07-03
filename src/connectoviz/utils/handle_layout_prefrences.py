# handling layout prefrences input
from typing import Dict, Any, Union
import pandas as pd
from connectoviz.io.parsers import check_layout_list_Multilayer


def handle_hemisphere(
    combined_metadata: pd.DataFrame, layout_dict: Dict[str, Any]
) -> Union[
    pd.DataFrame,
    tuple[pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
    """
    Handle the hemisphere preferences and return the combined metadata DataFrame with the hemisphere column.

    Parameters
    ----------
    combined_metadata : pd.DataFrame
        The combined metadata DataFrame.
    layout_dict : Dict[str, Any]
        The layout preferences dictionary.

    Returns
    -------
    pd.DataFrame
        The combined metadata DataFrame with the hemi column added.
    """
    if "node_index" not in combined_metadata.columns:
        raise ValueError(
            "The combined metadata DataFrame must contain a 'node_index' column."
        )

    # check if there is 'hemispheric' or hemi column
    if "hemispheric" in combined_metadata.columns:
        # set as hemi coloumn
        combined_metadata["hemi"] = combined_metadata["hemispheric"]
    elif "hemi" in combined_metadata.columns:
        # set as hemi coloumn
        combined_metadata["hemi"] = combined_metadata["hemi"]
    else:
        combined_metadata["hemi"] = combined_metadata["node_name"].apply(
            lambda x: (
                "left"
                if x.startswith("L_") or x.endswith("_L")
                else "right" if x.startswith("R_") or x.endswith("_R") else "other"
            )
        )
        # fill empty values with 'other'
        combined_metadata["hemi"].fillna("other", inplace=True)
        # remove columns that arent node_index, node_name, hemi, and group_name

        # divide to 3 matrices bsaed on hemi column
        right_metadata = combined_metadata[combined_metadata["hemi"] == "right"]
        left_metadata = combined_metadata[combined_metadata["hemi"] == "left"]
        other_metadata = combined_metadata[combined_metadata["hemi"] == "other"]

        if not layout_dict["other"]:
            # return just the right and left metadata
            return (right_metadata, left_metadata)
        elif layout_dict["other"]:
            # return all three metadata DataFrames as a tuple
            return (right_metadata, left_metadata, other_metadata)

    return combined_metadata


def handle_layout(
    combined_metadata: pd.DataFrame, layout_dict: Dict[str, Any]
) -> tuple[Dict[str, pd.DataFrame], bool, bool, bool]:
    """
    Handle the layout preferences and return the combined metadata DataFrame with the other features in a dict.

    Parameters
    ----------
    combined_metadata : pd.DataFrame
        The combined metadata DataFrame.
    layout_dict : Dict[str, Any]
        The layout preferences dictionary.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing the combined metadata DataFrame with the other features.
    display_node_name : bool
        A boolean indicating whether to display the node names.
    display_group_name : bool
        A boolean indicating whether to display the group names.
    hemi : bool
        A boolean indicating whether to handle the hemisphere preferences.
    """
    # check if grouping is in the layout_dict
    grouping = layout_dict.get("grouping", None)
    hemi = layout_dict.get("hemi", True)
    other = layout_dict.get("other", False)

    display_group_name = layout_dict.get("display_group_name", False)

    display_node_name = layout_dict.get("display_node_name", False)
    if hemi:
        combined_metadatas = handle_hemisphere(combined_metadata, layout_dict)
        # sort each metadata DataFrame by the grouping column if it exists
        if isinstance(combined_metadatas, tuple):
            combined_metadatas = list(combined_metadatas)  # make mutable
            for i, metadata in enumerate(combined_metadatas):
                if grouping is not None and grouping in metadata.columns:
                    combined_metadatas[i] = metadata.sort_values(
                        by=grouping
                    )  # change in all coloumns the grouping column name to 'group_name'
                    combined_metadatas[i].rename(
                        columns={grouping: "group_name"}, inplace=True
                    )
            # unpack the tuple into separate variables
            right_metadata, left_metadata = combined_metadatas[:2]

            if other:
                other_metadata = combined_metadatas[2]
                # return all three metadata DataFrames as a dict and the display boolians
                return (
                    {"L": right_metadata, "R": left_metadata, "Other": other_metadata},
                    display_node_name,
                    display_group_name,
                    hemi,
                )

            return (
                {"L": right_metadata, "R": left_metadata},
                display_node_name,
                display_group_name,
                hemi,
            )

    if grouping is not None:
        if not hemi:
            # sort the DataFrame by the grouping column
            combined_metadata = combined_metadata.sort_values(by=grouping)
            # change the grouping column name to 'group_name'
            combined_metadata.rename(columns={grouping: "group_name"}, inplace=True)

            return (
                {"All": combined_metadata},
                display_node_name,
                display_group_name,
                hemi,
            )
    else:
        # if grouping is None, just return the combined metadata DataFrame
        # keep columns node_index, node_name, and hemi
        combined_metadata = combined_metadata[["node_index", "node_name", "hemi"]]
        return {"All": combined_metadata}, display_node_name, display_group_name, hemi
    # just for Mypy to shut up
    return {"All": combined_metadata}, display_node_name, display_group_name, hemi


def handle_layers(comb_meta: pd.DataFrame, layers_list: list[str,]) -> pd.DataFrame:
    """
    Handle the layers preferences and return the combined metadata DataFrame with the layers column.
    Parameters
    ----------
    comb_meta : pd.DataFrame

    The combined metadata DataFrame.
    layers_list : list[str]
        The list of layers to handle.

    Returns
    -------
    pd.DataFrame
        The combined metadata DataFrame filtered based on layers column added.
    """
    # use check_layout_list_Multilayer to check the validity of the layers_list
    validity_bool = check_layout_list_Multilayer(layers_list, comb_meta)
    if not validity_bool:
        raise ValueError("The layers_list is not valid. Please check the layers names.")

    # check if group_name in the columns of comb_meta
    initial_lis = []
    if "group_name" not in comb_meta.columns:
        initial_lis = ["node_index", "node_name"]
    else:
        initial_lis = ["node_index", "node_name", "group_name"]
    # now try to keep node_index, node_name, and layers_list columns
    filtered_df = comb_meta[initial_lis + layers_list]
    # check if the layers_list columns are in the filtered_df
    for layer in layers_list:
        if layer not in filtered_df.columns:
            raise ValueError(
                f"The layer '{layer}' is not in the combined metadata DataFrame."
            )
    # now return the filtered_df
    return filtered_df

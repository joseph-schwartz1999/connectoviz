# connectoviz/io/parsers.py

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.io
import h5py
from pathlib import Path
from typing import Any, Union, Dict, List,Optional
import nibabel as nib

# Function to parse a connectivity matrix from various formats
def parse_matrix(data: Any) -> np.ndarray:
    """Accept various formats and convert to NumPy matrix
     the expected formats for the connectivty matrix as input are:
      numpy.ndarray, pandas.DataFrame, scipy.sparse matrix,
      list, str (file path to .npy, .npz, .mat,
      .h5, .csv, .tsv, .nii, .nii.gz, .txt) or a
      scipy.io.loadmat dictionary containing the matrix.
      returns a NumPy array representation of the matrix
      or raises an error if the input format is unsupported.
        ."""
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.DataFrame):
        # Convert DataFrame to NumPy array
        if data.index.name is not None:
            data = data.reset_index(drop=True)
        return data.to_numpy()
    
    elif sparse.issparse(data):
        return data.toarray()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, str):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Matrix file not found: {data}")
        if data.endswith(".npy") or data.endswith(".npz"):
            return np.load(data)
        elif data.endswith(".mat"):
            mat = scipy.io.loadmat(data)
            # Filter out MATLAB metadats keys
            valid_keys = [k for k in mat.keys() if not k.startswith("__")]
            if not valid_keys:
                raise ValueError("No valid matrix key found in .mat file.")
            return mat[valid_keys[0]]

        elif data.endswith(".h5"):
            with h5py.File(data, "r") as f:
                keys = list(f.keys())
                if not keys:
                    raise ValueError("Empty HDF5 file.")
                return f[keys[0]][()]

        elif data.endswith(".csv"):
            return pd.read_csv(data, index_col=0).to_numpy()
        elif data.endswith(".tsv"):
            return pd.read_csv(data, sep="\t", index_col=0).to_numpy()
        elif data.endswith(".nii") or data.endswith(".nii.gz"):
            img = nib.load(data)
            return img.get_fdata()
        elif data.endswith(".txt"):
            return np.loadtxt(data)

    raise TypeError("Unsupported matrix input format.")


#handling all metadats formats that can be passed to the parser as input
def parse_metadats(meta: Union[pd.DataFrame, Dict[str, List], str]) -> pd.DataFrame:
    """Accept various formats and convert to Pandas DataFrame.
    The expected formats for the metadats input are:
    pandas.DataFrame, dict (with string keys and list values),
    or str (file path to .csv,.tsv ,.json or .txt).
    Returns a Pandas DataFrame representation of the metadats
    or raises an error if the input format is unsupported.
    """
    if isinstance(meta, pd.DataFrame):
        return meta
    elif isinstance(meta, dict):
        return pd.DataFrame(meta)
    elif isinstance(meta, str):
        meta = Path(meta)
        if not meta.exists():
            raise FileNotFoundError(f"metadats file not found: {meta}")
        if meta.endswith(".csv"):
            return pd.read_csv(meta, index_col=0)
        elif meta.endswith(".tsv"):
            return pd.read_csv(meta, sep="\t", index_col=0)
        elif meta.endswith(".xlsx"):
            return pd.read_excel(meta, index_col=0)
        elif meta.endswith(".json"):
            return pd.read_json(meta, orient='index')
        elif meta.endswith(".txt"):
            return pd.read_csv(meta, sep="\t", index_col=0)
    raise TypeError("Unsupported metadats input format.")

def check_namings(col_names:List,df:pd.DataFrame) -> bool:
    """Check if the column name is present in the DataFrame.
    This function is used to ensure that the column names
    in the metadats DataFrame match the expected naming conventions.
    """
    # Check if the column name is a string
    for col_name in col_names:
        if not isinstance(col_name, str):
            raise TypeError("Column name must be a string.")
        if col_name in df.columns:
            return True
        else:
            raise ValueError(f"Column '{col_name}' not found in DataFrame. Available columns: {df.columns.tolist()}")

def check_col_existence(cols_name: list) -> bool:
    if cols_name is None:
        raise ValueError("No column with 'label' found in the atlas DataFrame.")
    elif len(cols_name) > 1:
        raise ValueError(f"Multiple columns fitting found: {cols_name}. Please specify a the relevant col.")
    return True
        
def atlas_check(atlas: Union[pd.DataFrame, Dict[str, List], str],index_col:Union[str,None]=None ,
                label_col: Union[str, None] = None) -> List:
    """Check if the atlas is in the correct format and convert it to a DataFrame.
    The expected formats for the atlas input are:
    pandas.DataFrame, dict (with string keys and list values),
    or str (file path to .csv, .tsv, .json or .txt).
    Returns a Pandas DataFrame representation of the atlas
    or raises an error if the input format is unsupported.
    """
    atlas=parse_metadats(atlas)
    # Check if the atlas has a coloumn with the word 'label' in it  and the same for index_col
    if label_col is None:
        # If no label_col is provided, check if any column contains 'label'
        label_cols = [col for col in atlas.columns if 'label' in col.lower()]

        if check_col_existence(label_cols):
            label_col = label_cols[0]
    #same for index_col
    if index_col is None:
        # If no index_col is provided, check if any column contains 'index'
        index_cols = [col for col in atlas.columns if 'index' in col.lower()]
        if check_col_existence(index_cols):
            index_col = index_cols[0]   
    
    if index_col is not None and label_col is not None:
        # If naming  is provided, check if it exists in the DataFrame
        if not check_namings([label_col,index_col], atlas):
            raise ValueError(f"Specified label_col '{label_col}' not found in the atlas DataFrame.")
    # Ensure the label_col is a string
    if not isinstance(label_col, str):
        raise TypeError("label_col must be a string representing the column name.")
    return [index_col, label_col] 

def check_metadata(metadata: Union[pd.DataFrame, Dict[str, List], str],atlas:Union[pd.DataFrame, Dict[str, List], str] 
                   ,index_col: Union[str, None] = None, label_col:Union[str, None] = None) -> pd.DataFrame:
    """Check if the metadata is in the correct format and convert it to a DataFrame.
    The expected formats for the metadata input are:
    pandas.DataFrame, dict (with string keys and list values),
    or str (file path to .csv, .tsv, .json or .txt).
    Returns a Pandas DataFrame representation of the metadata
    or raises an error if the input format is unsupported.
    """
    #metadata = parse_metadats(metadata)
    # If index_col is provided, check if it exists in the DataFrame
    #check atlas for index_col and label_col
    index_col, label_col = atlas_check(atlas, index_col=index_col, label_col=label_col)
    #check if label_col exists in the metadata DataFrame
    if label_col not in metadata.columns:
        raise ValueError(f"Specified label_col '{label_col}' not found in the metadata DataFrame. Available columns: {metadata.columns.tolist()}")
    #check if index_col exists in the metadata DataFrame
    if index_col is not None and index_col not in metadata.columns:
        #raise warning as index_col is optional and the naming may not fit the atlas naming
        print(f"Warning: Specified index_col '{index_col}' not found in the metadata DataFrame. naming may not fit the atlas naming.")

    # ensurin label_col values match the atlas label_col values
    if not metadata[label_col].isin(atlas[label_col]).all():
        raise ValueError(f"Metadata '{label_col}' values do not match atlas '{label_col}' values. "
                         f"Metadata values: {metadata[label_col].unique()}, "
                         f"Atlas values: {atlas[label_col].unique()}")
    

#func to merge all relevant metadats into a single DataFrame
def merge_metadats(*metadats: pd.DataFrame) -> pd.DataFrame:
    """Merge multiple metadats DataFrames into a single DataFrame.
    This function accepts multiple DataFrames(expects  1-2 usuallly) as input and merges them
    This function concatenates the provided DataFrames along the columns,
    ensuring that they have the same number of rows and a common index.

    """
    if not metadats:
        raise ValueError("At least one metadats DataFrame must be provided.")
    if not all(isinstance(m, pd.DataFrame) for m in metadats):
        #try to parse metadats if not already a DataFrame
        metadats = [parse_metadats(m) if not isinstance(m, pd.DataFrame) else m for m in metadats]
    #assuming metadatas[0] is the atlas and the rest are metadata DataFrames   
    #check if all in same number of rows
    row_counts = [m.shape[0] for m in metadats]
    if len(set(row_counts)) != 1:
        raise ValueError("All metadats DataFrames must have the same number of rows.")
    #check if all have the same index
    indices = [m.index for m in metadats]
    if not all(idx.equals(indices[0]) for idx in indices):
        raise ValueError("All metadats DataFrames must have the same index.")


    #Concatenate DataFrames along columns, aligning by index
    merged = pd.concat(metadats, axis=1, join='outer')
    if merged.empty:
        raise ValueError("Merged DataFrame is empty. Check input metadats.")
    #if names of coloumns arent unique,raise error with name of the first column that is not unique
    if not merged.columns.is_unique:
        non_unique_cols = merged.columns[merged.columns.duplicated()].unique()
        raise ValueError(f"Non-unique column names found: {non_unique_cols.tolist()}")

    return merged




#fuction that checks if mapping exists and in the right format
def check_mapping(
    mapping: Optional[Union[pd.DataFrame, Dict[str, str]]] = None,
    node_vec: Optional[List[str]] = None,
    label_vec: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Processes mapping information either from a DataFrame/dict or from two separate lists.

    Parameters:
    - mapping: A DataFrame with two columns or a dictionary mapping node to label.
    - node_vec: A list of node names (used only if mapping is None).
    - label_vec: A list of labels corresponding to node_vec.

    Returns:
    - A dictionary mapping nodes to labels.

    Raises:
    - ValueError: if inputs are invalid or inconsistent.
    """
    if mapping is None:
        if node_vec is None or label_vec is None:
            raise ValueError("If mapping is None, both node_vec and label_vec must be provided.")
        #check if both are in same size
        if len(node_vec) != len(label_vec):
            raise ValueError("node_vec and label_vec must have the same length.")
        #convert all values in label_vec to string
        label_vec = [str(label) for label in label_vec]
        #create a dictionary from node_vec and label_vec
        return dict(zip(node_vec, label_vec))

    if isinstance(mapping, pd.DataFrame):
        if mapping.shape[1] != 2:
            raise ValueError("DataFrame mapping must have exactly two columns.")
        return dict(zip(mapping.iloc[:, 0], mapping.iloc[:, 1]))

    if isinstance(mapping, dict):
        return mapping

    raise TypeError("Mapping must be a DataFrame, dictionary, or None.")
# connectoviz/io/parsers.py

import numpy as np
import pandas as pd
from scipy import sparse
import scipy.io
import h5py
from pathlib import Path
from typing import Any, Union, Dict, List


def parse_matrix(data: Any) -> np.ndarray:
    """Accept various formats and convert to NumPy matrix."""
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.DataFrame):
        return data.values
    elif sparse.issparse(data):
        return data.toarray()
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, str):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Matrix file not found: {data}")
        if data.endswith(".npy"):
            return np.load(data)
        elif data.endswith(".mat"):
            mat = scipy.io.loadmat(data)
            key = next((k for k in mat if not k.startswith("__")), None)
            return mat[key]
        elif data.endswith(".h5"):
            with h5py.File(data, "r") as f:
                return f[next(iter(f.keys()))][()]
    raise TypeError("Unsupported matrix input format.")


def parse_metadata(meta: Union[pd.DataFrame, Dict[str, List], str]) -> pd.DataFrame:
    if isinstance(meta, pd.DataFrame):
        return meta
    elif isinstance(meta, dict):
        return pd.DataFrame(meta)
    elif isinstance(meta, str):
        if meta.endswith(".csv"):
            return pd.read_csv(meta, index_col=0)
    raise TypeError("Unsupported metadata input format.")

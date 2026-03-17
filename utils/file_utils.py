import h5py
import numpy as np
import os
import pickle
import yaml
import shutil

from typing import Any, List, Union


def resolve_existing_path(path: str,
                          root_dir: str | None = None,
                          include_raw_fallback: bool = True,
                          label: str = 'Path') -> str:
    """Resolve an input path and fail with attempted candidates if missing."""
    if path is None:
        raise ValueError(f'{label} cannot be None')

    candidates = []
    if os.path.isabs(path):
        candidates.append(path)
    else:
        if root_dir is None:
            raise ValueError(f'root_dir is required to resolve relative {label.lower()}: {path}')
        direct_path = os.path.join(root_dir, path)
        candidates.append(direct_path)
        if include_raw_fallback:
            raw_path = os.path.join(root_dir, 'raw', path)
            if raw_path not in candidates:
                candidates.append(raw_path)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    tried = '\n'.join(f'  - {candidate}' for candidate in candidates)
    raise FileNotFoundError(f'{label} not found: {path}\nTried:\n{tried}')

def read_yaml_file(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_to_yaml_file(filepath: str, data: dict):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, 'w') as file:
        yaml.dump(data, file)

def read_shp_file_as_numpy(filepath: str, columns: str | list) -> np.ndarray:
    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'geopandas is required to read shapefiles; install it or avoid .shp inputs.'
        ) from exc
    file = gpd.read_file(filepath)
    np_data = file[columns].to_numpy()
    return np_data

def read_hdf_file_as_numpy(filepath: str, property_path: str, separator: str = '.') -> np.ndarray:
    with h5py.File(filepath, 'r') as hec:
        data = get_property_from_path(hec, property_path, separator)
        np_data = np.array(data)
    return np_data

def get_property_from_path(dict: dict, dict_path: str, separator: str = '.') -> Any:
    keys = dict_path.split(sep=separator)
    d = dict
    for key in keys:
        if key in d:
            d = d[key]
        else:
            raise KeyError(f'Key {key} not found in dictionary for path {dict_path}')
    return d

def read_pickle_file(filepath: str) -> Any:
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def save_to_pickle_file(filepath: str, data: Any):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

def create_temp_dirs(paths: Union[List[str], str], folder_name: str = '_temp_dir') -> List[str]:
    if isinstance(paths, str):
        paths = [paths]

    temp_dir_paths = []
    for path in paths:
        temp_dir_path = os.path.join(path, folder_name)
        if not os.path.exists(temp_dir_path):
            os.makedirs(temp_dir_path)
        temp_dir_paths.append(temp_dir_path)
    return temp_dir_paths

def delete_temp_dirs(paths: Union[List[str], str]):
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)

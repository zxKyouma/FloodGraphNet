from typing import Literal

from .csv_flood_dataset import CsvAutoregressiveFloodDataset, CsvFloodDataset

_HAS_HECRAS_DEPS = True
try:
    from .autoregressive_flood_dataset import AutoregressiveFloodDataset
    from .flood_event_dataset import FloodEventDataset
    from .in_memory_autoregressive_flood_dataset import InMemoryAutoregressiveFloodDataset
    from .in_memory_flood_dataset import InMemoryFloodDataset
except ModuleNotFoundError:
    _HAS_HECRAS_DEPS = False

    class FloodEventDataset:  # type: ignore
        pass

    class AutoregressiveFloodDataset(FloodEventDataset):  # type: ignore
        pass

    class InMemoryFloodDataset(FloodEventDataset):  # type: ignore
        pass

    class InMemoryAutoregressiveFloodDataset(AutoregressiveFloodDataset):  # type: ignore
        pass

def dataset_factory(storage_mode: Literal['memory', 'disk'],
                    autoregressive: bool,
                    dataset_type: Literal['hecras', 'csv'] = 'hecras',
                    *args,
                    **kwargs) -> FloodEventDataset:
    if dataset_type == 'csv':
        if autoregressive:
            return CsvAutoregressiveFloodDataset(*args, **kwargs)
        return CsvFloodDataset(*args, **kwargs)

    if not _HAS_HECRAS_DEPS:
        raise ModuleNotFoundError(
            'HEC-RAS dataset dependencies are missing; install rasterio/geopandas to use dataset_type="hecras".'
        )

    if autoregressive:
        if storage_mode == 'memory':
            return InMemoryAutoregressiveFloodDataset(*args, **kwargs)
        elif storage_mode == 'disk':
            return AutoregressiveFloodDataset(*args, **kwargs)

    if storage_mode == 'memory':
        return InMemoryFloodDataset(*args, **kwargs)
    elif storage_mode == 'disk':
        return FloodEventDataset(*args, **kwargs)

    raise ValueError(f'Dataset class is not defined.')

__all__ = [
    'AutoregressiveFloodDataset',
    'CsvAutoregressiveFloodDataset',
    'CsvFloodDataset',
    'FloodEventDataset',
    'InMemoryAutoregressiveFloodDataset',
    'InMemoryFloodDataset',
    'dataset_factory',
]

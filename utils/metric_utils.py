import torch
import numpy as np

try:
    from data.hecras_data_retrieval import get_wl_vol_interp_points_for_cell, get_cell_area
except ModuleNotFoundError:
    get_wl_vol_interp_points_for_cell = None
    get_cell_area = None
from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss

STANDARDIZED_RMSE_STD_DEVS = {
    1: {1: 16.877747, 2: 14.378797},
    2: {1: 3.191784, 2: 2.727131},
}

def RMSE(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse_loss(pred, target))

def MAE(pred: Tensor, target: Tensor) -> Tensor:
    return l1_loss(pred, target)

def NSE(pred: Tensor, target: Tensor) -> Tensor:
    '''Nash Sutcliffe Efficiency'''
    model_sse = torch.sum((target - pred)**2)
    mean_model_sse = torch.sum((target - target.mean())**2)
    return 1 - (model_sse / mean_model_sse)

def safe_nse(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    denominator = np.sum((target - np.mean(target)) ** 2)
    if denominator < eps:
        return np.nan
    return 1 - np.sum((target - pred) ** 2) / denominator

def safe_nodewise_nse_from_series(pred_series: np.ndarray,
                                  target_series: np.ndarray,
                                  eps: float = 1e-6) -> float:
    pred_series = np.asarray(pred_series)
    target_series = np.asarray(target_series)
    if pred_series.shape != target_series.shape:
        raise ValueError('pred_series and target_series must have the same shape.')
    if pred_series.ndim != 2:
        raise ValueError('pred_series and target_series must be 2D [timesteps, nodes].')
    pred_series = pred_series.astype(np.float64)
    target_series = target_series.astype(np.float64)
    num_timesteps = target_series.shape[0]
    if num_timesteps <= 1:
        return np.nan
    sum_target = target_series.sum(axis=0)
    sum_target_sq = (target_series ** 2).sum(axis=0)
    sum_err = ((target_series - pred_series) ** 2).sum(axis=0)
    denom = sum_target_sq - (sum_target ** 2) / float(num_timesteps)
    nse = np.full_like(denom, np.nan, dtype=np.float64)
    valid = denom >= eps
    nse[valid] = 1.0 - (sum_err[valid] / denom[valid])
    nse = nse[np.isfinite(nse)]
    return float(np.mean(nse)) if nse.size > 0 else np.nan

def kaggle_nse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denominator

def standardized_rmse(y_true: np.ndarray, y_pred: np.ndarray, std_dev: float) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if std_dev == 0 or np.isnan(std_dev):
        return np.nan
    rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(rmse_val / std_dev)

def kaggle_event_std_rmse_from_arrays(node_types: np.ndarray,
                                      node_ids: np.ndarray,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      std_devs: dict) -> float:
    node_types = np.asarray(node_types)
    node_ids = np.asarray(node_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    se = (y_true - y_pred) ** 2

    def _per_type(type_value: int) -> float:
        type_mask = node_types == type_value
        if not np.any(type_mask):
            return np.nan
        ids = node_ids[type_mask]
        se_vals = se[type_mask]
        order = np.argsort(ids)
        ids_sorted = ids[order]
        se_sorted = se_vals[order]
        unique_ids, idx_start, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
        sum_se = np.add.reduceat(se_sorted, idx_start)
        mean_mse = sum_se / counts
        rmse = np.sqrt(mean_mse)
        std = std_devs.get(type_value, np.nan)
        if std == 0 or np.isnan(std):
            return np.nan
        std_rmse = rmse / std
        std_rmse = std_rmse[np.isfinite(std_rmse)]
        return float(np.mean(std_rmse)) if std_rmse.size > 0 else np.nan

    std_rmse_1d = _per_type(1)
    std_rmse_2d = _per_type(2)
    valid = [x for x in (std_rmse_1d, std_rmse_2d) if not np.isnan(x)]
    return float(np.mean(valid)) if valid else np.nan

def kaggle_event_std_rmse_from_series_exact(node_types: np.ndarray,
                                            node_ids: np.ndarray,
                                            y_true_series: np.ndarray,
                                            y_pred_series: np.ndarray,
                                            std_devs: dict) -> float:
    y_true_series = np.asarray(y_true_series, dtype=np.float64)
    y_pred_series = np.asarray(y_pred_series, dtype=np.float64)
    if y_true_series.shape != y_pred_series.shape:
        raise ValueError('y_true_series and y_pred_series must have the same shape.')
    if y_true_series.ndim != 2:
        raise ValueError('Expected y_true_series/y_pred_series to be 2D [timesteps, nodes].')

    num_timesteps, num_nodes = y_true_series.shape
    node_types = np.asarray(node_types)
    node_ids = np.asarray(node_ids)
    if len(node_types) != num_nodes or len(node_ids) != num_nodes:
        raise ValueError('node_types and node_ids must match the number of nodes.')

    # Vectorized per-column MSE over time.
    mse_mean_per_col = np.mean((y_true_series - y_pred_series) ** 2, axis=0)

    def _per_type(type_value: int) -> float:
        idx = np.where(node_types == type_value)[0]
        if idx.size == 0:
            return np.nan
        ids = node_ids[idx]
        mse_cols = mse_mean_per_col[idx]
        order = np.argsort(ids)
        ids_sorted = ids[order]
        mse_sorted = mse_cols[order]
        unique_ids, idx_start, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
        sum_mse = np.add.reduceat(mse_sorted, idx_start)
        mean_mse = sum_mse / counts
        rmse = np.sqrt(mean_mse)
        std = std_devs.get(type_value, np.nan)
        if std == 0 or np.isnan(std):
            return np.nan
        std_rmse = rmse / std
        std_rmse = std_rmse[np.isfinite(std_rmse)]
        return float(np.mean(std_rmse)) if std_rmse.size > 0 else np.nan

    std_rmse_1d = _per_type(1)
    std_rmse_2d = _per_type(2)
    valid = [x for x in [std_rmse_1d, std_rmse_2d] if not np.isnan(x)]
    return float(np.mean(valid)) if valid else np.nan

def infer_model_id_from_root(root_dir: str) -> int:
    if not root_dir:
        return None
    root = root_dir.lower()
    if "model_1" in root or "model1" in root:
        return 1
    if "model_2" in root or "model2" in root:
        return 2
    return None

def kaggle_event_nse_from_arrays(node_types: np.ndarray,
                                 node_ids: np.ndarray,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 eps: float = 1e-6) -> float:
    node_types = np.asarray(node_types)
    node_ids = np.asarray(node_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = y_true.astype(np.float64, copy=False)
    y_pred = y_pred.astype(np.float64, copy=False)
    err = (y_true - y_pred) ** 2

    def _per_type(type_value: int) -> float:
        type_mask = node_types == type_value
        if not np.any(type_mask):
            return np.nan
        ids = node_ids[type_mask]
        y_t = y_true[type_mask]
        err_t = err[type_mask]
        order = np.argsort(ids)
        ids_sorted = ids[order]
        y_sorted = y_t[order]
        err_sorted = err_t[order]
        unique_ids, idx_start, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
        sum_y = np.add.reduceat(y_sorted, idx_start)
        sum_y2 = np.add.reduceat(y_sorted ** 2, idx_start)
        sum_err = np.add.reduceat(err_sorted, idx_start)
        denom = sum_y2 - (sum_y ** 2) / counts
        nse = np.full_like(denom, np.nan, dtype=np.float64)
        valid = denom >= eps
        nse[valid] = 1.0 - (sum_err[valid] / denom[valid])
        nse = nse[np.isfinite(nse)]
        return float(np.mean(nse)) if nse.size > 0 else np.nan

    nse_1d = _per_type(1)
    nse_2d = _per_type(2)
    valid = [x for x in (nse_1d, nse_2d) if not np.isnan(x)]
    return float(np.mean(valid)) if valid else np.nan

def kaggle_event_nse_from_series(node_types: np.ndarray,
                                 node_ids: np.ndarray,
                                 y_true_series: np.ndarray,
                                 y_pred_series: np.ndarray,
                                 eps: float = 1e-6) -> float:
    y_true_series = np.asarray(y_true_series, dtype=np.float64)
    y_pred_series = np.asarray(y_pred_series, dtype=np.float64)
    if y_true_series.shape != y_pred_series.shape:
        raise ValueError('y_true_series and y_pred_series must have the same shape.')
    if y_true_series.ndim != 2:
        raise ValueError('Expected y_true_series/y_pred_series to be 2D [timesteps, nodes].')

    num_timesteps, num_nodes = y_true_series.shape
    node_types = np.asarray(node_types)
    node_ids = np.asarray(node_ids)
    if len(node_types) != num_nodes or len(node_ids) != num_nodes:
        raise ValueError('node_types and node_ids must match the number of nodes.')

    def _per_type(type_value: int) -> float:
        type_mask = node_types == type_value
        if not type_mask.any():
            return np.nan
        nses = []
        for node_id in np.unique(node_ids[type_mask]):
            cols = np.where(type_mask & (node_ids == node_id))[0]
            if cols.size == 0:
                continue
            yt = y_true_series[:, cols].mean(axis=1)
            yp = y_pred_series[:, cols].mean(axis=1)
            if yt.size > 1:
                node_nse = kaggle_nse(yt, yp, eps=eps)
                if not np.isnan(node_nse):
                    nses.append(node_nse)
        return float(np.mean(nses)) if nses else np.nan

    nse_1d = _per_type(1)
    nse_2d = _per_type(2)
    valid = [x for x in (nse_1d, nse_2d) if not np.isnan(x)]
    return float(np.mean(valid)) if valid else np.nan

def kaggle_event_nse_from_series_exact(node_types: np.ndarray,
                                       node_ids: np.ndarray,
                                       y_true_series: np.ndarray,
                                       y_pred_series: np.ndarray,
                                       eps: float = 1e-6) -> float:
    y_true_series = np.asarray(y_true_series, dtype=np.float64)
    y_pred_series = np.asarray(y_pred_series, dtype=np.float64)
    if y_true_series.shape != y_pred_series.shape:
        raise ValueError('y_true_series and y_pred_series must have the same shape.')
    if y_true_series.ndim != 2:
        raise ValueError('Expected y_true_series/y_pred_series to be 2D [timesteps, nodes].')

    num_timesteps, num_nodes = y_true_series.shape
    node_types = np.asarray(node_types)
    node_ids = np.asarray(node_ids)
    if len(node_types) != num_nodes or len(node_ids) != num_nodes:
        raise ValueError('node_types and node_ids must match the number of nodes.')

    node_types_rep = np.tile(node_types, num_timesteps)
    node_ids_rep = np.tile(node_ids, num_timesteps)
    y_true = y_true_series.reshape(-1)
    y_pred = y_pred_series.reshape(-1)
    return kaggle_event_nse_from_arrays(node_types_rep, node_ids_rep, y_true, y_pred, eps=eps)

def CSI(binary_pred: Tensor, binary_target: Tensor):
    TP = (binary_pred & binary_target).sum() #true positive
    # TN = (~binary_pred & ~binary_target).sum() #true negative
    FP = (binary_pred & ~binary_target).sum() #false positive
    FN = (~binary_pred & binary_target).sum() #false negative

    return TP / (TP + FN + FP)

def interpolate_wl_from_vol(water_volume: np.ndarray, hec_ras_file_path: str, num_nodes: int = None):
    if get_wl_vol_interp_points_for_cell is None or get_cell_area is None:
        raise ModuleNotFoundError(
            "HEC-RAS interpolation helpers are unavailable in this repo snapshot."
        )

    if num_nodes is None:
        num_nodes = water_volume.shape[1]

    interp_values_cache = {}
    area = get_cell_area(hec_ras_file_path)

    num_timesteps = water_volume.shape[0]
    water_level = np.zeros_like(water_volume)
    for t in range(num_timesteps):
        for cell_idx in range(num_nodes):
            if cell_idx not in interp_values_cache:
                interp_values_cache[cell_idx] = get_wl_vol_interp_points_for_cell(cell_idx, hec_ras_file_path)
            water_level_interp, volume_interp = interp_values_cache[cell_idx]

            max_vol_interp = volume_interp.max()
            curr_vol = water_volume[t, cell_idx]
            if curr_vol <= max_vol_interp:
                # Interpolation within the range, assume water_level_interp and volume_interp are sorted in ascending order
                interpolated_wl = np.interp(curr_vol, volume_interp, water_level_interp)
            else:
                # Extrapolation beyond the maximum known elevation using linear approximation
                max_wl = water_level_interp[-1]
                delta_vol = curr_vol - max_vol_interp
                interpolated_wl = max_wl + (delta_vol / area[cell_idx])
            water_level[t, cell_idx] = interpolated_wl

        if t % 100 == 0:
            print(f'Completed interpolation for timestep {t}/{num_timesteps}')

    return water_level

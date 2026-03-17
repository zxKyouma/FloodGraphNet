import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch_geometric.loader import DataLoader

from data import dataset_factory
from models import model_factory
from utils import file_utils, physics_utils
from utils import postprocess_utils


@dataclass
class ModelBundle:
    model: torch.nn.Module
    dataset: object
    event_id_to_idx: Dict[int, int]
    num_nodes_1d: int
    num_nodes_2d: int
    start_node_target_idx: int
    end_node_target_idx: int
    start_edge_target_idx: int
    end_edge_target_idx: int
    device: str
    sigma_by_col: Optional[np.ndarray]
    sigma_by_col_torch: Optional[torch.Tensor]
    sigma_freeze: float
    sigma_soft: float
    sigma_k: float
    sigma_hard_freeze: Optional[float]
    hard_freeze_warmup_sigma: Optional[float]
    hard_freeze_round: Optional[float]
    warm_start_timesteps: int
    stability_anchor: str
    clip_pred_min: Optional[float]
    clip_pred_location: str
    blend_baseline: bool
    blend_baseline_k: int
    blend_c: float
    blend_c_1d: Optional[float]
    blend_c_2d: Optional[float]
    blend_anchor: str
    blend_use_delta_std: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Kaggle submission for DualFloodGNN.')
    parser.add_argument('--sample', required=True, help='Path to sample_submission.parquet')
    parser.add_argument('--output', required=True, help='Path to output submission parquet')
    parser.add_argument('--model', default='DUALFloodGNN', help='Model name in config (default: DUALFloodGNN)')
    parser.add_argument('--model1-config', required=True, help='Config for model_id=1')
    parser.add_argument('--model1-checkpoint', required=True, help='Checkpoint for model_id=1')
    parser.add_argument('--model2-config', required=True, help='Config for model_id=2')
    parser.add_argument('--model2-checkpoint', required=True, help='Checkpoint for model_id=2')
    parser.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'), help='cpu or cuda')
    parser.add_argument('--batch-size', type=int, default=200000, help='Rows per parquet batch')
    parser.add_argument('--float-format', default='%.6f', help='String format for water_level values')
    parser.add_argument('--max-batches', type=int, default=None, help='Stop after this many parquet batches')
    parser.add_argument('--sentinel-model-id', type=int, default=None, help='Model ID for sentinel override')
    parser.add_argument('--sentinel-event-id', type=int, default=None, help='Event ID for sentinel override')
    parser.add_argument('--sentinel-node-type', type=int, default=None, help='Node type for sentinel override')
    parser.add_argument('--sentinel-node-id', type=int, default=None, help='Node ID for sentinel override')
    parser.add_argument('--sentinel-offset', type=float, default=100000.0, help='Base value for sentinel override')
    parser.add_argument('--sigma-freeze', type=float, default=0.0, help='Freeze threshold in real units')
    parser.add_argument('--sigma-soft', type=float, default=0.0, help='Soft clamp threshold in real units')
    parser.add_argument('--sigma-k', type=float, default=5.0, help='Clamp multiplier for sigma-soft nodes')
    parser.add_argument('--sigma-hard-freeze', type=float, default=0.0, help='Hard freeze threshold in real units')
    parser.add_argument('--hard-freeze-warmup-sigma', type=float, default=0.02, help='Warmup std threshold for hard freeze')
    parser.add_argument('--hard-freeze-round', type=float, default=None, help='Optional rounding step for hard freeze nodes')
    parser.add_argument('--stability-anchor', choices=('last', 'mean'), default='last', help='Anchor for stability clamp')
    parser.add_argument(
        '--stability-sigma-mode',
        choices=('col', 'group_mean', 'group_rms', 'group_max'),
        default='group_max',
        help='Sigma map mode for stability (default: group_max)',
    )
    parser.add_argument(
        '--clip-pred-min',
        type=float,
        default=0.0,
        help='Minimum value for prediction clipping in real units (default: 0).',
    )
    parser.add_argument(
        '--no-clip',
        action='store_true',
        help='Disable prediction clipping (overrides --clip-pred-min).',
    )
    parser.add_argument(
        '--clip-location',
        choices=('pre_stability', 'post_stability', 'post_bias', 'post_final'),
        default='post_final',
        help='Where to apply prediction clipping (default: post_final).',
    )
    parser.add_argument('--blend-baseline', action='store_true', help='Enable variance-aware baseline blending.')
    parser.add_argument('--blend-k', type=int, default=5, help='Warmup window length for baseline blending.')
    parser.add_argument('--blend-c', type=float, default=0.02, help='Blend gate constant for alpha = std/(std+c).')
    parser.add_argument('--blend-c-1d', type=float, default=None, help='Optional blend gate constant override for 1D nodes.')
    parser.add_argument('--blend-c-2d', type=float, default=None, help='Optional blend gate constant override for 2D nodes.')
    parser.add_argument(
        '--blend-anchor',
        choices=('warmup_mean', 'warmup_last', 'prev'),
        default='warmup_mean',
        help='Baseline anchor for blending.',
    )
    parser.add_argument(
        '--blend-use-delta-std',
        action='store_true',
        default=True,
        help='Use warmup delta std for blend gating (default: on).',
    )
    parser.add_argument(
        '--blend-use-level-std',
        action='store_false',
        dest='blend_use_delta_std',
        help='Use warmup level std for blend gating.',
    )
    parser.add_argument(
        '--sequence-window-stride',
        type=int,
        default=None,
        help='Stride (timesteps) between sequence model windows. Enables sliding-window inference when set.',
    )
    return parser.parse_args()


def _should_clip_pred(clip_pred_min: Optional[float], clip_location: str, location: str) -> bool:
    if clip_pred_min is None:
        return False
    if clip_location == "pre_stability":
        return location in ("pre_stability", "post_bias")
    return clip_location == location


def _maybe_clip_pred(
    pred: torch.Tensor,
    clip_pred_min: Optional[float],
    clip_location: str,
    location: str,
) -> torch.Tensor:
    if not _should_clip_pred(clip_pred_min, clip_location, location):
        return pred
    return pred.clamp(min=clip_pred_min)


def _alpha_from_std(std_val: float, c_val: float) -> float:
    if c_val is None or c_val <= 0:
        return 1.0
    denom = std_val + c_val
    if denom <= 0:
        return 1.0
    return float(std_val / denom)


def _model_needs_rain(model: torch.nn.Module) -> bool:
    return (
        bool(getattr(model, "rain_residual_enabled", False))
        or bool(getattr(model, "type1_event_residual_enabled", False))
        or bool(getattr(model, "coeff_moe_enabled", False))
        or bool(getattr(model, "coeff_operator_enabled", False))
    )


def _unpack_dual_outputs(outputs):
    if isinstance(outputs, tuple):
        pred_diff = outputs[0] if len(outputs) >= 1 else None
        edge_pred_diff = outputs[1] if len(outputs) >= 2 else None
        return pred_diff, edge_pred_diff
    return outputs, None


def _compute_blend_params_for_window(
    dataset,
    warmup_window: torch.Tensor,
    blend_baseline_k: int,
    blend_c: float,
    blend_c_1d: Optional[float],
    blend_c_2d: Optional[float],
    blend_anchor: str,
    blend_use_delta_std: bool,
 ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if warmup_window is None or warmup_window.numel() == 0:
        return None, None
    k = min(max(int(blend_baseline_k or 1), 1), warmup_window.shape[1])
    if k <= 0:
        return None, None
    warmup_tail = warmup_window[:, -k:]
    if blend_anchor == "warmup_last":
        baseline = warmup_tail[:, [-1]]
    else:
        baseline = warmup_tail.mean(dim=1, keepdim=True)

    if blend_use_delta_std and warmup_tail.shape[1] > 1:
        deltas = warmup_tail[:, 1:] - warmup_tail[:, :-1]
        std_by_col = deltas.std(dim=1, keepdim=True, unbiased=False)
    else:
        std_by_col = warmup_tail.std(dim=1, keepdim=True, unbiased=False)

    node_types = getattr(dataset, "kaggle_node_types", None)
    node_ids = getattr(dataset, "kaggle_node_ids", None)
    if node_types is None or node_ids is None:
        alpha = torch.ones_like(std_by_col)
        if blend_c > 0:
            alpha = std_by_col / (std_by_col + blend_c)
        return baseline, alpha

    node_types = np.asarray(node_types)
    node_ids = np.asarray(node_ids)
    alpha = torch.ones_like(std_by_col)
    for tval in (1, 2):
        tmask = node_types == tval
        if not tmask.any():
            continue
        c_val = blend_c
        if tval == 1 and blend_c_1d is not None:
            c_val = float(blend_c_1d)
        if tval == 2 and blend_c_2d is not None:
            c_val = float(blend_c_2d)
        for nid in np.unique(node_ids[tmask]):
            cols = np.where(tmask & (node_ids == nid))[0]
            if cols.size == 0:
                continue
            std_val = float(std_by_col[cols].mean())
            alpha_val = _alpha_from_std(std_val, c_val)
            alpha[cols] = alpha_val

    return baseline, alpha


def _sigma_by_group_from_sigma_by_col(
    sigma_by_col: np.ndarray,
    node_types: np.ndarray,
    node_ids: np.ndarray,
    mode: str,
) -> np.ndarray:
    sigma_by_col = np.asarray(sigma_by_col, dtype=np.float64)
    node_types = np.asarray(node_types)
    node_ids = np.asarray(node_ids)
    out = np.zeros_like(sigma_by_col, dtype=np.float64)

    for tval in (1, 2):
        tmask = node_types == tval
        if not tmask.any():
            continue
        for nid in np.unique(node_ids[tmask]):
            cols = np.where(tmask & (node_ids == nid))[0]
            if cols.size == 0:
                continue
            s = sigma_by_col[cols]
            s = s[np.isfinite(s)]
            if s.size == 0:
                g = 0.0
            elif mode == "group_mean":
                g = float(np.mean(s))
            elif mode == "group_rms":
                g = float(np.sqrt(np.mean(s * s)))
            elif mode == "group_max":
                g = float(np.max(s))
            else:
                raise ValueError(f"Unknown stability sigma mode: {mode}")
            out[cols] = g

    return out


def _get_sliding_window_indices(dataset) -> Tuple[int, int, int, int]:
    previous_timesteps = dataset.previous_timesteps
    sliding_window_length = previous_timesteps + 1

    target_nodes_idx = dataset.DYNAMIC_NODE_FEATURES.index(dataset.NODE_TARGET_FEATURE)
    start_node_target_idx = dataset.num_static_node_features + (target_nodes_idx * sliding_window_length)
    end_node_target_idx = start_node_target_idx + sliding_window_length

    target_edges_idx = dataset.DYNAMIC_EDGE_FEATURES.index(dataset.EDGE_TARGET_FEATURE)
    start_edge_target_idx = dataset.num_static_edge_features + (target_edges_idx * sliding_window_length)
    end_edge_target_idx = start_edge_target_idx + sliding_window_length

    return start_node_target_idx, end_node_target_idx, start_edge_target_idx, end_edge_target_idx


def _compute_sigma_by_col(config: Dict, sigma_mode: str) -> np.ndarray:
    dataset_parameters = config['dataset_parameters']
    dataset_type = dataset_parameters.get('dataset_type', 'hecras')
    training_cfg = dataset_parameters['training']
    base_dataset_config = {
        'root_dir': dataset_parameters['root_dir'],
        'nodes_shp_file': dataset_parameters.get('nodes_shp_file'),
        'edges_shp_file': dataset_parameters.get('edges_shp_file'),
        'dem_file': dataset_parameters.get('dem_file'),
        'features_stats_file': dataset_parameters.get('features_stats_file', 'features_stats.yaml'),
        'previous_timesteps': dataset_parameters.get('previous_timesteps', 1),
        'warm_start_timesteps': dataset_parameters.get('warm_start_timesteps'),
        'normalize': dataset_parameters.get('normalize', True),
        'timestep_interval': dataset_parameters.get('timestep_interval'),
        'spin_up_time': dataset_parameters.get('spin_up_time'),
        'time_from_peak': dataset_parameters.get('time_from_peak'),
        'inflow_boundary_nodes': dataset_parameters.get('inflow_boundary_nodes', []),
        'outflow_boundary_nodes': dataset_parameters.get('outflow_boundary_nodes', []),
        'file_prefix': dataset_parameters.get('file_prefix', ''),
        'static_dir': dataset_parameters.get('static_dir'),
        'mode': 'test',
        'dataset_summary_file': training_cfg['dataset_summary_file'],
        'event_stats_file': training_cfg['event_stats_file'],
        'debug': False,
        'logger': None,
        'force_reload': True,
    }
    storage_mode = dataset_parameters['storage_mode']
    dataset = dataset_factory(storage_mode=storage_mode,
                              autoregressive=False,
                              dataset_type=dataset_type,
                              **base_dataset_config)
    sigma_by_col, _ = postprocess_utils.compute_sigma_by_col(
        dataset,
        warm_start_timesteps=dataset_parameters.get('warm_start_timesteps'),
    )
    if sigma_mode != "col":
        node_types = getattr(dataset, "kaggle_node_types", None)
        node_ids = getattr(dataset, "kaggle_node_ids", None)
        if node_types is not None and node_ids is not None:
            sigma_by_col = _sigma_by_group_from_sigma_by_col(sigma_by_col, node_types, node_ids, sigma_mode)
    return sigma_by_col


def _compute_warmup_anchor(bundle: ModelBundle,
                           sliding_window: torch.Tensor,
                           event_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = bundle.dataset
    warmup_window = sliding_window
    if dataset.is_normalized:
        warmup_window = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, warmup_window)
    warmup_window = torch.clip(warmup_window, min=0)
    if bundle.warm_start_timesteps:
        warmup_window = warmup_window[:, -bundle.warm_start_timesteps:]
    warm_start_value = warmup_window[:, [-1]]
    warmup_mean = warmup_window.mean(dim=1, keepdim=True)
    warmup_std = warmup_window.std(dim=1, keepdim=True, unbiased=False)

    if hasattr(dataset, "events_dynamic") and bundle.warm_start_timesteps:
        try:
            event = dataset.events_dynamic[event_idx]
            dynamic_nodes = event["dynamic_nodes"]
            node_idx = dataset.DYNAMIC_NODE_FEATURES.index(dataset.NODE_TARGET_FEATURE)
            warmup_series = dynamic_nodes[: bundle.warm_start_timesteps, :, node_idx]
            if dataset.is_normalized:
                warmup_series = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, warmup_series)
            warmup_series = np.clip(warmup_series, a_min=0, a_max=None)
            mean = np.nanmean(warmup_series, axis=0)
            last = warmup_series[bundle.warm_start_timesteps - 1]
            std = np.nanstd(warmup_series, axis=0)
            mean_t = torch.tensor(mean, device=bundle.device, dtype=warmup_window.dtype).unsqueeze(1)
            last_t = torch.tensor(last, device=bundle.device, dtype=warmup_window.dtype).unsqueeze(1)
            std_t = torch.tensor(std, device=bundle.device, dtype=warmup_window.dtype).unsqueeze(1)
            warmup_mean = torch.where(torch.isfinite(mean_t), mean_t, warmup_mean)
            warm_start_value = torch.where(torch.isfinite(last_t), last_t, warm_start_value)
            warmup_std = torch.where(torch.isfinite(std_t), std_t, warmup_std)
        except Exception:
            pass

    return warm_start_value, warmup_mean, warmup_std


def _load_bundle(config_path: str,
                 checkpoint_path: str,
                 model_name: str,
                 device: str,
                 sigma_freeze: float,
                 sigma_soft: float,
                 sigma_k: float,
                 sigma_hard_freeze: Optional[float],
                 hard_freeze_warmup_sigma: Optional[float],
                 hard_freeze_round: Optional[float],
                 stability_anchor: str,
                 sigma_mode: str,
                 clip_pred_min: Optional[float],
                 clip_pred_location: str,
                 blend_baseline: bool,
                 blend_baseline_k: int,
                 blend_c: float,
                 blend_c_1d: Optional[float],
                 blend_c_2d: Optional[float],
                 blend_anchor: str,
                 blend_use_delta_std: bool) -> ModelBundle:
    config = file_utils.read_yaml_file(config_path)
    dataset_parameters = config['dataset_parameters']
    dataset_type = dataset_parameters.get('dataset_type', 'hecras')

    base_dataset_config = {
        'root_dir': dataset_parameters['root_dir'],
        'nodes_shp_file': dataset_parameters.get('nodes_shp_file'),
        'edges_shp_file': dataset_parameters.get('edges_shp_file'),
        'dem_file': dataset_parameters.get('dem_file'),
        'features_stats_file': dataset_parameters.get('features_stats_file', 'features_stats.yaml'),
        'previous_timesteps': dataset_parameters.get('previous_timesteps', 1),
        'warm_start_timesteps': dataset_parameters.get('warm_start_timesteps'),
        'normalize': dataset_parameters.get('normalize', True),
        'timestep_interval': dataset_parameters.get('timestep_interval'),
        'spin_up_time': dataset_parameters.get('spin_up_time'),
        'time_from_peak': dataset_parameters.get('time_from_peak'),
        'inflow_boundary_nodes': dataset_parameters.get('inflow_boundary_nodes', []),
        'outflow_boundary_nodes': dataset_parameters.get('outflow_boundary_nodes', []),
        'file_prefix': dataset_parameters.get('file_prefix', ''),
        'static_dir': dataset_parameters.get('static_dir'),
    }

    testing_cfg = dataset_parameters['testing']
    dataset_config = {
        **base_dataset_config,
        'mode': 'test',
        'dataset_summary_file': testing_cfg['dataset_summary_file'],
        'event_stats_file': testing_cfg['event_stats_file'],
        'debug': False,
        'logger': None,
        'force_reload': True,
    }

    storage_mode = dataset_parameters['storage_mode']
    dataset = dataset_factory(storage_mode=storage_mode,
                              autoregressive=False,
                              dataset_type=dataset_type,
                              **dataset_config)

    model_params = config['model_parameters'][model_name]
    base_model_params = {
        'static_node_features': dataset.num_static_node_features,
        'dynamic_node_features': dataset.num_dynamic_node_features,
        'static_edge_features': dataset.num_static_edge_features,
        'dynamic_edge_features': dataset.num_dynamic_edge_features,
        'previous_timesteps': dataset.previous_timesteps,
        'device': device,
    }
    model_config = {**model_params, **base_model_params}
    model = model_factory(model_name, **model_config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    event_id_to_idx = {event_id: idx for idx, event_id in enumerate(dataset.hec_ras_run_ids)}
    num_nodes_1d = len(dataset.node_idx_map_1d)
    num_nodes_2d = len(dataset.node_idx_map_2d)

    start_node_target_idx, end_node_target_idx, start_edge_target_idx, end_edge_target_idx = _get_sliding_window_indices(dataset)
    warm_start_timesteps = int(dataset_parameters.get('warm_start_timesteps') or 0)
    print(f'Computing sigma map from training data for {config_path}...')
    sigma_by_col = _compute_sigma_by_col(config, sigma_mode)
    sigma_by_col_torch = (
        torch.tensor(sigma_by_col, device=device, dtype=torch.float32)
        if sigma_by_col is not None
        else None
    )

    return ModelBundle(
        model=model,
        dataset=dataset,
        event_id_to_idx=event_id_to_idx,
        num_nodes_1d=num_nodes_1d,
        num_nodes_2d=num_nodes_2d,
        start_node_target_idx=start_node_target_idx,
        end_node_target_idx=end_node_target_idx,
        start_edge_target_idx=start_edge_target_idx,
        end_edge_target_idx=end_edge_target_idx,
        device=device,
        sigma_by_col=sigma_by_col,
        sigma_by_col_torch=sigma_by_col_torch,
        sigma_freeze=sigma_freeze,
        sigma_soft=sigma_soft,
        sigma_k=sigma_k,
        sigma_hard_freeze=sigma_hard_freeze,
        hard_freeze_warmup_sigma=hard_freeze_warmup_sigma,
        hard_freeze_round=hard_freeze_round,
        warm_start_timesteps=warm_start_timesteps,
        stability_anchor=stability_anchor,
        clip_pred_min=clip_pred_min,
        clip_pred_location=clip_pred_location,
        blend_baseline=blend_baseline,
        blend_baseline_k=blend_baseline_k,
        blend_c=blend_c,
        blend_c_1d=blend_c_1d,
        blend_c_2d=blend_c_2d,
        blend_anchor=blend_anchor,
        blend_use_delta_std=blend_use_delta_std,
    )


def _predict_event_seq_sliding(bundle: ModelBundle, event_id: int, stride: int) -> np.ndarray:
    dataset = bundle.dataset
    if event_id not in bundle.event_id_to_idx:
        raise KeyError(f'Event ID {event_id} not found in dataset.')
    event_idx = bundle.event_id_to_idx[event_id]

    event_start_idx = dataset.event_start_idx[event_idx]
    event_end_idx = dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(dataset.event_start_idx) else dataset.total_rollout_timesteps
    event_dataset = dataset[event_start_idx:event_end_idx]

    init_x = dataset[event_start_idx].x
    init_edge_attr = dataset[event_start_idx].edge_attr
    if init_x.dim() == 3:
        init_x = init_x[:, :, 0]
    if init_edge_attr.dim() == 3:
        init_edge_attr = init_edge_attr[:, :, 0]

    sliding_window = init_x[:, bundle.start_node_target_idx:bundle.end_node_target_idx].clone().to(bundle.device)
    edge_sliding_window = init_edge_attr[:, bundle.start_edge_target_idx:bundle.end_edge_target_idx].clone().to(bundle.device)

    warm_start_value, warmup_mean, warmup_std = _compute_warmup_anchor(bundle, sliding_window, event_idx)
    warmup_window = sliding_window
    if dataset.is_normalized:
        warmup_window = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, warmup_window)
    warmup_window = torch.clip(warmup_window, min=0)
    if bundle.warm_start_timesteps:
        warmup_window = warmup_window[:, -bundle.warm_start_timesteps:]

    blend_baseline = None
    blend_alpha = None
    if bundle.blend_baseline:
        blend_baseline, blend_alpha = _compute_blend_params_for_window(
            dataset=dataset,
            warmup_window=warmup_window,
            blend_baseline_k=bundle.blend_baseline_k,
            blend_c=bundle.blend_c,
            blend_c_1d=bundle.blend_c_1d,
            blend_c_2d=bundle.blend_c_2d,
            blend_anchor=bundle.blend_anchor,
            blend_use_delta_std=bundle.blend_use_delta_std,
        )

    num_steps = len(event_dataset)
    num_nodes = dataset.num_nodes
    preds_sum = np.zeros((num_steps, num_nodes), dtype=np.float32)
    preds_count = np.zeros((num_steps, 1), dtype=np.float32)

    horizon = int(getattr(bundle.model, "sequence_horizon", 1))
    stride = max(1, min(int(stride), horizon))

    with torch.no_grad():
        step_idx = 0
        while step_idx < num_steps:
            graph = event_dataset[step_idx].to(bundle.device)

            x = graph.x
            edge_attr = graph.edge_attr
            if x.dim() == 3:
                x = x[:, :, 0]
            if edge_attr.dim() == 3:
                edge_attr = edge_attr[:, :, 0]
            x = torch.concat([x[:, :bundle.start_node_target_idx], sliding_window, x[:, bundle.end_node_target_idx:]], dim=1)
            edge_attr = torch.concat([edge_attr[:, :bundle.start_edge_target_idx], edge_sliding_window, edge_attr[:, bundle.end_edge_target_idx:]], dim=1)
            edge_index = graph.edge_index
            node_types = getattr(graph, "node_types", None)
            rainfall_seq = None
            if _model_needs_rain(bundle.model) and hasattr(graph, "local_mass_info"):
                rainfall_seq = physics_utils.get_rainfall(graph)
            if "rainfall_seq" in bundle.model.forward.__code__.co_varnames and rainfall_seq is not None:
                pred_seq = bundle.model(x, edge_index, edge_attr, node_types=node_types, rainfall_seq=rainfall_seq)
            else:
                pred_seq = bundle.model(x, edge_index, edge_attr, node_types=node_types)
            if isinstance(pred_seq, tuple):
                pred_seq = pred_seq[0]
            if pred_seq.dim() == 3 and pred_seq.size(1) == 1:
                pred_seq = pred_seq.squeeze(1)
            if pred_seq.dim() == 2:
                pred_seq = pred_seq.unsqueeze(-1)

            prev_node_pred = sliding_window[:, [-1]]
            pred_state_seq = prev_node_pred + torch.cumsum(pred_seq, dim=-1)

            prev_real = prev_node_pred
            if dataset.is_normalized:
                prev_real = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, prev_real)
            prev_real = torch.clip(prev_real, min=0)

            for h in range(pred_state_seq.size(-1)):
                out_idx = step_idx + h
                if out_idx >= num_steps:
                    break
                pred_real = pred_state_seq[:, [h]]
                if dataset.is_normalized:
                    pred_real = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, pred_real)
                pred_real = _maybe_clip_pred(pred_real, bundle.clip_pred_min, bundle.clip_pred_location, "pre_stability")

                if bundle.sigma_by_col_torch is not None and out_idx >= bundle.warm_start_timesteps:
                    anchor = warm_start_value
                    if bundle.stability_anchor == "mean":
                        anchor = warmup_mean
                    pred_real = postprocess_utils.apply_stability_postprocess_torch(
                        pred_real,
                        anchor,
                        bundle.sigma_by_col_torch,
                        bundle.sigma_freeze,
                        bundle.sigma_soft,
                        bundle.sigma_k,
                        hard_freeze_sigma=bundle.sigma_hard_freeze,
                        hard_freeze_warm_start=warm_start_value,
                        hard_freeze_round=bundle.hard_freeze_round,
                        hard_freeze_warmup_sigma=bundle.hard_freeze_warmup_sigma,
                        warmup_sigma_by_col=warmup_std,
                    )
                    pred_real = _maybe_clip_pred(pred_real, bundle.clip_pred_min, bundle.clip_pred_location, "post_stability")

                if blend_alpha is not None:
                    if bundle.blend_anchor == "prev":
                        blend_base = prev_real
                    else:
                        blend_base = blend_baseline
                    if blend_base is not None:
                        pred_real = blend_base + blend_alpha * (pred_real - blend_base)

                pred_real = _maybe_clip_pred(pred_real, bundle.clip_pred_min, bundle.clip_pred_location, "post_bias")
                pred_real_for_metrics = _maybe_clip_pred(
                    pred_real, bundle.clip_pred_min, bundle.clip_pred_location, "post_final"
                )

                preds_sum[out_idx, :] += pred_real_for_metrics.squeeze(1).detach().cpu().numpy()
                preds_count[out_idx, 0] += 1.0
                prev_real = pred_real

            # Advance sliding window by stride using normalized predictions.
            advance = min(stride, num_steps - step_idx)
            for a in range(advance):
                next_val = pred_state_seq[:, [a]]
                sliding_window = torch.concat((sliding_window[:, 1:], next_val), dim=1)
                edge_last = edge_sliding_window[:, [-1]]
                edge_sliding_window = torch.concat((edge_sliding_window[:, 1:], edge_last), dim=1)
            step_idx += advance

    counts = np.maximum(preds_count, 1.0)
    preds = preds_sum / counts
    return preds.T


def _predict_event(bundle: ModelBundle, event_id: int, sequence_stride: Optional[int] = None) -> np.ndarray:
    if sequence_stride is not None and getattr(bundle.model, "sequence_horizon", 1) > 1:
        return _predict_event_seq_sliding(bundle, event_id, sequence_stride)
    dataset = bundle.dataset
    if event_id not in bundle.event_id_to_idx:
        raise KeyError(f'Event ID {event_id} not found in dataset.')
    event_idx = bundle.event_id_to_idx[event_id]

    event_start_idx = dataset.event_start_idx[event_idx]
    event_end_idx = dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(dataset.event_start_idx) else dataset.total_rollout_timesteps
    event_dataset = dataset[event_start_idx:event_end_idx]

    dataloader = DataLoader(event_dataset, batch_size=1, shuffle=False)

    init_x = dataset[event_start_idx].x
    init_edge_attr = dataset[event_start_idx].edge_attr
    if init_x.dim() == 3:
        init_x = init_x[:, :, 0]
    if init_edge_attr.dim() == 3:
        init_edge_attr = init_edge_attr[:, :, 0]

    sliding_window = init_x[:, bundle.start_node_target_idx:bundle.end_node_target_idx].clone()
    edge_sliding_window = init_edge_attr[:, bundle.start_edge_target_idx:bundle.end_edge_target_idx].clone()
    sliding_window = sliding_window.to(bundle.device)
    edge_sliding_window = edge_sliding_window.to(bundle.device)
    warm_start_value, warmup_mean, warmup_std = _compute_warmup_anchor(bundle, sliding_window, event_idx)
    warmup_window = sliding_window
    if dataset.is_normalized:
        warmup_window = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, warmup_window)
    warmup_window = torch.clip(warmup_window, min=0)
    if bundle.warm_start_timesteps:
        warmup_window = warmup_window[:, -bundle.warm_start_timesteps:]
    blend_baseline = None
    blend_alpha = None
    if bundle.blend_baseline:
        blend_baseline, blend_alpha = _compute_blend_params_for_window(
            dataset=dataset,
            warmup_window=warmup_window,
            blend_baseline_k=bundle.blend_baseline_k,
            blend_c=bundle.blend_c,
            blend_c_1d=bundle.blend_c_1d,
            blend_c_2d=bundle.blend_c_2d,
            blend_anchor=bundle.blend_anchor,
            blend_use_delta_std=bundle.blend_use_delta_std,
        )

    num_steps = len(event_dataset)
    num_nodes = dataset.num_nodes
    preds = np.zeros((num_steps, num_nodes), dtype=np.float32)

    with torch.no_grad():
        for step_idx, graph in enumerate(dataloader):
            graph = graph.to(bundle.device)

            x = graph.x
            edge_attr = graph.edge_attr
            if x.dim() == 3:
                x = x[:, :, 0]
            if edge_attr.dim() == 3:
                edge_attr = edge_attr[:, :, 0]
            x = torch.concat([x[:, :bundle.start_node_target_idx], sliding_window, x[:, bundle.end_node_target_idx:]], dim=1)
            edge_attr = torch.concat([edge_attr[:, :bundle.start_edge_target_idx], edge_sliding_window, edge_attr[:, bundle.end_edge_target_idx:]], dim=1)
            edge_index = graph.edge_index

            rainfall_seq = None
            if _model_needs_rain(bundle.model) and hasattr(graph, "local_mass_info"):
                rainfall_seq = physics_utils.get_rainfall(graph)
            if "rainfall_seq" in bundle.model.forward.__code__.co_varnames and rainfall_seq is not None:
                outputs = bundle.model(x, edge_index, edge_attr, rainfall_seq=rainfall_seq)
            else:
                outputs = bundle.model(x, edge_index, edge_attr)
            pred_diff, edge_pred_diff = _unpack_dual_outputs(outputs)

            prev_node_pred = sliding_window[:, [-1]]
            pred = prev_node_pred + pred_diff

            if edge_pred_diff is None:
                edge_pred = edge_sliding_window[:, [-1]]
            else:
                prev_edge_pred = edge_sliding_window[:, [-1]]
                edge_pred = prev_edge_pred + edge_pred_diff

            sliding_window = torch.concat((sliding_window[:, 1:], pred), dim=1)
            edge_sliding_window = torch.concat((edge_sliding_window[:, 1:], edge_pred), dim=1)

            pred_real = pred
            if dataset.is_normalized:
                pred_real = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, pred_real)
            pred_real = _maybe_clip_pred(pred_real, bundle.clip_pred_min, bundle.clip_pred_location, "pre_stability")

            if bundle.sigma_by_col_torch is not None and step_idx >= bundle.warm_start_timesteps:
                anchor = warm_start_value
                if bundle.stability_anchor == "mean":
                    anchor = warmup_mean
                pred_real = postprocess_utils.apply_stability_postprocess_torch(
                    pred_real,
                    anchor,
                    bundle.sigma_by_col_torch,
                    bundle.sigma_freeze,
                    bundle.sigma_soft,
                    bundle.sigma_k,
                    hard_freeze_sigma=bundle.sigma_hard_freeze,
                    hard_freeze_warm_start=warm_start_value,
                    hard_freeze_round=bundle.hard_freeze_round,
                    hard_freeze_warmup_sigma=bundle.hard_freeze_warmup_sigma,
                    warmup_sigma_by_col=warmup_std,
                )
                pred_real = _maybe_clip_pred(pred_real, bundle.clip_pred_min, bundle.clip_pred_location, "post_stability")

            if blend_alpha is not None:
                if bundle.blend_anchor == "prev":
                    prev_real = prev_node_pred
                    if dataset.is_normalized:
                        prev_real = dataset.normalizer.denormalize(dataset.NODE_TARGET_FEATURE, prev_real)
                    blend_base = prev_real
                else:
                    blend_base = blend_baseline
                if blend_base is not None:
                    pred_real = blend_base + blend_alpha * (pred_real - blend_base)

            pred_real = _maybe_clip_pred(pred_real, bundle.clip_pred_min, bundle.clip_pred_location, "post_bias")
            pred_real_for_window = pred_real
            pred_real_for_metrics = _maybe_clip_pred(
                pred_real, bundle.clip_pred_min, bundle.clip_pred_location, "post_final"
            )

            if dataset.is_normalized:
                mean, std = dataset.normalizer.get_feature_mean_std(dataset.NODE_TARGET_FEATURE)
                pred = dataset.normalizer.normalize(pred_real_for_window, mean, std)
            else:
                pred = pred_real_for_window

            preds[step_idx, :] = pred_real_for_metrics.squeeze(1).detach().cpu().numpy()

    return preds.T


def _node_index(bundle: ModelBundle, node_type: int, node_id: int) -> int:
    if node_type == 1:
        return bundle.dataset.node_idx_map_1d[node_id]
    if node_type == 2:
        return bundle.dataset.node_idx_map_2d[node_id] + bundle.num_nodes_1d
    raise ValueError(f'Unexpected node_type {node_type}; expected 1 or 2.')


def _sanitize_values(values: np.ndarray) -> Tuple[np.ndarray, int, float]:
    nonfinite_mask = ~np.isfinite(values)
    nonfinite_count = int(nonfinite_mask.sum())
    if np.isfinite(values).any():
        max_abs = float(np.nanmax(np.abs(values)))
    else:
        max_abs = 0.0
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values.astype(np.float32, copy=False), nonfinite_count, max_abs


def _format_values(values: np.ndarray, fmt: str) -> Tuple[np.ndarray, int, float]:
    sanitized, nonfinite_count, max_abs = _sanitize_values(values)
    return np.char.mod(fmt, sanitized), nonfinite_count, max_abs


def generate_submission(sample_path: str,
                        output_path: str,
                        bundles: Dict[int, ModelBundle],
                        batch_size: int,
                        fmt: str,
                        max_batches: Optional[int] = None,
                        sequence_stride: Optional[int] = None,
                        sentinel_key: Optional[Tuple[int, int, int, int]] = None,
                        sentinel_offset: float = 100000.0) -> None:
    parquet_file = pq.ParquetFile(sample_path)
    schema = parquet_file.schema_arrow
    water_idx = schema.get_field_index('water_level')
    water_field = schema.field(water_idx)
    write_strings = pa.types.is_string(water_field.type) or pa.types.is_large_string(water_field.type)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    writer = pq.ParquetWriter(output_path, schema)

    current_key: Optional[Tuple[int, int, int, int]] = None
    current_timestep_idx = 0
    expected_steps: Optional[int] = None
    current_preds: Optional[np.ndarray] = None

    current_bundle: Optional[ModelBundle] = None
    current_model_id: Optional[int] = None
    current_event_id: Optional[int] = None
    total_rows = 0
    event_switches = 0
    nonfinite_total = 0
    nonfinite_max_abs = 0.0
    nonfinite_examples = []

    batch_idx = 0

    def _ensure_node_complete(key: Optional[Tuple[int, int, int, int]], count: int):
        if key is None or expected_steps is None:
            return
        if count != expected_steps:
            raise ValueError(
                f'Node sequence length mismatch for {key}: {count} rows vs expected {expected_steps}.'
            )

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_idx += 1
        model_ids = batch.column('model_id').to_numpy()
        event_ids = batch.column('event_id').to_numpy()
        node_types = batch.column('node_type').to_numpy()
        node_ids = batch.column('node_id').to_numpy()

        if len(model_ids) == 0:
            continue
        total_rows += len(model_ids)

        change = (
            (model_ids[1:] != model_ids[:-1]) |
            (event_ids[1:] != event_ids[:-1]) |
            (node_types[1:] != node_types[:-1]) |
            (node_ids[1:] != node_ids[:-1])
        )
        group_starts = np.concatenate(([0], np.where(change)[0] + 1))
        group_ends = np.concatenate((group_starts[1:], [len(model_ids)]))

        water_levels = np.empty(len(model_ids), dtype=object if write_strings else np.float32)

        for start, end in zip(group_starts, group_ends):
            model_id = int(model_ids[start])
            event_id = int(event_ids[start])
            node_type = int(node_types[start])
            node_id = int(node_ids[start])
            key = (model_id, event_id, node_type, node_id)

            if key != current_key:
                _ensure_node_complete(current_key, current_timestep_idx)
                current_key = key
                current_timestep_idx = 0

            if model_id != current_model_id or event_id != current_event_id:
                if current_model_id is not None:
                    event_switches += 1
                if model_id not in bundles:
                    raise KeyError(f'Missing bundle for model_id={model_id}')
                current_bundle = bundles[model_id]
                current_model_id = model_id
                current_event_id = event_id
                current_preds = _predict_event(current_bundle, event_id, sequence_stride=sequence_stride)
                expected_steps = current_preds.shape[1]

            assert current_bundle is not None and current_preds is not None
            node_idx = _node_index(current_bundle, node_type, node_id)
            seg_len = end - start
            if sentinel_key is not None and key == sentinel_key:
                segment = sentinel_offset + np.arange(current_timestep_idx, current_timestep_idx + seg_len, dtype=np.float32)
            else:
                segment = current_preds[node_idx, current_timestep_idx:current_timestep_idx + seg_len]
            if len(segment) != (end - start):
                raise ValueError(
                    f'Not enough predictions for {key}: needed {end - start}, got {len(segment)}.'
                )
            if write_strings:
                formatted, nonfinite_count, max_abs = _format_values(segment, fmt)
                water_levels[start:end] = formatted
            else:
                sanitized, nonfinite_count, max_abs = _sanitize_values(segment)
                water_levels[start:end] = sanitized
            if nonfinite_count:
                nonfinite_total += nonfinite_count
                if max_abs > nonfinite_max_abs:
                    nonfinite_max_abs = max_abs
                if len(nonfinite_examples) < 5:
                    nonfinite_examples.append((key, nonfinite_count))
            current_timestep_idx += (end - start)

        columns = [batch.column(i) for i in range(batch.num_columns)]
        columns[water_idx] = pa.array(water_levels, type=water_field.type)
        out_batch = pa.RecordBatch.from_arrays(columns, schema=schema)
        writer.write_batch(out_batch)

    _ensure_node_complete(current_key, current_timestep_idx)
    writer.close()
    if total_rows > 0:
        switch_rate = event_switches * 100000.0 / total_rows
        print(f'Event switches: {event_switches} over {total_rows} rows ({switch_rate:.2f} per 100k).')
    if nonfinite_total:
        print(f'WARNING: replaced {nonfinite_total} non-finite values; max |pred| before clamp: {nonfinite_max_abs:.6f}')
        for key, count in nonfinite_examples:
            print(f'  first non-finite example: {key} count={count}')
    else:
        print('Non-finite replacements: 0')


def main() -> None:
    args = parse_args()
    clip_pred_min = None if args.no_clip else args.clip_pred_min

    bundles = {
        1: _load_bundle(
            args.model1_config,
            args.model1_checkpoint,
            args.model,
            args.device,
            args.sigma_freeze,
            args.sigma_soft,
            args.sigma_k,
            args.sigma_hard_freeze,
            args.hard_freeze_warmup_sigma,
            args.hard_freeze_round,
            args.stability_anchor,
            args.stability_sigma_mode,
            clip_pred_min,
            args.clip_location,
            args.blend_baseline,
            args.blend_k,
            args.blend_c,
            args.blend_c_1d,
            args.blend_c_2d,
            args.blend_anchor,
            args.blend_use_delta_std,
        ),
        2: _load_bundle(
            args.model2_config,
            args.model2_checkpoint,
            args.model,
            args.device,
            args.sigma_freeze,
            args.sigma_soft,
            args.sigma_k,
            args.sigma_hard_freeze,
            args.hard_freeze_warmup_sigma,
            args.hard_freeze_round,
            args.stability_anchor,
            args.stability_sigma_mode,
            clip_pred_min,
            args.clip_location,
            args.blend_baseline,
            args.blend_k,
            args.blend_c,
            args.blend_c_1d,
            args.blend_c_2d,
            args.blend_anchor,
            args.blend_use_delta_std,
        ),
    }

    generate_submission(
        sample_path=args.sample,
        output_path=args.output,
        bundles=bundles,
        batch_size=args.batch_size,
        fmt=args.float_format,
        max_batches=args.max_batches,
        sequence_stride=args.sequence_window_stride,
        sentinel_key=(args.sentinel_model_id,
                      args.sentinel_event_id,
                      args.sentinel_node_type,
                      args.sentinel_node_id) if args.sentinel_model_id is not None else None,
        sentinel_offset=args.sentinel_offset,
    )


if __name__ == '__main__':
    main()

import os
import pandas as pd
from datetime import datetime

from constants import EDGE_MODELS, NODE_EDGE_MODELS
from typing import Tuple

from .logger import Logger
from . import file_utils
from .file_utils import create_temp_dirs
from .model_utils import get_loss_func

def split_dataset_events(root_dir: str,
                         dataset_summary_file: str,
                         percent_validation: float,
                         seed: int | None = None) -> Tuple[str, str]:
    if not (0 < percent_validation < 1):
        raise ValueError(f'Invalid percent_split: {percent_validation}. Must be between 0 and 1.')

    raw_dir_path = os.path.join(root_dir, 'raw')
    dataset_summary_path = file_utils.resolve_existing_path(
        dataset_summary_file,
        root_dir=root_dir,
        include_raw_fallback=True,
        label='Dataset summary file',
    )

    summary_df = pd.read_csv(dataset_summary_path)
    assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'

    num_val_events = max(int(len(summary_df) * percent_validation), 1)
    split_idx = len(summary_df) - num_val_events

    dataset_summary_dir = os.path.dirname(dataset_summary_path)
    dataset_summary_basename = os.path.splitext(os.path.basename(dataset_summary_file))[0]
    split_suffix = 'split' if seed is None else f'split_seed{seed}'
    split_folder = os.path.join(dataset_summary_dir, f'{dataset_summary_basename}_{split_suffix}')

    train_df_file = os.path.join(split_folder, 'train_split.csv')
    val_df_file = os.path.join(split_folder, 'val_split.csv')

    # Return only relative paths
    rel_base = raw_dir_path
    try:
        summary_abs = os.path.abspath(dataset_summary_path)
        raw_abs = os.path.abspath(raw_dir_path)
        if os.path.commonpath([summary_abs, raw_abs]) != raw_abs:
            rel_base = root_dir
    except ValueError:
        rel_base = root_dir

    if os.path.exists(train_df_file) and os.path.exists(val_df_file):
        # Reuse existing split for reproducibility across runs.
        return os.path.relpath(train_df_file, rel_base), os.path.relpath(val_df_file, rel_base)

    if not os.path.exists(split_folder):
        os.makedirs(split_folder)

    if seed is not None:
        summary_df = summary_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_rows = summary_df[:split_idx]
    train_rows.to_csv(train_df_file, index=False)

    val_rows = summary_df[split_idx:]
    val_rows.to_csv(val_df_file, index=False)

    train_df_relative = os.path.relpath(train_df_file, rel_base)
    val_df_relative = os.path.relpath(val_df_file, rel_base)

    return train_df_relative, val_df_relative

def get_trainer_config(model_name: str, config: dict, logger: Logger = None) -> dict:
    def log(msg):
        if logger:
            logger.log(msg)

    trainer_params = {}

    train_config = config['training_parameters']
    loss_func_parameters = config['loss_func_parameters']

    # Base Trainer parameters
    node_loss_func = loss_func_parameters['node_loss']
    edge_loss_func = loss_func_parameters['edge_loss']
    node_criterion = get_loss_func(node_loss_func, **loss_func_parameters.get(node_loss_func, {}))
    edge_criterion = get_loss_func(edge_loss_func, **loss_func_parameters.get(edge_loss_func, {}))
    loss_func = edge_criterion if model_name in EDGE_MODELS else node_criterion

    early_stopping_patience = train_config['early_stopping_patience']
    num_epochs = train_config['num_epochs']
    num_epochs_dyn_loss = train_config['num_epochs_dyn_loss']
    node_loss_weight = loss_func_parameters['node_loss_weight']
    kaggle_proxy_loss_config = loss_func_parameters.get('kaggle_proxy_loss', {})
    log(f'Using dynamic loss weight adjustment for the first {num_epochs_dyn_loss}/{num_epochs} epochs')
    log(f'Applying importance weight of {node_loss_weight} to node prediction loss after scaling')
    run_timestamp = train_config.get('run_timestamp') or datetime.now().strftime('%Y%m%d_%H%M%S')

    def _expand_checkpoint_path(path: str | None) -> str | None:
        if not path:
            return path
        if '{timestamp}' in path:
            return path.format(timestamp=run_timestamp)
        return path

    base_config = {
        'num_epochs': num_epochs,
        'num_epochs_dyn_loss': num_epochs_dyn_loss,
        'batch_size': train_config['batch_size'],
        'grad_accum_steps': train_config.get('gradient_accumulation_steps', 1),
        'gradient_clip_value': train_config['gradient_clip_value'],
        'pred_delta_clip_value': train_config.get('pred_delta_clip_value'),
        'loss_func': loss_func,
        'node_loss_weight': node_loss_weight,
        'early_stopping_patience': early_stopping_patience,
        'debug_loss_stats': train_config.get('debug_loss_stats', False),
        'kaggle_proxy_loss_config': kaggle_proxy_loss_config,
        'late_step_weighting_enabled': loss_func_parameters.get('late_step_weighting', {}).get('enabled', False),
        'late_step_weighting_lambda': loss_func_parameters.get('late_step_weighting', {}).get('lambda', 0.0),
        'late_step_weighting_mode': loss_func_parameters.get('late_step_weighting', {}).get('mode', 'linear'),
        'bias_penalty_weight': loss_func_parameters.get('bias_penalty', {}).get('weight', 0.0),
        'use_amp': train_config.get('use_amp', False),
        'amp_dtype': train_config.get('amp_dtype', 'fp16'),
        'dataloader_num_workers': train_config.get('dataloader_num_workers', 0),
        'dataloader_pin_memory': train_config.get('dataloader_pin_memory', False),
        'dataloader_persistent_workers': train_config.get('dataloader_persistent_workers', False),
        'dataloader_prefetch_factor': train_config.get('dataloader_prefetch_factor', 2),
        'checkpoint_save_path': _expand_checkpoint_path(train_config.get('checkpoint_save_path')),
        'checkpoint_every': train_config.get('checkpoint_every'),
        'best_checkpoint_path': _expand_checkpoint_path(train_config.get('best_checkpoint_path')),
        'node_loss_std_scale': loss_func_parameters.get('node_loss_std_scale', False),
    }
    log(f'Using training configuration: {base_config}')
    trainer_params.update(base_config)

    # Physics-informed training parameters
    if model_name not in EDGE_MODELS:
        use_global_mass_loss = loss_func_parameters['use_global_mass_loss']
        global_mass_loss_scale = loss_func_parameters['global_mass_loss_scale']
        global_mass_loss_weight = loss_func_parameters['global_mass_loss_weight']
        if use_global_mass_loss:
            log(f'Using global mass conservation loss with scale {global_mass_loss_scale} with importance weight {global_mass_loss_weight}')

        use_local_mass_loss = loss_func_parameters['use_local_mass_loss']
        local_mass_loss_scale = loss_func_parameters['local_mass_loss_scale']
        local_mass_loss_weight = loss_func_parameters['local_mass_loss_weight']
        if use_local_mass_loss:
            log(f'Using local mass conservation loss with scale {local_mass_loss_scale} with importance weight {local_mass_loss_weight}')

        trainer_params.update({
            'use_global_loss': use_global_mass_loss,
            'global_mass_loss_scale': global_mass_loss_scale,
            'global_mass_loss_weight': global_mass_loss_weight,
            'use_local_loss': use_local_mass_loss,
            'local_mass_loss_scale': local_mass_loss_scale,
            'local_mass_loss_weight': local_mass_loss_weight,
        })

    # Autoregressive training parameters
    autoregressive_train_config = train_config['autoregressive']
    autoregressive_enabled = autoregressive_train_config.get('enabled', False)
    if autoregressive_enabled:
        init_num_timesteps = autoregressive_train_config['init_num_timesteps']
        total_num_timesteps = autoregressive_train_config['total_num_timesteps']
        learning_rate_decay = autoregressive_train_config['learning_rate_decay']
        lr_scheduler_cfg = autoregressive_train_config.get('lr_scheduler', {}) or {}
        lr_scheduler_type = lr_scheduler_cfg.get('type', 'step')
        max_curriculum_epochs = autoregressive_train_config['max_curriculum_epochs']
        timestep_increment = autoregressive_train_config.get('timestep_increment', 1)
        log(f'Using autoregressive training for {init_num_timesteps}/{total_num_timesteps} timesteps and curriculum learning with timestep increment {timestep_increment}, patience {early_stopping_patience}, max {max_curriculum_epochs} epochs and learning rate decay {learning_rate_decay}')
        if str(lr_scheduler_type).lower() == 'plateau':
            log(
                "Using ReduceLROnPlateau scheduler "
                f"(factor={lr_scheduler_cfg.get('factor', 0.5)}, "
                f"patience={lr_scheduler_cfg.get('patience', 2)}, "
                f"threshold={lr_scheduler_cfg.get('threshold', 1e-3)}, "
                f"min_lr={lr_scheduler_cfg.get('min_lr', 1e-5)})"
            )
        elif str(lr_scheduler_type).lower() == 'cosine':
            log(
                "Using Cosine scheduler "
                f"(warmup_epochs={lr_scheduler_cfg.get('warmup_epochs', 0)}, "
                f"warmup_start_factor={lr_scheduler_cfg.get('warmup_start_factor', 0.1)}, "
                f"min_lr={lr_scheduler_cfg.get('min_lr', 1e-5)})"
            )

        trainer_params.update({
            'init_num_timesteps': init_num_timesteps,
            'total_num_timesteps': total_num_timesteps,
            'learning_rate_decay': learning_rate_decay,
            'lr_scheduler_type': lr_scheduler_type,
            'lr_scheduler_plateau_factor': lr_scheduler_cfg.get('factor', 0.5),
            'lr_scheduler_plateau_patience': lr_scheduler_cfg.get('patience', 2),
            'lr_scheduler_plateau_threshold': lr_scheduler_cfg.get('threshold', 1e-3),
            'lr_scheduler_plateau_min_lr': lr_scheduler_cfg.get('min_lr', 1e-5),
            'lr_scheduler_plateau_cooldown': lr_scheduler_cfg.get('cooldown', 0),
            'lr_scheduler_cosine_min_lr': lr_scheduler_cfg.get('min_lr', 1e-5),
            'lr_scheduler_warmup_epochs': lr_scheduler_cfg.get('warmup_epochs', 0),
            'lr_scheduler_warmup_start_factor': lr_scheduler_cfg.get('warmup_start_factor', 0.1),
            'max_curriculum_epochs': max_curriculum_epochs,
            'timestep_increment': timestep_increment,
            'scheduled_sampling_start': autoregressive_train_config.get('scheduled_sampling_start', 1.0),
            'scheduled_sampling_end': autoregressive_train_config.get('scheduled_sampling_end', 1.0),
            'scheduled_sampling_decay_epochs': autoregressive_train_config.get('scheduled_sampling_decay_epochs', 0),
            'history_noise_std': autoregressive_train_config.get('history_noise_std', 0.0),
            'rollout_state_loss_enabled': autoregressive_train_config.get('rollout_state_loss', {}).get('enabled', False),
            'rollout_state_loss_weight': autoregressive_train_config.get('rollout_state_loss', {}).get('weight', 0.0),
            'rollout_state_loss_horizon': autoregressive_train_config.get('rollout_state_loss', {}).get('horizon'),
            'rollout_state_loss_terminal_only': autoregressive_train_config.get('rollout_state_loss', {}).get('terminal_only', False),
            'kaggle_sequence_loss_enabled': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('enabled', False),
            'kaggle_sequence_loss_weight': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('weight', 0.0),
            'kaggle_sequence_loss_mask_horizon': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('mask_horizon'),
            'kaggle_sequence_loss_mask_schedule': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('mask_schedule', []),
            'kaggle_sequence_loss_type1_weight': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('type1_weight', 1.0),
            'kaggle_sequence_loss_type2_weight': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('type2_weight', 1.0),
            'kaggle_sequence_peak_weight_enabled': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('peak_weight', {}).get('enabled', False),
            'kaggle_sequence_peak_weight_alpha': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('peak_weight', {}).get('alpha', 1.0),
            'kaggle_sequence_peak_weight_alpha_schedule': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('peak_weight', {}).get('alpha_schedule', []),
            'kaggle_sequence_peak_weight_clip': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('peak_weight', {}).get('clip', 2.0),
            'kaggle_sequence_peak_weight_quantile': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('peak_weight', {}).get('quantile', 0.9),
            'kaggle_sequence_asym_enabled': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('asymmetric', {}).get('enabled', False),
            'kaggle_sequence_asym_type1_only': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('asymmetric', {}).get('type1_only', True),
            'kaggle_sequence_asym_event_only': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('asymmetric', {}).get('event_only', True),
            'kaggle_sequence_asym_event_quantile': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('asymmetric', {}).get('event_quantile', 0.95),
            'kaggle_sequence_asym_under_weight': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('asymmetric', {}).get('under_weight', 1.0),
            'kaggle_sequence_event_tail_enabled': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('event_tail', {}).get('enabled', False),
            'kaggle_sequence_event_tail_quantile': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('event_tail', {}).get('quantile', 0.95),
            'kaggle_sequence_event_tail_weight': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('event_tail', {}).get('weight', 0.0),
            'kaggle_sequence_event_tail_type1_only': autoregressive_train_config.get('kaggle_sequence_loss', {}).get('event_tail', {}).get('type1_only', False),
            'late_delta_penalty_enabled': autoregressive_train_config.get('late_delta_penalty', {}).get('enabled', False),
            'late_delta_penalty_weight': autoregressive_train_config.get('late_delta_penalty', {}).get('weight', 0.0),
            'late_delta_penalty_start_horizon': autoregressive_train_config.get('late_delta_penalty', {}).get('start_horizon', 224),
            'late_delta_penalty_type1_only': autoregressive_train_config.get('late_delta_penalty', {}).get('type1_only', True),
            'horizon_weight_ramp_enabled': autoregressive_train_config.get('horizon_weight_ramp', {}).get('enabled', False),
            'horizon_weight_ramp_type1_only': autoregressive_train_config.get('horizon_weight_ramp', {}).get('type1_only', True),
            'horizon_weight_ramp_start_horizon': autoregressive_train_config.get('horizon_weight_ramp', {}).get('start_horizon', 224),
            'horizon_weight_ramp_power': autoregressive_train_config.get('horizon_weight_ramp', {}).get('power', 2.0),
            'horizon_weight_ramp_alpha': autoregressive_train_config.get('horizon_weight_ramp', {}).get('alpha', 0.0),
            'event_dy_loss_enabled': autoregressive_train_config.get('event_dy_loss', {}).get('enabled', False),
            'event_dy_loss_weight': autoregressive_train_config.get('event_dy_loss', {}).get('weight', 0.0),
            'event_dy_loss_type1_only': autoregressive_train_config.get('event_dy_loss', {}).get('type1_only', True),
            'event_dy_loss_event_only': autoregressive_train_config.get('event_dy_loss', {}).get('event_only', True),
            'event_dy_loss_peak_weight': autoregressive_train_config.get('event_dy_loss', {}).get('peak_weight', 1.0),
            'event_dy_loss_peak_quantile': autoregressive_train_config.get('event_dy_loss', {}).get('peak_quantile', 0.9),
            'event_dy_loss_peak_dilate': autoregressive_train_config.get('event_dy_loss', {}).get('peak_dilate', 0),
            'inlet_aux_loss_enabled': autoregressive_train_config.get('inlet_aux_loss', {}).get('enabled', False),
            'inlet_aux_loss_weight': autoregressive_train_config.get('inlet_aux_loss', {}).get('weight', 0.0),
            'inlet_aux_loss_type1_only': autoregressive_train_config.get('inlet_aux_loss', {}).get('type1_only', True),
            'inlet_aux_loss_event_weight': autoregressive_train_config.get('inlet_aux_loss', {}).get('event_weight', 1.0),
            'inlet_aux_loss_event_quantile': autoregressive_train_config.get('inlet_aux_loss', {}).get('event_quantile', 0.95),
            'inlet_aux_loss_tail_quantile': autoregressive_train_config.get('inlet_aux_loss', {}).get('tail_quantile', 0.98),
            'inlet_aux_loss_tail_weight': autoregressive_train_config.get('inlet_aux_loss', {}).get('tail_weight', 1.0),
            'inlet_aux_loss_max_weight': autoregressive_train_config.get('inlet_aux_loss', {}).get('max_weight', 20.0),
            'type1_node_loss_weight': autoregressive_train_config.get('node_type_loss_weights', {}).get('type1', 1.0),
            'type2_node_loss_weight': autoregressive_train_config.get('node_type_loss_weights', {}).get('type2', 1.0),
            'direct_multi_horizon_enabled': autoregressive_train_config.get('direct_multi_horizon', {}).get('enabled', False),
            'direct_multi_horizon_horizon': autoregressive_train_config.get('direct_multi_horizon', {}).get('horizon'),
        })

    # Node/Edge prediction parameters
    if model_name in NODE_EDGE_MODELS or model_name == "DLinearNode":
        edge_pred_loss_scale = loss_func_parameters['edge_pred_loss_scale']
        edge_pred_loss_weight = loss_func_parameters['edge_loss_weight']
        log(f'Using edge prediction loss with scale {edge_pred_loss_scale} with importance weight {edge_pred_loss_weight}')
        log(f"Using {edge_criterion.__class__.__name__} loss for edge prediction")
        trainer_params.update({
            'edge_loss_func': edge_criterion,
            'edge_pred_loss_scale': edge_pred_loss_scale,
            'edge_loss_weight': edge_pred_loss_weight,
        })

    testing_config = config.get('testing_parameters', {})
    stability_enabled = bool(testing_config.get('stability', False))
    if stability_enabled:
        log('Validation metrics will use testing_parameters stability postprocess.')
    validation_use_tester = bool(testing_config.get('validation_use_tester', stability_enabled))
    trainer_params.update({
        'validation_use_tester': validation_use_tester,
        'validation_tester_config': {
            'rollout_start': testing_config.get('rollout_start', 0),
            'rollout_timesteps': testing_config.get('rollout_timesteps'),
            'pred_delta_clip_value': train_config.get('pred_delta_clip_value'),
            'val_pred_delta_clip_value': testing_config.get('validation_pred_delta_clip_value'),
            'stability': stability_enabled,
            'sigma_freeze': testing_config.get('sigma_freeze', 0.01),
            'sigma_soft': testing_config.get('sigma_soft', 0.10),
            'sigma_k': testing_config.get('sigma_k', 5.0),
            'sigma_hard_freeze': testing_config.get('sigma_hard_freeze'),
            'hard_freeze_warmup_sigma': testing_config.get('hard_freeze_warmup_sigma'),
            'hard_freeze_round': testing_config.get('hard_freeze_round'),
            'stability_anchor': testing_config.get('stability_anchor', 'last'),
            'warm_start_timesteps': testing_config.get('warm_start_timesteps'),
            'teacher_forcing': testing_config.get('teacher_forcing', False),
            'max_events': None if (
                testing_config.get('validation_max_events') is None
                or int(testing_config.get('validation_max_events', 0)) <= 0
            ) else int(testing_config.get('validation_max_events')),
            'full_eval_on_end': testing_config.get('validation_full_eval_on_end', True),
            'one_step_loss': testing_config.get('validation_one_step_loss', False),
            'zero_delta_baseline': testing_config.get('validation_zero_delta_baseline', False),
            'log_rollout_rmse_curve': testing_config.get('validation_log_rollout_rmse_curve', False),
            'debug_val_nan': testing_config.get('validation_debug_val_nan', False),
            'debug': testing_config.get('validation_debug_tester', False),
            'direct_multi_horizon_validation': testing_config.get('validation_direct_multi_horizon', False),
            'hybrid_sliding_window': testing_config.get('validation_hybrid_sliding_window', False),
            'hybrid_stride': testing_config.get('validation_hybrid_stride'),
            'hybrid_every': testing_config.get('validation_hybrid_every', 1),
            'diag_kaggle_breakdown': testing_config.get('validation_diag_kaggle_breakdown', False),
            'rmse_bucket_edges': testing_config.get('validation_rmse_bucket_edges'),
        },
    })
    log(f"Validation tester config: {trainer_params['validation_tester_config']}")

    return trainer_params

def divide_losses(losses: Tuple, divisor: float) -> Tuple:
    if divisor == 0:
        raise ValueError("Divisor cannot be zero")
    return tuple(loss / divisor for loss in losses)

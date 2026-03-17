import numpy as np
import os
import pandas as pd
import traceback
import torch
import gc
import random

from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from datetime import datetime
from data import dataset_factory, FloodEventDataset
from models import model_factory
from test import get_test_dataset_config, run_test
from training import trainer_factory
from typing import Dict, Optional, Tuple
from utils import Logger, file_utils, postprocess_utils, train_utils
from constants import EDGE_MODELS, NODE_EDGE_MODELS
from testing import DualAutoregressiveTester, EdgeAutoregressiveTester, NodeAutoregressiveTester

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--with_test", type=bool, default=False, help='Whether to run test after training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def load_dataset(config: Dict, args: Namespace, logger: Logger) -> Tuple[FloodEventDataset, Optional[FloodEventDataset]]:
    dataset_parameters = config['dataset_parameters']
    dataset_type = dataset_parameters.get('dataset_type', 'hecras')
    root_dir = dataset_parameters['root_dir']
    train_dataset_parameters = dataset_parameters['training']
    loss_func_parameters = config['loss_func_parameters']
    base_datset_config = {
        'root_dir': root_dir,
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
        'build_data_list': dataset_parameters.get('build_data_list', True),
        'mask_future_inputs': dataset_parameters.get('mask_future_inputs', False),
        'random_window_sampling': dataset_parameters.get('random_window_sampling', False),
        'random_window_seed': dataset_parameters.get('random_window_seed', 0),
        'node_derived_features_enabled': dataset_parameters.get('node_derived_features_enabled', False),
        'node_derived_feature_windows': dataset_parameters.get('node_derived_feature_windows', [1, 3, 5]),
        'node_derived_rain_windows': dataset_parameters.get('node_derived_rain_windows', [6, 10]),
        'node_derived_rain_max_windows': dataset_parameters.get('node_derived_rain_max_windows', [6, 12, 24]),
        'node_derived_rain_decay_alphas': dataset_parameters.get('node_derived_rain_decay_alphas', [0.9, 0.97, 0.99]),
        'node_derived_interactions_enabled': dataset_parameters.get('node_derived_interactions_enabled', False),
        'node_derived_interaction_static_features': dataset_parameters.get(
            'node_derived_interaction_static_features',
            ['area', 'flow_accumulation', 'centroid_elevation']
        ),
        'node_type_filter': dataset_parameters.get('node_type_filter'),
        'with_global_mass_loss': loss_func_parameters['use_global_mass_loss'],
        'with_local_mass_loss': loss_func_parameters['use_local_mass_loss'],
        'debug': args.debug,
        'logger': logger,
        'force_reload': True,
    }

    dataset_summary_file = train_dataset_parameters['dataset_summary_file']
    event_stats_file = train_dataset_parameters['event_stats_file']
    storage_mode = dataset_parameters['storage_mode']

    train_config = config['training_parameters']
    autoregressive_train_params = train_config['autoregressive']
    autoregressive_enabled = autoregressive_train_params.get('enabled', False)
    early_stopping_patience = train_config['early_stopping_patience']
    if early_stopping_patience is None:
        # No validation dataset needed
        dataset_config = {
            'mode': 'train',
            'dataset_summary_file': dataset_summary_file,
            'event_stats_file': event_stats_file,
            **base_datset_config,
        }
        if autoregressive_enabled:
            dataset_config.update({
                'num_label_timesteps': autoregressive_train_params['total_num_timesteps'],
                'dynamic_label_horizon': autoregressive_train_params.get('dynamic_label_horizon', False),
            })
        logger.log(f'Using dataset configuration: {dataset_config}')

        dataset = dataset_factory(storage_mode, autoregressive=autoregressive_enabled, dataset_type=dataset_type, **dataset_config)
        logger.log(f'Loaded train dataset with {len(dataset)} samples')
        return dataset, None

    percent_validation = train_config['val_split_percent']
    assert percent_validation is not None, 'Validation split percentage must be specified if early stopping is used.'

    if dataset_type == 'csv':
        summary_path = file_utils.resolve_existing_path(
            dataset_summary_file,
            root_dir=root_dir,
            include_raw_fallback=True,
            label='CSV training dataset summary file',
        )
        summary_df = pd.read_csv(summary_path)
        if len(summary_df) < 2:
            logger.log('Only one CSV event found; using the same dataset for validation.')
            dataset_config = {
                'mode': 'train',
                'dataset_summary_file': dataset_summary_file,
                'event_stats_file': event_stats_file,
                **base_datset_config,
            }
            if autoregressive_enabled:
                dataset_config.update({
                    'num_label_timesteps': autoregressive_train_params['total_num_timesteps'],
                    'dynamic_label_horizon': autoregressive_train_params.get('dynamic_label_horizon', False),
                })
            train_dataset = dataset_factory(storage_mode, autoregressive=autoregressive_enabled, dataset_type=dataset_type, **dataset_config)
            return train_dataset, train_dataset

    # Split dataset into training and validation sets for autoregressive training
    logger.log(f'Splitting dataset into training and validation sets with {percent_validation * 100}% for validation')
    split_seed = train_config.get('split_seed')
    train_summary_file, val_summary_file = train_utils.split_dataset_events(
        root_dir,
        dataset_summary_file,
        percent_validation,
        seed=split_seed,
    )

    event_stats_dir = os.path.dirname(event_stats_file)
    event_stats_basename = os.path.basename(event_stats_file)
    train_event_stats_file = os.path.join(event_stats_dir, f'train_{event_stats_basename}')
    train_dataset_config = {
        'mode': 'train',
        'dataset_summary_file': train_summary_file,
        'event_stats_file': train_event_stats_file,
        **base_datset_config,
    }
    if autoregressive_enabled:
        train_dataset_config.update({
            'num_label_timesteps': autoregressive_train_params['total_num_timesteps'],
            'dynamic_label_horizon': autoregressive_train_params.get('dynamic_label_horizon', False),
        })
    logger.log(f'Using training dataset configuration: {train_dataset_config}')
    train_dataset = dataset_factory(storage_mode, autoregressive=autoregressive_enabled, dataset_type=dataset_type, **train_dataset_config)

    val_event_stats_file = os.path.join(event_stats_dir, f'val_{event_stats_basename}')
    val_dataset_config = {
        'mode': 'test',
        'dataset_summary_file': val_summary_file,
        'event_stats_file': val_event_stats_file,
        **base_datset_config,
    }
    val_autoregressive = False
    testing_cfg = config.get('testing_parameters', {})
    if autoregressive_enabled and testing_cfg.get('validation_direct_multi_horizon', False):
        val_autoregressive = True
        val_dataset_config.update({
            'num_label_timesteps': autoregressive_train_params['total_num_timesteps'],
            'dynamic_label_horizon': autoregressive_train_params.get('dynamic_label_horizon', False),
        })
    logger.log(f'Using validation dataset configuration: {val_dataset_config}')
    val_dataset = dataset_factory(storage_mode, autoregressive=val_autoregressive, dataset_type=dataset_type, **val_dataset_config)

    logger.log(f'Loaded train dataset with {len(train_dataset)} samples and validation dataset with {len(val_dataset)} samples')
    return train_dataset, val_dataset

def run_train(model: torch.nn.Module,
              model_name: str,
              train_dataset: FloodEventDataset,
              logger: Logger,
              config: Dict,
              val_dataset: Optional[FloodEventDataset] = None,
              stats_dir: Optional[str] = None,
              model_dir: Optional[str] = None,
              device: str = 'cpu',
              resume_checkpoint_path: Optional[str] = None) -> str:
        train_config = config['training_parameters']

        # Loss function and optimizer
        optimizer_type = str(train_config.get('optimizer_type', 'adam')).lower()
        lr = train_config['learning_rate']
        wd = train_config['adam_weight_decay']
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            logger.log(f'Using AdamW optimizer with learning rate {lr} and weight decay {wd}')
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            logger.log(f'Using Adam optimizer with learning rate {lr} and weight decay {wd}')

        base_trainer_params = train_utils.get_trainer_config(model_name, config, logger)
        trainer_params = {
            'model': model,
            'dataset': train_dataset,
            'val_dataset': val_dataset,
            'optimizer': optimizer,
            'logger': logger,
            'device': device,
            **base_trainer_params,
        }

        autoregressive_train_config = train_config['autoregressive']
        autoregressive_enabled = autoregressive_train_config.get('enabled', False)
        trainer = trainer_factory(model_name, autoregressive_enabled, **trainer_params)
        if resume_checkpoint_path:
            logger.log(f'Resuming training from checkpoint: {resume_checkpoint_path}')
            trainer.load_checkpoint(resume_checkpoint_path)
        trainer.train()

        trainer.print_stats_summary()

        # Ensure we save best validation weights when early stopping tracks them.
        if hasattr(trainer, 'early_stopping'):
            best_weights = getattr(trainer.early_stopping, 'best_weights', None)
            if best_weights is not None:
                trainer.model.load_state_dict(best_weights)
                logger.log('Loaded best validation weights before saving model.')

        # Save training stats and model
        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if stats_dir is not None:
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            saved_metrics_path = os.path.join(stats_dir, f'{model_name}_{curr_date_str}_train_stats.npz')
            trainer.save_stats(saved_metrics_path)

        model_path = f'{model_name}_{curr_date_str}.pt'
        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = os.path.join(model_dir, f'{model_name}_{curr_date_str}.pt')
            trainer.save_model(model_path)

        return model_path

def run_postprocess_eval(model: torch.nn.Module,
                         model_name: str,
                         train_dataset: FloodEventDataset,
                         val_dataset: Optional[FloodEventDataset],
                         logger: Logger,
                         config: Dict,
                         device: str) -> None:
    testing_cfg = config.get('testing_parameters', {})
    if not testing_cfg.get('postprocess_eval_on_train_end', False):
        return
    if val_dataset is None:
        logger.log('Postprocess eval skipped: no validation dataset.')
        return
    if not testing_cfg.get('stability', False):
        logger.log('Postprocess eval skipped: stability disabled in testing_parameters.')
        return
    eval_dataset = val_dataset
    dataset_parameters = config.get('dataset_parameters', {})
    dataset_type = dataset_parameters.get('dataset_type', 'hecras')
    if dataset_type == 'csv':
        train_dataset_parameters = dataset_parameters.get('training', {})
        summary_path = train_dataset_parameters.get('dataset_summary_file')
        if summary_path is not None:
            summary_path = file_utils.resolve_existing_path(
                summary_path,
                root_dir=dataset_parameters.get('root_dir', ''),
                include_raw_fallback=True,
                label='CSV postprocess dataset summary file',
            )
            try:
                summary_df = pd.read_csv(summary_path)
                if len(summary_df) < 2:
                    dataset_config = {
                        'mode': 'test',
                        'dataset_summary_file': train_dataset_parameters.get('dataset_summary_file'),
                        'event_stats_file': train_dataset_parameters.get('event_stats_file'),
                        'root_dir': dataset_parameters.get('root_dir'),
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
                        'build_data_list': dataset_parameters.get('build_data_list', True),
                        'node_type_filter': dataset_parameters.get('node_type_filter'),
                        'with_global_mass_loss': False,
                        'with_local_mass_loss': False,
                        'debug': False,
                        'logger': None,
                        'force_reload': True,
                    }
                    storage_mode = dataset_parameters.get('storage_mode', 'memory')
                    eval_dataset = dataset_factory(storage_mode, autoregressive=False, dataset_type=dataset_type, **dataset_config)
            except Exception:
                logger.log('Postprocess eval: failed to build test-mode eval dataset for CSV; using provided val_dataset.')

    sigma_by_col, _ = postprocess_utils.compute_sigma_by_col(
        train_dataset,
        warm_start_timesteps=testing_cfg.get('warm_start_timesteps'),
    )
    tester_kwargs = {
        'model': model,
        'dataset': eval_dataset,
        'include_physics_loss': False,
        'device': device,
        'rollout_start': testing_cfg.get('rollout_start', 0),
        'rollout_timesteps': testing_cfg.get('rollout_timesteps'),
        'pred_delta_clip_value': config.get('training_parameters', {}).get('pred_delta_clip_value'),
        'stability_sigma_by_col': sigma_by_col,
        'stability_sigma_freeze': testing_cfg.get('sigma_freeze', 0.01),
        'stability_sigma_soft': testing_cfg.get('sigma_soft', 0.10),
        'stability_sigma_k': testing_cfg.get('sigma_k', 5.0),
        'stability_anchor': testing_cfg.get('stability_anchor', 'last'),
        'stability_sigma_hard_freeze': testing_cfg.get('sigma_hard_freeze'),
        'stability_hard_freeze_round': testing_cfg.get('hard_freeze_round'),
        'stability_hard_freeze_warmup_sigma': testing_cfg.get('hard_freeze_warmup_sigma'),
    }
    if model_name in NODE_EDGE_MODELS:
        tester = DualAutoregressiveTester(**tester_kwargs)
    elif model_name in EDGE_MODELS:
        tester = EdgeAutoregressiveTester(**tester_kwargs)
    else:
        tester = NodeAutoregressiveTester(**tester_kwargs)

    with open(os.devnull, "w") as f, redirect_stdout(f):
        tester.test()

    logger.log('Postprocess validation metrics (testing_parameters):')
    if model_name in NODE_EDGE_MODELS:
        logger.log(f'\tNode RMSE: {tester.get_avg_node_rmse():.4e}')
        logger.log(f'\tEdge RMSE: {tester.get_avg_edge_rmse():.4e}')
        logger.log(f'\tNode NSE: {tester.get_avg_node_nse():.4e}')
        logger.log(f'\tEdge NSE: {tester.get_avg_edge_nse():.4e}')
        kaggle_vals = [stat.get_kaggle_std_rmse() for stat in tester.events_validation_stats]
        kaggle_exact_vals = [stat.get_kaggle_std_rmse_exact() for stat in tester.events_validation_stats]
        kaggle_vals = [v for v in kaggle_vals if not np.isnan(v)]
        kaggle_exact_vals = [v for v in kaggle_exact_vals if not np.isnan(v)]
        if kaggle_vals:
            logger.log(f'\tKaggle Std RMSE: {np.mean(kaggle_vals):.4e}')
        if kaggle_exact_vals:
            logger.log(f'\tKaggle Std RMSE (exact): {np.mean(kaggle_exact_vals):.4e}')
    else:
        logger.log(f'\tNode RMSE: {tester.get_avg_node_rmse():.4e}')
        logger.log(f'\tNode NSE: {tester.get_avg_node_nse():.4e}')

def main():
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    train_config = config['training_parameters']
    log_path = train_config['log_path']
    logger = Logger(log_path=log_path)

    try:
        logger.log('================================================')

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.log(f'Setting random seed to {args.seed}')

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')
        if train_config.get('enable_tf32', True) and args.device != 'cpu':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.log('TF32 enabled for CUDA matmul/cudnn.')

        # Dataset
        train_dataset, val_dataset = load_dataset(config, args, logger)

        # Model
        model_params = config['model_parameters'][args.model]
        base_model_params = {
            'static_node_features': train_dataset.num_static_node_features,
            'dynamic_node_features': train_dataset.num_dynamic_node_features,
            'static_edge_features': train_dataset.num_static_edge_features,
            'dynamic_edge_features': train_dataset.num_dynamic_edge_features,
            'previous_timesteps': train_dataset.previous_timesteps,
            'device': args.device,
        }
        model_config = {**model_params, **base_model_params}
        model = model_factory(args.model, **model_config)
        logger.log(f'Using model: {args.model}')
        logger.log(f'Using model configuration: {model_config}')
        num_train_params = model.get_model_size()
        logger.log(f'Number of trainable model parameters: {num_train_params}')

        resume_checkpoint_path = train_config.get('resume_checkpoint_path', None)
        checkpoint_path = train_config.get('checkpoint_path', None)
        if resume_checkpoint_path is None and checkpoint_path is not None:
            logger.log(f'Loading model from checkpoint: {checkpoint_path}')
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=args.device))
        elif resume_checkpoint_path is not None:
            logger.log(f'Resume checkpoint configured: {resume_checkpoint_path}')

        stats_dir = train_config['stats_dir']
        model_dir = train_config['model_dir']
        model_path = run_train(model=model,
                               model_name=args.model,
                               train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               logger=logger,
                               config=config,
                               stats_dir=stats_dir,
                               model_dir=model_dir,
                               device=args.device,
                               resume_checkpoint_path=resume_checkpoint_path)
        run_postprocess_eval(model=model,
                             model_name=args.model,
                             train_dataset=train_dataset,
                             val_dataset=val_dataset,
                             logger=logger,
                             config=config,
                             device=args.device)

        logger.log('================================================')

        if not args.with_test:
            return

        # =================== Testing ===================
        logger.log(f'Starting testing for model: {model_path}')

        dataset_parameters = config['dataset_parameters']
        base_datset_config = {
            'root_dir': dataset_parameters['root_dir'],
            'nodes_shp_file': dataset_parameters['nodes_shp_file'],
            'edges_shp_file': dataset_parameters['edges_shp_file'],
            'dem_file': dataset_parameters['dem_file'],
            'features_stats_file': dataset_parameters['features_stats_file'],
            'previous_timesteps': dataset_parameters['previous_timesteps'],
            'normalize': dataset_parameters['normalize'],
            'timestep_interval': dataset_parameters['timestep_interval'],
            'spin_up_time': dataset_parameters['spin_up_time'],
            'time_from_peak': dataset_parameters['time_from_peak'],
            'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
            'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
            'node_type_filter': dataset_parameters.get('node_type_filter'),
            'debug': args.debug,
            'logger': logger,
            'force_reload': True,
        }
        test_dataset_config = get_test_dataset_config(base_datset_config, config)
        logger.log(f'Using test dataset configuration: {test_dataset_config}')

        # Clear memory before loading test dataset
        del dataset
        gc.collect()

        storage_mode = dataset_parameters['storage_mode']
        dataset = dataset_factory(storage_mode, autoregressive=False, **test_dataset_config)
        logger.log(f'Loaded test dataset with {len(dataset)} samples')

        logger.log(f'Using model checkpoint for {args.model}: {model_path}')
        logger.log(f'Using model configuration: {model_config}')

        test_config = config['testing_parameters']
        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        output_dir = test_config['output_dir']
        run_test(model=model,
                 model_path=model_path,
                 dataset=dataset,
                 logger=logger,
                 rollout_start=rollout_start,
                 rollout_timesteps=rollout_timesteps,
                 output_dir=output_dir,
                 device=args.device)

        logger.log('================================================')

    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()

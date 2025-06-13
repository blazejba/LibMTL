import os
import sys
import json
import types
import wandb
import gc
import time 
import pathlib
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args

from metadata import admet_metrics
from evaluator import CheckpointEvaluator
from examples.tdc.pm6_utils import dataloader_factory   
from helper_functions import parse_args, build_stage_logger
from data import SparseMultitaskDataset, load_data, remove_overlap
from model import get_decoders, freeze_encoder, copy_encoder_weights
from grit import GritTransformer

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.enable_flash_sdp(True)


if __name__ == '__main__':
    params = parse_args(LibMTL_args)

    if params.save_path is not None and getattr(params, 'load_path', None) is None:
        existing_ckpt = os.path.join(params.save_path, 'pm6', 'last.pt')
        if os.path.isfile(existing_ckpt):
            print(f'Found checkpoint {existing_ckpt}. Resuming training.')
            params.load_path = existing_ckpt

    backend = getattr(params, "model_backend", "GPS").upper()
    assert backend in {"GPS", "GRIT"}, f"Unknown model_backend {backend}"
    kwargs, optim_param, scheduler_param = prepare_args(params)
    set_device(params.gpu_id); set_random_seed(params.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scheduler_param = {
        'scheduler': 'cos',
        'eta_min': params.lr * 0.01,
        'warmup_start_factor': 0.01
    }
    if backend == "GPS":
        model_param = {
            'channels': params.model_encoder_channels,
            'pe_dim': params.model_encoder_pe_dim,
            'num_layers': params.model_encoder_num_layers,
            'attn_type': 'multihead',
            'attn_kwargs': {'dropout': params.model_encoder_dropout}
        }
    else:
        model_param = {
            'rrwp_dim': params.model_encoder_pe_dim,
            'hidden_dim': params.model_encoder_channels,
            'num_layers': params.model_encoder_num_layers,
            'num_heads': params.model_encoder_num_heads,                
            'dropout': params.model_encoder_dropout,
            'attn_dropout': params.model_encoder_dropout,
            'clamp': 5.0,
            'bn_momentum': 0.1,
        }


    all_params = vars(params) | optim_param | scheduler_param 
    
    run_name, run_id = None, None

    if params.load_path is not None:
        exp_dir = pathlib.Path(params.load_path).parent.parent
        params.save_path = str(exp_dir)
        meta_file = os.path.join(params.save_path, 'wandb_run.json')
        if os.path.isfile(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            run_name = meta.get('run_name')
            run_id   = meta.get('run_id')
            if run_id:
                print(f"Resuming WandB run {run_name} ({run_id})")
    else:
        if params.save_path is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            params.save_path = os.path.join("results/with_pretraining",
                                            f'{params.weighting}_{date_str}')
            run_name = f'{params.weighting}_{date_str}'
        else:
            run_name = pathlib.Path(params.save_path).name

    os.makedirs(params.save_path, exist_ok=True)
    os.makedirs(os.path.join(params.save_path, 'pm6'), exist_ok=True)
    os.makedirs(os.path.join(params.save_path, 'tdc'), exist_ok=True)

    wandb.init(name=run_name,
               id=run_id,
               resume='must' if run_id else None,
               save_code=True,
               config={**all_params, "backend": backend},
               entity='BorgwardtLab',
               project='libmtl_tdc',
               mode='online' if params.wandb else 'disabled')

    if run_id is None and params.save_path is not None:
        with open(os.path.join(params.save_path, 'wandb_run.json'), 'w') as f:
            json.dump({'run_id': wandb.run.id, 'run_name': wandb.run.name}, f)

    wandb.define_metric('pre_step')
    wandb.define_metric('pretrain/*', step_metric='pre_step')
    wandb.define_metric('ft_step')
    wandb.define_metric('finetune/*', step_metric='ft_step')



    print("PM6 pretraining stage...")
    start_time = time.time()
    train_loader, valid_loader, model_param['node_dim'], model_param['edge_dim'], task_dict = dataloader_factory(
        train_batch_size=params.train_batch_size,
        cache_dir="data/pm6_processed/",
        pe_dim=params.model_encoder_pe_dim
    )  
    print(f"Time taken to build PM6 dataloaders: {time.time() - start_time:.2f} seconds")

    def encoder_class():
        return GritTransformer(**model_param)

    scheduler_param['warmup_steps'] = len(train_loader)
    scheduler_param['T_max'] = len(train_loader) * params.epochs
    decoders: nn.ModuleDict = get_decoders(task_dict=task_dict,
                                           in_dim=params.model_encoder_channels, 
                                           hidden_dim=None,
                                           num_layers=0, 
                                           dropout=None)

    trainer = Trainer(task_dict=task_dict,
                      weighting=params.weighting,
                      architecture='HPS', 
                      encoder_class=encoder_class, 
                      decoders=decoders, 
                      rep_grad=params.rep_grad, 
                      multi_input=False, 
                      optim_param=optim_param, 
                      scheduler_param=scheduler_param, 
                      save_path=os.path.join(params.save_path, 'pm6'),
                      load_path=params.load_path,
                      time_limit=params.time_limit,
                      **kwargs)
    
    trainer.meter.log_wandb = types.MethodType(build_stage_logger('pretrain'), trainer.meter)
    trainer.train(train_loader, valid_loader, epochs=params.epochs)
    pretrained_model = trainer.model

    # free up GPU memory
    pretrained_model.to('cpu')
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    #






    print("Starting ADMET fine-tuning...")
    N_FINETUNE_EPOCHS = 100

    if params.more_tasks:       
        from metadata import more_tasks
        datasets_to_use = {**admet_metrics, **more_tasks}
    else:
        datasets_to_use = admet_metrics

    df_train, df_valid, df_test, task_dict = load_data(datasets_to_use, params.loss_reduction)

    if params.smi_leakage_method != 'none':
        df_train = remove_overlap(df_train, df_test)
        if params.smi_leakage_method == 'test+valid':
            df_train = remove_overlap(df_train, df_valid)

    label_cols = [c for c in df_train.columns if c != 'smi']

    partial_dataset = partial(SparseMultitaskDataset, label_cols=label_cols, ksteps=params.model_encoder_pe_dim)
    train_dataset = partial_dataset(df=df_train)
    valid_dataset = partial_dataset(df=df_valid)
    test_dataset  = partial_dataset(df=df_test)

    train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size,   shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params.train_batch_size*2, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=params.train_batch_size*2, shuffle=False)

    scheduler_param['warmup_steps'] = len(train_loader)
    scheduler_param['T_max'] = len(train_loader) * N_FINETUNE_EPOCHS
    
    decoders: nn.ModuleDict = get_decoders(task_dict=task_dict,
                                           in_dim=64, 
                                           hidden_dim=params.model_decoder_channels,
                                           num_layers=params.model_decoder_num_layers, 
                                           dropout=params.model_decoder_dropout)

    trainer = Trainer(task_dict=task_dict,
                      weighting="EW",
                      architecture='HPS', 
                      encoder_class=encoder_class, 
                      decoders=decoders, 
                      rep_grad=params.rep_grad, 
                      multi_input=False, 
                      optim_param=optim_param, 
                      scheduler_param=scheduler_param, 
                      save_path=os.path.join(params.save_path, 'tdc'),
                      load_path=params.load_path,
                      **kwargs)
    
    trainer.meter.log_wandb = types.MethodType(build_stage_logger('finetune'), trainer.meter)
    trainer.model = copy_encoder_weights(pretrained_model, trainer.model)
    trainer.model = freeze_encoder(trainer.model)
    
    # free up GPU memory
    del pretrained_model
    gc.collect()
    torch.cuda.empty_cache()
    # 

    trainer.train(train_loader, valid_loader, epochs=N_FINETUNE_EPOCHS)

    print("Evaluating on TDC ADMET")
    evaluator = CheckpointEvaluator(trainer, test_loader, wandb.run.id, task_dict, os.path.join(params.save_path, 'tdc'))

    if params.save_path is not None:
        for ckpt_selection_method in params.eval_methods:
            print(f'Evaluating with {ckpt_selection_method} method')
            evaluator.evaluate_by_method(ckpt_selection_method, N_FINETUNE_EPOCHS)
    else:
        trainer.test(test_loader, mode='test', reinit=False)
        results = trainer.meter.results.copy()
        wandb.log(results)

    sys.exit(0)
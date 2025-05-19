import os
import types
import wandb
import numpy as np
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args

from metadata import admet_metrics
from helper_functions import parse_args, build_stage_logger
from evaluator import CheckpointEvaluator
from examples.tdc.pm6_utils import dataloader_factory   
from data import SparseMultitaskDataset, load_data, remove_overlap
from model import GPS, get_decoders, freeze_encoder, copy_encoder_weights


if __name__ == '__main__':
    params = parse_args(LibMTL_args)
    kwargs, optim_param, scheduler_param = prepare_args(params)
    set_device(params.gpu_id); set_random_seed(params.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scheduler_param = {
        'scheduler': 'cos',
        'eta_min': params.lr * 0.01,
        'warmup_start_factor': 0.01
    }
    model_param = {
        'channels': params.model_encoder_channels, 
        'pe_dim': params.model_encoder_pe_dim, 
        'num_layers': params.model_encoder_num_layers, 
        'attn_type': 'multihead', 
        'attn_kwargs': {'dropout': params.model_encoder_dropout}
    }

    all_params = vars(params) | optim_param | scheduler_param | model_param
    
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if params.save_path is not None:
        params.save_path = os.path.join(params.save_path, f'{params.weighting}_{date_str}')
        os.makedirs(params.save_path, exist_ok=True)
        os.makedirs(os.path.join(params.save_path, 'pm6'), exist_ok=True)
        os.makedirs(os.path.join(params.save_path, 'tdc'), exist_ok=True)
    
    wandb.init(name=f'{params.weighting}_{date_str}',               
               save_code=True,
               config=all_params,
               entity='BorgwardtLab',
               project='libmtl_tdc',
               mode='online' if params.wandb else 'disabled')

    wandb.define_metric('pre_step')
    wandb.define_metric('pretrain/*', step_metric='pre_step')
    wandb.define_metric('ft_step')
    wandb.define_metric('finetune/*', step_metric='ft_step')

    def encoder_class():
        return GPS(**model_param)








    print("PM6 pretraining stage...")
    train_loader, valid_loader, _, task_dict = dataloader_factory(params.train_batch_size, "data/pm6/")  
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
                      **kwargs)
    
    trainer.meter.log_wandb = types.MethodType(build_stage_logger('pretrain'), trainer.meter)
    trainer.train(train_loader, valid_loader, epochs=params.epochs)
    pretrained_model = trainer.model







    print("Starting ADMET fine-tuning...")
    N_FINETUNE_EPOCHS = 150

    if params.more_tasks:       
        from metadata import more_tasks
        datasets_to_use = {**admet_metrics, **more_tasks}
    else:
        datasets_to_use = admet_metrics

    df_train, df_valid, df_test, task_dict = load_data(datasets_to_use, params.loss_reduction)
    scheduler_param['warmup_steps'] = len(train_loader)
    scheduler_param['T_max'] = len(train_loader) * N_FINETUNE_EPOCHS

    if params.smi_leakage_method != 'none':
        df_train = remove_overlap(df_train, df_test)
        if params.smi_leakage_method == 'test+valid':
            df_train = remove_overlap(df_train, df_valid)

    label_cols = [c for c in df_train.columns if c != 'smi']

    transform = T.AddRandomWalkPE(walk_length=params.model_encoder_pe_dim, attr_name='pe')

    partial_dataset = partial(SparseMultitaskDataset, label_cols=label_cols, transform=transform)
    train_dataset = partial_dataset(df_train)
    valid_dataset = partial_dataset(df_valid)
    test_dataset  = partial_dataset(df_test)

    partial_loader = partial(DataLoader, pin_memory=True)
    train_loader = partial_loader(train_dataset, batch_size=params.train_batch_size,   shuffle=True)
    valid_loader = partial_loader(valid_dataset, batch_size=params.train_batch_size*2, shuffle=False)
    test_loader  = partial_loader(test_dataset,  batch_size=params.train_batch_size*2, shuffle=False)

    decoders: nn.ModuleDict = get_decoders(task_dict=task_dict,
                                           in_dim=params.model_encoder_channels, 
                                           hidden_dim=params.model_decoder_channels,
                                           num_layers=params.model_decoder_num_layers, 
                                           dropout=params.model_decoder_dropout)


    if params.weighting_finetune == 'FairGrad':
        kwargs['weight_args']['FairGrad_alpha'] = 0.75
    elif params.weighting_finetune == 'DB_MTL':
        kwargs['weight_args']['DB_beta'] = 0.9
        kwargs['weight_args']['DB_beta_sigma'] = 0.0

    trainer = Trainer(task_dict=task_dict,
                      weighting=params.weighting_finetune,
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

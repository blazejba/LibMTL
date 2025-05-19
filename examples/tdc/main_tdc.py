import os
import wandb
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
from model import GPS, get_decoders
from helper_functions import parse_args
from evaluator import CheckpointEvaluator
from data import SparseMultitaskDataset, load_data, remove_overlap


def get_sharing_factor(model, decoders):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decoder_params = sum(sum(p.numel() for p in decoder.parameters() if p.requires_grad)
                                       for decoder in decoders.values())
    return total_params / decoder_params


if __name__ == '__main__':
    params = parse_args(LibMTL_args)
    kwargs, optim_param, scheduler_param = prepare_args(params)
    set_device(params.gpu_id); set_random_seed(params.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scheduler_param = {
        'scheduler': 'reduce',
        'mode': 'max',
        'factor': params.lr_factor, 
        'patience': params.patience,
        'min_lr': 0.00001
    }
    model_param = {
        'channels': params.model_encoder_channels, 
        'pe_dim': params.model_encoder_pe_dim, 
        'num_layers': params.model_encoder_num_layers, 
        'attn_type': 'multihead', 
        'attn_kwargs': {'dropout': params.model_encoder_dropout}
    }

    all_params = vars(params) | optim_param | scheduler_param | model_param
    
    if params.save_path is not None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        params.save_path = os.path.join(params.save_path, f'{params.arch}_{params.weighting}_{date_str}')
        os.makedirs(params.save_path, exist_ok=True)
    
    wandb.init(name=f'{params.arch}_{params.weighting}',               
               save_code=True,
               config=all_params,
               entity='BorgwardtLab',
               project='libmtl_tdc',
               mode='online' if params.wandb else 'disabled')

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

    transform = T.AddRandomWalkPE(walk_length=params.model_encoder_pe_dim, attr_name='pe')

    partial_dataset = partial(SparseMultitaskDataset, label_cols=label_cols, transform=transform)
    train_dataset = partial_dataset(df_train)
    valid_dataset = partial_dataset(df_valid)
    test_dataset  = partial_dataset(df_test)

    partial_loader = partial(DataLoader, pin_memory=True)
    train_loader = partial_loader(train_dataset, batch_size=params.train_batch_size, shuffle=True)
    valid_loader = partial_loader(valid_dataset, batch_size=params.train_batch_size*2, shuffle=False)
    test_loader  = partial_loader(test_dataset,  batch_size=params.train_batch_size*2, shuffle=False)

    def encoder_class():
        return GPS(**model_param)
    
    decoders: nn.ModuleDict = get_decoders(task_dict=task_dict,
                                           in_dim=params.model_encoder_channels, 
                                           hidden_dim=params.model_decoder_channels,
                                           num_layers=params.model_decoder_num_layers, 
                                           dropout=params.model_decoder_dropout)
    
    sharing_factor = get_sharing_factor(encoder_class(), decoders)
    print(f'Sharing factor: {sharing_factor}')
    wandb.config.update({'sharing_factor': sharing_factor})

    trainer = Trainer(task_dict=task_dict,
                      weighting=params.weighting,
                      architecture=params.arch, 
                      encoder_class=encoder_class, 
                      decoders=decoders, 
                      rep_grad=params.rep_grad, 
                      multi_input=False, 
                      optim_param=optim_param, 
                      scheduler_param=scheduler_param, 
                      save_path=params.save_path,
                      load_path=params.load_path,
                      **kwargs)

    trainer.train(train_loader, valid_loader, epochs=params.epochs)

    evaluator = CheckpointEvaluator(trainer, test_loader, wandb.run.id, task_dict, params.save_path)

    if params.save_path is not None:
        for ckpt_selection_method in params.eval_methods:
            print(f'Evaluating with {ckpt_selection_method} method')
            evaluator.evaluate_by_method(ckpt_selection_method, params.epochs)
    else:
        trainer.test(test_loader, mode='test', reinit=False)
        results = trainer.meter.results.copy()
        wandb.log(results)



import os
import time
import wandb
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args

from stats import get_meta_info
from model import GPS, get_decoders
from data import ADMEDataset, load_data
from metadata import admet_metrics
from evaluator import CheckpointEvaluator


def parse_args(parser):
    # model
    parser.add_argument('--loss-reduction', default='sum', choices=['mean', 'sum'])
    ## encoder
    parser.add_argument('--model-encoder-channels', default=64, type=int)
    parser.add_argument('--model-encoder-pe-dim', default=20, type=int)
    parser.add_argument('--model-encoder-num-layers', default=3, type=int)
    parser.add_argument('--model-encoder-dropout', default=0.5, type=float)
    ## decoder
    parser.add_argument('--model-decoder-channels', default=64, type=int)
    parser.add_argument('--model-decoder-num-layers', default=2, type=int)
    parser.add_argument('--model-decoder-dropout', default=0.1, type=float)
    # training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr-factor', default=0.9, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--train-batch-size', default=512, type=int)
    # evaluation
    parser.add_argument('--eval-methods', default=['improvement', 'pps', 'last', 'independent'], nargs='+', type=str)
    # misc
    parser.add_argument('--wandb', action='store_true', help='use wandb')

    args = parser.parse_args()
    assert len(args.eval_methods) > 0, 'No evaluation method provided'
    print(args.eval_methods)
    return args


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
    
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    params.save_path = os.path.join(params.save_path, f'{params.arch}_{params.weighting}_{date_str}')
    os.makedirs(params.save_path, exist_ok=True)
    
    wandb.init(name=f'{params.arch}_{params.weighting}',               
               save_code=True,
               config=all_params,
               entity='BorgwardtLab',
               project='libmtl_tdc',
               mode='online' if params.wandb else 'disabled')

    df_train, df_valid, df_test, task_dict = load_data(admet_metrics,
                                                       loss_reduction=params.loss_reduction)
    get_meta_info(df_train, df_valid, df_test)

    label_cols = [c for c in df_train.columns if c != 'smi']

    transform = T.AddRandomWalkPE(walk_length=params.model_encoder_pe_dim, attr_name='pe')

    train_dataset = ADMEDataset(df_train, label_cols, transform=transform)
    valid_dataset = ADMEDataset(df_valid, label_cols, transform=transform)
    test_dataset  = ADMEDataset(df_test,  label_cols, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, 
                              shuffle=True,  pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params.train_batch_size*2, 
                              shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=params.train_batch_size*2, 
                              shuffle=False, pin_memory=True)

    def encoder_class():
        return GPS(**model_param)

    decoders: nn.ModuleDict = get_decoders(task_dict=task_dict,
                                           channels=params.model_encoder_channels, 
                                           num_layers=params.model_decoder_num_layers, 
                                           dropout=params.model_decoder_dropout)

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
    for ckpt_selection_method in params.eval_methods:
        print(f'Evaluating with {ckpt_selection_method} method')
        evaluator.evaluate_by_method(ckpt_selection_method, params.epochs)

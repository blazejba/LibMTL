import wandb

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


def parse_args(parser):
    # model
    ## encoder
    parser.add_argument('--model-encoder-channels', default=64, type=int, help='model encoder channels')
    parser.add_argument('--model-encoder-pe-dim', default=20, type=int, help='model encoder pe dim')
    parser.add_argument('--model-encoder-num-layers', default=3, type=int, help='model encoder num layers')
    parser.add_argument('--loss-reduction', default='sum', choices=['mean', 'sum'], type=str, help='loss reduction')
    ## decoder
    parser.add_argument('--model-decoder-channels', default=64, type=int, help='model decoder channels')
    parser.add_argument('--model-decoder-num-layers', default=2, type=int, help='model decoder num layers')
    parser.add_argument('--model-decoder-dropout', default=0.1, type=float, help='model decoder dropout')
    # training
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--lr-factor', default=0.9, type=float, help='learning rate factor')
    parser.add_argument('--patience', default=5, type=int, help='patience')
    parser.add_argument('--train-batch-size', default=512, type=int, help='batch size')
    # misc
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    return parser.parse_args()


def minmax_normalize(history):
    return (history - history.min(axis=0)) / (history.max(axis=0) - history.min(axis=0))


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
        'attn_kwargs': {'dropout': 0.5}
    }

    all_params = vars(params) | optim_param | scheduler_param | model_param
    
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
    
    if params.arch in ['MMoE', 'CGC', 'PLE']:
        sample = train_dataset[0]
        node_dim = sample.x.shape[1]
    
        kwargs.update({
            'input_type': 'graph',
            'node_dim': node_dim
        })

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
    best_epoch = trainer.meter.best_result['epoch']
    trainer.test(test_loader, epoch=best_epoch, mode='test', reinit=False)
    
    from metadata import leaderboard
    import numpy as np

    ranks = []
    test_results = trainer.meter.results
    for task_name in task_dict.keys():
        leaderboard_scores = leaderboard[task_name]
        scores = np.array(leaderboard_scores)
        if task_dict[task_name]['weight'] == [1]:  # higher is better
            rank = np.sum(scores > test_results[task_name]) + 1
        else:  # lower is better
            rank = np.sum(scores < test_results[task_name]) + 1
        ranks.append(rank)    
    wandb.log({'test/average_rank': sum(ranks) / len(ranks)})

    trainer.meter.reinit()

    import time
    time.sleep(45)

    run_id = wandb.run.id
    api = wandb.Api()
    run_path = f"BorgwardtLab/libmtl_tdc/runs/{run_id}"
    run = api.run(run_path)
    history = run.history()

    phase = 'val/'

    spearman_cols = [col for col in history.columns if phase in col and '_spearman' in col]
    mae_cols      = [col for col in history.columns if phase in col and '_mae' in col]
    roc_auc_cols  = [col for col in history.columns if phase in col and '_roc-auc' in col]
    pr_auc_cols   = [col for col in history.columns if phase in col and '_pr-auc' in col]

    spearman_history = history[spearman_cols]
    mae_history      = history[mae_cols]
    roc_auc_history  = history[roc_auc_cols]
    pr_auc_history   = history[pr_auc_cols]

    spearman_history_normed = minmax_normalize(spearman_history)
    mae_history_normed      = 1 - minmax_normalize(mae_history)
    roc_auc_history_normed  = minmax_normalize(roc_auc_history)
    pr_auc_history_normed   = minmax_normalize(pr_auc_history)

    all_metrics = np.concatenate([mae_history_normed.values[:-1],
                                  spearman_history_normed.values[:-1],
                                  roc_auc_history_normed.values[:-1],
                                  pr_auc_history_normed.values[:-1]], axis=1)

    metric_product = np.prod(all_metrics, axis=1)
    pps_idx = np.argmax(metric_product)
    pps = metric_product[pps_idx]
    print(f"all_metrics: {all_metrics}")
    print(f"metric_product: {metric_product}")
    print(f"pps_idx: {pps_idx}")
    print(f"pps: {pps}")

    trainer.test(test_loader, epoch=pps_idx, mode='test', reinit=False)
    ranks = []
    test_results = trainer.meter.results
    for task_name in task_dict.keys():
        leaderboard_scores = leaderboard[task_name]
        scores = np.array(leaderboard_scores)
        if task_dict[task_name]['weight'] == [1]: 
            rank = np.sum(scores > test_results[task_name]) + 1
        else:
            rank = np.sum(scores < test_results[task_name]) + 1
        ranks.append(rank)    
    wandb.log({'test/average_rank_pps': sum(ranks) / len(ranks)})
    wandb.log({'val/pps_max': pps})
    for i, mp_i in enumerate(metric_product):
        wandb.log({'val/metric_product': mp_i, 'epoch': i})

#wandb sweep --entity BorgwardtLab --project libmtl_tdc  sweep_hparams.yaml
#wandb sweep --entity BorgwardtLab --project libmtl_tdc  sweep_archs.yaml
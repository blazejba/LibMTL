import wandb

import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.utils import count_improvement

class _PerformanceMeter(object):
    def __init__(self, task_dict, multi_input, base_result=None):
        
        self.task_dict = task_dict
        self.multi_input = multi_input
        self.task_num = len(self.task_dict)
        self.task_name = list(self.task_dict.keys())
        
        self.higher_is_better_dict = {
            task: self.task_dict[task]['higher_is_better'] for task in self.task_name
        }
        self.base_result = base_result
        self.best_result = {'improvement': -1e+2, 'epoch': 0, 'result': 0}
        self.improvement = None
        
        self.losses = {task: self.task_dict[task]['loss_fn'] for task in self.task_name}
        self.metrics = {task: self.task_dict[task]['metrics_fn'] for task in self.task_name}
        
        self.results = {task:[] for task in self.task_name}
        self.loss_item = np.zeros(self.task_num)
        
        self.has_val = False
        
        self._init_display()
        
    def record_time(self, mode='begin'):
        if mode == 'begin':
            self.beg_time = time.time()
        elif mode == 'end':
            self.end_time = time.time()
        else:
            raise ValueError('No support time mode {}'.format(mode))
        
    def update(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)
        
    def get_score(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                self.results[task] = self.metrics[task].score_fun()
                self.loss_item[tn] = self.losses[task]._average_loss()
    
    def _init_display(self):
        print('='*40)
        print('LOG FORMAT | ', end='')
        for tn, task in enumerate(self.task_name):
            print(task+'_LOSS ', end='')
            for m in self.task_dict[task]['metrics']:
                print(m+' ', end='')
            print('| ', end='')
        print('TIME')
    
    def display(self, mode, epoch):
        if epoch is not None:
            if epoch == 0 and self.base_result is None and mode==('val' if self.has_val else 'test'):
                self.base_result = self.results
            if mode == 'train':
                print('Epoch: {:04d} | '.format(epoch), end='')
            if not self.has_val and mode == 'test':
                self._update_best_result(self.results, epoch)
            if self.has_val and mode != 'train':
                self._update_best_result_by_val(self.results, epoch, mode)
        if mode == 'train':
            p_mode = 'TRAIN'
        elif mode == 'val':
            p_mode = 'VAL'
        else:
            p_mode = 'TEST'
        print('{}: '.format(p_mode), end='')
        for tn, task in enumerate(self.task_name):
            print('{:.4f} '.format(self.loss_item[tn]), end='')
            for i in range(len(self.results[task])):
                print('{:.4f} '.format(self.results[task][i]), end='')
            print('| ', end='')
        print('Time: {:.4f}'.format(self.end_time-self.beg_time), end='')
        print(' | ', end='') if mode!='test' else print()

    def log_wandb(self, epoch, mode):
        log_dict = {}
        for tn, task in enumerate(self.task_name):
            log_dict[f"{mode}/{task}_loss"] = self.loss_item[tn]
            for metric_name, metric_value in zip(self.task_dict[task]['metrics'], self.results[task]):
                log_dict[f"{mode}/{task}_{metric_name}"] = metric_value

        log_dict[f"{mode}/overall_loss"] = np.sum(self.loss_item)

        total_score = {
            'mae': [],
            'roc-auc': [],
            'pr-auc': [],
            'spearman': [],
        }
        for task in self.task_name:
            for metric_name, metric_value in zip(self.task_dict[task]['metrics'], self.results[task]):
                total_score[metric_name].append(metric_value)
        for metric_name, metric_value in total_score.items():
            log_dict[f"{mode}/{metric_name}"] = np.mean(metric_value)

        wandb.log(log_dict, step=epoch)

    def display_best_result(self):
        print('='*40)
        print('Best Result: Epoch {}, result {}'.format(self.best_result['epoch'], self.best_result['result']))
        print('='*40)
        
    def _update_best_result_by_val(self, new_result, epoch, mode):
        if mode == 'val':
            improvement = count_improvement(self.base_result, new_result, self.higher_is_better_dict)
            self.improvement = improvement
            if improvement > self.best_result['improvement']:
                self.best_result['improvement'] = improvement
                self.best_result['epoch'] = epoch
        else:
            if epoch == self.best_result['epoch']:
                self.best_result['result'] = new_result
        
    def _update_best_result(self, new_result, epoch):
        improvement = count_improvement(self.base_result, new_result, self.higher_is_better_dict)
        self.improvement = improvement
        if improvement > self.best_result['improvement']:
            self.best_result['improvement'] = improvement
            self.best_result['epoch'] = epoch
            self.best_result['result'] = new_result
        
    def reinit(self):
        for task in self.task_name:
            self.losses[task]._reinit()
            self.metrics[task].reinit()
        self.loss_item = np.zeros(self.task_num)
        self.results = {task:[] for task in self.task_name}


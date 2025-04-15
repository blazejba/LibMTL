import os
import time
import wandb
import numpy as np
import torch

from metadata import leaderboard

class CheckpointEvaluator:
    def __init__(self, trainer, test_loader, wandb_run_id, task_dict, save_path):
        self.history = self.get_wandb_history(wandb_run_id)
        self.trainer = trainer
        self.test_loader = test_loader
        self.task_dict = task_dict
        self.save_path = save_path
        self.cache = {}

    def clean_task_name(self, task_name: str) -> str:
        return "_".join(task_name.replace('val/', '').split('_')[:-1])

    def evaluate_epoch(self, epoch: int, tasks: list = None):
        if epoch in self.cache:
            print(f'Using cached results for epoch {epoch}')
            results = self.cache[epoch]
        else:
            print(f'Running evaluation for epoch {epoch}')
            ckpt_path = os.path.join(self.save_path, f'epoch_{epoch}.pt')
            self.trainer.model.load_state_dict(torch.load(ckpt_path), strict=False)
            self.trainer.test(self.test_loader, mode='test', reinit=False)
            results = self.trainer.meter.results.copy()
            self.cache[epoch] = results
            self.trainer.meter.reinit()
        if tasks is not None:
            subset = {}
            for t in tasks:
                key = self.clean_task_name(t)
                if key in results:
                    subset[key] = results[key]
            return subset
        return results
    
    @staticmethod
    def minmax_normalize(history):
        return (history - history.min(axis=0)) / (history.max(axis=0) - history.min(axis=0))

    @staticmethod
    def get_wandb_history(run_id):
        api = wandb.Api()
        run_path = f"BorgwardtLab/libmtl_tdc/runs/{run_id}"
        run = api.run(run_path)
        return run.history()

    def get_average_rank(self, test_results: dict):
        ranks = []
        for task_name in self.task_dict.keys():
            if task_name not in leaderboard:
                continue
            leaderboard_scores = leaderboard[task_name]
            scores = np.array(leaderboard_scores)
            if self.task_dict[task_name]['weight'] == [1]:  # higher is better
                rank = np.sum(scores > test_results[task_name]) + 1
            else:  # lower is better
                rank = np.sum(scores < test_results[task_name]) + 1
            ranks.append(rank)
        return sum(ranks) / len(ranks)

    def get_metric_timeseries(self, metric: str):
        return [
            c for c in self.history.columns 
            if 'val/' in c and f"_{metric}" in c and 
            self.clean_task_name(c) in leaderboard.keys()
        ]

    def evaluate_by_method(self, method: str, n_epochs: int = None):
        log_dict = {}

        if method == 'improvement':
            selected_epoch = self.trainer.meter.best_result['epoch']
            results = self.evaluate_epoch(selected_epoch)
            avg_rank = self.get_average_rank(results)
            log_dict['test/average_rank_improvement'] = avg_rank

        elif method == 'pps':
            time.sleep(45)
            sp_cols  = self.get_metric_timeseries('spearman')
            mae_cols = self.get_metric_timeseries('mae')
            roc_cols = self.get_metric_timeseries('roc-auc')
            pr_cols  = self.get_metric_timeseries('pr-auc')
            sp_norm  = self.minmax_normalize(self.history[sp_cols].dropna())
            mae_norm = 1 - self.minmax_normalize(self.history[mae_cols].dropna())
            roc_norm = self.minmax_normalize(self.history[roc_cols].dropna())
            pr_norm  = self.minmax_normalize(self.history[pr_cols].dropna())
            all_metrics = np.concatenate([mae_norm.values, sp_norm.values,
                                          roc_norm.values, pr_norm.values], axis=1)
            metric_product = np.prod(all_metrics, axis=1)
            selected_epoch = int(np.argmax(metric_product))
            log_dict['val/pps'] = float(metric_product[selected_epoch])
            results = self.evaluate_epoch(selected_epoch)
            avg_rank = self.get_average_rank(results)
            log_dict['test/average_rank_pps'] = avg_rank

        elif method == 'last':
            if n_epochs is None:
                raise ValueError("n_epochs must be provided for 'last' method")
            selected_epoch = n_epochs - 1
            results = self.evaluate_epoch(selected_epoch)
            avg_rank = self.get_average_rank(results)
            log_dict['test/average_rank_last'] = avg_rank

        elif method == 'independent':
            time.sleep(45)
            sp_cols  = self.get_metric_timeseries('spearman')
            mae_cols = self.get_metric_timeseries('mae')
            roc_cols = self.get_metric_timeseries('roc-auc')
            pr_cols  = self.get_metric_timeseries('pr-auc')
            higher_better = sp_cols + roc_cols + pr_cols
            lower_better  = mae_cols

            task_ckpt_map = {}
            for col in higher_better:
                progress = self.history[col].dropna()
                task_ckpt_map[col] = progress.idxmax()
            for col in lower_better:
                progress = self.history[col].dropna()
                task_ckpt_map[col] = progress.idxmin()

            epoch_task_map = {}
            for task, ep in task_ckpt_map.items():
                epoch_task_map.setdefault(ep, []).append(task)

            test_results = {}
            for ep, tasks in epoch_task_map.items():
                res = self.evaluate_epoch(ep, tasks)
                for task in tasks:
                    key = self.clean_task_name(task)
                    if key in res:
                        test_results[key] = res[key]
            avg_rank = self.get_average_rank(test_results)
            log_dict['test/average_rank_independent'] = avg_rank

        else:
            raise ValueError(f"Unknown method: {method}")

        wandb.log(log_dict)
        self.trainer.meter.reinit()

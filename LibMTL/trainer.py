import os
import datetime

import wandb

import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast, GradScaler

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method

# def _parse_slurm_time(t: str) -> int:
#     """
#     Convert a SLURM time string like 'D-HH:MM:SS' or 'HH:MM:SS' to seconds.
#     """
#     if t is None:
#         return None
#     if '-' in t:
#         days, rest = t.split('-')
#         h, m, s = map(int, rest.split(':'))
#         return int(days) * 86400 + h * 3600 + m * 60 + s
#     else:
#         h, m, s = map(int, t.split(':'))
#         return h * 3600 + m * 60 + s
    
    
class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                            functions, meaning how to update thoes objectives in the training process and obtain the final \
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \
                            metric, where ``1`` means the higher the score is, the better the performance, \
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param,
                 save_path=None, load_path=None, time_limit=None, **kwargs):
        super(Trainer, self).__init__()
        
        self.device = torch.device('cuda:0')
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path
        self.weighting = weighting


        self.epoch_duration = []
        # self.time_limit = _parse_slurm_time(time_limit)
        self.start_time = datetime.datetime.now()
        self.start_epoch = 0

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param.copy())

        # if self.load_path is not None:
        #     self._resume_training(self.load_path)
        
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)

        # Enable GradScaler for mixed-precision training
        self.scaler = GradScaler()
        
    # def _is_enough_time(self):
    #     if self.time_limit is not None:
    #         return (datetime.datetime.now() - self.start_time).total_seconds() < self.time_limit
    #     return True
    
    def _prepare_model(self, weighting, architecture, encoder_class, decoders):

        weighting_class = weighting_method.__dict__[weighting] 
        architecture_class = architecture_method.__dict__[architecture]
        
        class MTLmodel(architecture_class, weighting_class):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))
        count_parameters(self.model)

    # def _resume_training(self, ckpt_path):
    #     if os.path.isdir(ckpt_path):
    #         ckpt_path = os.path.join(ckpt_path, 'last.pt')
    #     checkpoint = torch.load(ckpt_path, map_location=self.device)

    #     if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    #         self.model.load_state_dict(checkpoint['model_state'], strict=False)
    #         if checkpoint.get('optimizer_state') and self.optimizer is not None:
    #             self.optimizer.load_state_dict(checkpoint['optimizer_state'])
    #         if self.scheduler is not None and checkpoint.get('scheduler_state'):
    #             self.scheduler.load_state_dict(checkpoint['scheduler_state'])
    #         self.start_epoch = checkpoint.get('epoch', -1) + 1
    #         self.epoch_duration = checkpoint.get('epoch_duration', [])
    #         print(f'Loaded checkpoint from {ckpt_path}. Resuming at epoch {self.start_epoch}.')
    #     else:
    #         self.model.load_state_dict(checkpoint, strict=False)
    #         print(f'Loaded model weights from {ckpt_path} (no optimizer/scheduler).')
        
    def _prepare_optimizer(self, optim_param: dict, scheduler_param: dict):
        optim_dict = {
            'sgd':  torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'adagrad': torch.optim.Adagrad,
            'rmsprop': torch.optim.RMSprop,
        }
        scheduler_dict = {
            'exp':  torch.optim.lr_scheduler.ExponentialLR,
            'step': torch.optim.lr_scheduler.StepLR,
            'cos':  torch.optim.lr_scheduler.CosineAnnealingLR,
            'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
        }

        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)

        self.scheduler = None
        if scheduler_param is not None:
            warmup_steps   = scheduler_param.pop('warmup_steps', 0)
            warmup_factor  = scheduler_param.pop('warmup_start_factor', .1)
            scheduler_name = scheduler_param.pop('scheduler')
            scheduler_arg  = scheduler_param

            main_sched = scheduler_dict[scheduler_name](self.optimizer, **scheduler_arg)

            if warmup_steps > 0:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=warmup_factor,
                    total_iters=warmup_steps
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, main_sched],
                    milestones=[warmup_steps]
                )
            else:
                self.scheduler = main_sched

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
    
    def process_preds(self, preds, task_name=None):
        r'''The processing of prediction for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses
        
    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def forward4loss(self, model, inputs, gts, return_preds=False):
        if not self.multi_input:
            preds = model(inputs)
            preds = self.process_preds(preds)
            losses = self._compute_loss(preds, gts)
        else:
            losses = torch.zeros(self.task_num).to(self.device)
            preds = {}
            for tn, task in enumerate(self.task_name):
                inputs_t, gts_t = inputs[task], gts[task]
                preds_t = model(inputs_t, task)
                preds_t = preds_t[task]
                preds_t = self.process_preds(preds_t, task)
                losses[tn] = self._compute_loss(preds_t, gts_t, task)
                if return_preds:
                    preds[task] = preds_t
        if return_preds:
            return losses, preds
        else:
            return losses

    def train(self, train_dataloaders, val_dataloaders, epochs, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(self.start_epoch, epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                train_losses = []
                for sample_num in range(3 if self.weighting in ['MoDo', 'SDMGrad'] else 1):
                    if not self.multi_input:
                        train_inputs, train_gts = self._process_data(train_loader)
                    else:
                        train_inputs, train_gts = {}, {}
                        for tn, task in enumerate(self.task_name):
                            train_input, train_gt = self._process_data(train_loader[task])
                            train_inputs[task], train_gts[task] = train_input, train_gt

                    with autocast(dtype=torch.bfloat16, device_type="cuda"):
                        train_losses_, train_preds = self.forward4loss(self.model, train_inputs, train_gts, return_preds=True)
                    train_losses.append(train_losses_)
                train_losses = torch.stack(train_losses).squeeze(0)

                if not self.multi_input:
                    self.meter.update(train_preds, train_gts)
                else:
                    for tn, task in enumerate(self.task_name):
                        self.meter.update(train_preds[task], train_gts[task], task)

                self.optimizer.zero_grad(set_to_none=False)

                # Weighting methods differ: STCH returns (weights, scalar_loss),
                # others still perform .backward() internally and return only weights.
                with autocast(dtype=torch.bfloat16, device_type="cuda"):
                    backward_out = self.model.backward(train_losses, **self.kwargs['weight_args'])

                if isinstance(backward_out, tuple):
                    w, scalar_loss = backward_out
                    if w is not None:
                        self.batch_weight[:, epoch, batch_index] = w
                    self.scaler.scale(scalar_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    w = backward_out
                    if w is not None:
                        self.batch_weight[:, epoch, batch_index] = w
                    self.optimizer.step()

                if self.weighting == 'FAMO':
                    with torch.no_grad():
                        new_train_losses = self.forward4loss(self.model, train_inputs, train_gts, return_preds=False)
                        self.model.update_w(new_train_losses.detach())
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.log_wandb(epoch=epoch, mode='train')
            self.meter.reinit()
            
            self.meter.has_val = True
            val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(val_dataloaders, epoch, mode='test', wandb=False)

            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                    wandb.log({'lr': self.scheduler.get_last_lr()[0]})
                else:
                    self.scheduler.step()
                    
            if self.save_path:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, f'epoch_{epoch}.pt'))
                # torch.save({
                #     'epoch': epoch,
                #     'model_state': self.model.state_dict(),
                #     'optimizer_state': self.optimizer.state_dict(),
                #     'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                #     'epoch_duration': self.epoch_duration,
                # }, os.path.join(self.save_path, 'last.pt'))
                # print(f'Saved checkpoint for epoch {epoch} to {os.path.join(self.save_path, "last.pt")}')

                # if self.time_limit is not None:
                #     epoch_end_time = datetime.datetime.now()
                #     self.epoch_duration.append((epoch_end_time - epoch_start_time).total_seconds())
                #     if len(self.epoch_duration) >= 3: 
                #         est = np.percentile(self.epoch_duration, 95)
                #         elapsed = (epoch_end_time - self.start_time).total_seconds()
                #         remaining = self.time_limit - elapsed

                #         if remaining < est * 1.05:
                #             print(f"Not enough wall-time left (≈{remaining:.0f}s). Exiting with code 99 for resubmission.")
                #             sys.exit(99)
        self.meter.display_best_result()

        if return_weight:
            return self.batch_weight


    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False, wandb=True, reinit=True):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            for batch_index in range(test_batch):
                test_inputs, test_gts = self._process_data(test_loader)
                test_preds = self.model(test_inputs)
                test_preds = self.process_preds(test_preds)
                test_losses = self._compute_loss(test_preds, test_gts)
                self.meter.update(test_preds, test_gts)

        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        if wandb:
            self.meter.log_wandb(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        if reinit:
            self.meter.reinit()
        if return_improvement:
            return improvement
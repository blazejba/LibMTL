import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from LibMTL.loss import AbsLoss
from LibMTL.metrics import AbsMetric


class SparseMSELoss(AbsLoss):
    def __init__(self, loss_reduction: str = 'mean'):
        super(SparseMSELoss, self).__init__()
        self.loss_reduction = loss_reduction
        self.loss_fn = nn.MSELoss(reduction=loss_reduction)

    def compute_loss(self, pred, gt):
        pred = pred.squeeze()
        mask = ~torch.isnan(gt)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        loss = self.loss_fn(pred_valid, gt_valid)
        if self.loss_reduction == 'sum':
            loss = loss / pred.size(0)
        return loss

class SparseCELoss(AbsLoss):
    def __init__(self, loss_reduction: str = 'mean'):
        super(SparseCELoss, self).__init__()
        self.loss_reduction = loss_reduction
        self.loss_fn = nn.CrossEntropyLoss(reduction=loss_reduction)

    def compute_loss(self, pred, gt):
        mask = ~torch.isnan(gt)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred_valid = pred[mask]
        gt_valid = gt[mask].long()
        loss = self.loss_fn(pred_valid, gt_valid)
        if self.loss_reduction == 'sum':
            loss = loss / pred.size(0)
        return loss


class SparseAcc(AbsMetric):
    def __init__(self):
        super(SparseAcc, self).__init__()

    def update_fun(self, pred, gt):
        mask = ~torch.isnan(gt)
        if mask.sum() == 0:
            return
        gt_valid = gt[mask].long()
        pred_valid = pred[mask]
        pred_labels = F.softmax(pred_valid, dim=-1).max(-1)[1]
        correct = gt_valid.eq(pred_labels).sum().item()
        self.record.append(correct)
        self.bs.append(mask.sum().item())

    def score_fun(self):
        if not self.record:
            return [0.0]
        total_correct = sum(self.record)
        total_samples = sum(self.bs)
        return [total_correct / total_samples if total_samples > 0 else 0.0]

class SparseL1(AbsMetric):
    def __init__(self):
        super(SparseL1, self).__init__()

    def update_fun(self, pred, gt):
        if pred.dim() > 1 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        mask = ~torch.isnan(gt)
        if mask.sum() == 0:
            return
        abs_err = torch.abs(pred[mask] - gt[mask])
        mean_err = abs_err.mean().item()
        self.record.append(mean_err)
        self.bs.append(mask.sum().item())

    def score_fun(self):
        if not self.record:
            return [0.0]
        total_err = sum([r * b for r, b in zip(self.record, self.bs)])
        total_samples = sum(self.bs)
        return [total_err / total_samples if total_samples > 0 else 0.0]

class SparseAUROC(AbsMetric):
    def __init__(self):
        super(SparseAUROC, self).__init__()
        self.y_true_all = []
        self.y_score_all = []

    def update_fun(self, pred, gt):
        y_score = F.softmax(pred, dim=-1)[:, 1]
        mask = ~torch.isnan(gt)
        if mask.sum() > 0:
            gt_valid = gt[mask].cpu().numpy()
            y_score_valid = y_score[mask].detach().cpu().numpy()
            self.y_true_all.extend(gt_valid)
            self.y_score_all.extend(y_score_valid)

    def score_fun(self):
        if not self.y_true_all:
            return [0.0]
        try:
            score = roc_auc_score(self.y_true_all, self.y_score_all)
        except ValueError:
            score = 0.0
        return [score]

    def reinit(self):
        self.y_true_all = []
        self.y_score_all = []


class SparseAUPRC(AbsMetric):
    def __init__(self):
        super(SparseAUPRC, self).__init__()
        self.y_true_all = []
        self.y_score_all = []

    def update_fun(self, pred, gt):
        y_score = F.softmax(pred, dim=-1)[:, 1]
        mask = ~torch.isnan(gt) & ~torch.isnan(y_score)
        if torch.isnan(y_score).any():
            print(f"NaNs in predictions: {torch.isnan(y_score).sum()}")
        if mask.sum() > 0:
            gt_valid = gt[mask].cpu().numpy()
            y_score_valid = y_score[mask].detach().cpu().numpy()
            self.y_true_all.extend(gt_valid)
            self.y_score_all.extend(y_score_valid)

    def score_fun(self):
        if not self.y_true_all:
            return [0.0]
        score = average_precision_score(self.y_true_all, self.y_score_all)
        return [score]

    def reinit(self):
        self.y_true_all = []
        self.y_score_all = []


class SparseSpearman(AbsMetric):
    def __init__(self):
        super(SparseSpearman, self).__init__()
        self.pred_all = []
        self.gt_all = []

    def update_fun(self, pred, gt):
        if pred.dim() > 1 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        mask = ~torch.isnan(gt)
        if mask.sum() > 0:
            pred_valid = pred[mask].detach().cpu().tolist()
            gt_valid = gt[mask].cpu().tolist()
            self.pred_all.extend(pred_valid)
            self.gt_all.extend(gt_valid)

    def score_fun(self):
        if len(self.gt_all) < 2:
            return [0.0]
        correlation, _ = spearmanr(self.pred_all, self.gt_all)
        if np.isnan(correlation):
            return [0.0]
        return [correlation]

    def reinit(self):
        self.pred_all = []
        self.gt_all = []

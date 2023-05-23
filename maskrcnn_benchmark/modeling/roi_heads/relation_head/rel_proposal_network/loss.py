import math

import ipdb
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.boxlist_ops import squeeze_tensor


def reporting_focal_loss_link_predictor(link_prob, rel_labels, loss, soft_label=None):
    # selected_cls_logits = link_prob[rel_labels != -1]
    selected_cls_logits = link_prob
    
    bin_logits = selected_cls_logits[:, -1]

    # selected_labels = rel_labels[rel_labels != -1].long()
    selected_labels = rel_labels.long()
    
    onehot = torch.zeros_like(bin_logits)

    if soft_label is not None:
        onehot = soft_label
    else:
        onehot[selected_labels > 0] = 1
        
    loss_val_bin = loss(inputs=bin_logits, targets=onehot, reduce=True)

    # return loss_val_bin
    return loss_val_bin


class GSL_Loss(nn.Module):
    def __init__(self, cfg):
        super(GSL_Loss, self).__init__()
        alpha = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.FOCAL_LOSS_ALPHA # 0.2
        gamma = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.FOCAL_LOSS_GAMMA # 2.0
        self.loss_module = FocalLossFGBGNormalization(alpha, gamma, logits=False)
        
    def forward(self, link_prob, rel_weak_labels):
        return reporting_focal_loss_link_predictor(link_prob, rel_weak_labels, self.loss_module)
    
class FocalLoss(nn.Module):
    def __init__(
        self, alpha=1.0, gamma=2.0, logits=False, reduce=True, ignored_label_idx=None
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.loss_type = torch.nn.BCELoss(reduction='none')
        self.ignored_label_idx = ignored_label_idx

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1.0 - pt) ** self.gamma * BCE_loss
        if self.ignored_label_idx:
            F_loss = F_loss[targets != self.ignored_label_idx]

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLossMultiTemplate(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True):
        super(FocalLossMultiTemplate, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.focal_loss = FocalLoss(alpha, gamma, logits, reduce=False)

    def forward(self, inputs, targets):

        loss = self.focal_loss(inputs, targets).sum(-1).mean(-1)

        return loss


class FocalLossFGBGNormalization(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, logits=True, fgbgnorm=True):
        super(FocalLossFGBGNormalization, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_loss = FocalLoss(alpha, gamma, logits, reduce=False)



    def forward(self, inputs, targets, reduce=True):
        loss = self.focal_loss(inputs, targets)
        
        loss = loss.sum(-1)
        loss /= (len(torch.nonzero(targets)) + 1)

        return loss.mean(-1)



class WrappedBCELoss(nn.Module):
    def __init__(self):
        super(WrappedBCELoss, self).__init__()
        self.loss = F.binary_cross_entropy_with_logits

    def forward(self, inputs, targets, reduce=True):
        return self.loss(inputs, targets)

def loss_eval_bincls_single_level(pre_cls_logits, rel_labels, loss):

    bin_logits = pre_cls_logits[rel_labels != -1]

    selected_labels = rel_labels[rel_labels != -1].long()

    onehot = torch.zeros_like(bin_logits)
    onehot[selected_labels > 0] = 1
    loss_val = loss[0](inputs=bin_logits, targets=onehot, reduce=True)

    return loss_val


def loss_eval_mulcls_single_level(pre_cls_logits, rel_labels, loss):

    selected_cls_logits = pre_cls_logits[rel_labels != -1]

    selected_labels = rel_labels[rel_labels != -1].long()
    onehot = torch.zeros_like(selected_cls_logits)

    if len(onehot.shape) > 1:
        selected_fg_idx = squeeze_tensor(torch.nonzero(selected_labels > 0))
        onehot[selected_fg_idx, selected_labels[selected_fg_idx] - 1] = 1
    else:
        onehot[selected_labels > 0] = 1

    onehot = onehot.view(-1)
    selected_cls_logits = selected_cls_logits.view(-1)

    loss_val = loss(inputs=selected_cls_logits, targets=onehot)

    return loss_val

def loss_eval_hybrid_level(pre_cls_logits, rel_labels, loss):
    selected_cls_logits = pre_cls_logits[rel_labels != -1]

    mulitlabel_logits = selected_cls_logits[:, :-1]
    bin_logits = selected_cls_logits[:, -1]

    selected_labels = rel_labels[rel_labels != -1].long()

    onehot = torch.zeros_like(mulitlabel_logits)
    selected_fg_idx = squeeze_tensor(torch.nonzero(selected_labels > 0))
    onehot[selected_fg_idx, selected_labels[selected_fg_idx] - 1] = 1

    loss_val_mulabel = loss[0](inputs=mulitlabel_logits, targets=onehot, reduce=True)

    onehot = torch.zeros_like(bin_logits)
    onehot[selected_labels > 0] = 1
    loss_val_bin = loss[1](inputs=bin_logits, targets=onehot, reduce=True)

    # return loss_val_bin
    return loss_val_bin * 0.8 + loss_val_mulabel * 0.2


class RelAwareLoss(nn.Module):
    def __init__(self, cfg):
        super(RelAwareLoss, self).__init__()
        alpha = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.FOCAL_LOSS_ALPHA
        gamma = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.FOCAL_LOSS_GAMMA

        self.loss_module = (
            FocalLossFGBGNormalization(alpha, gamma, fgbgnorm=True),
            FocalLossFGBGNormalization(alpha, gamma, fgbgnorm=True),
        )

    def forward(self, pred_logit, rel_labels):
        return loss_eval_hybrid_level(pred_logit, rel_labels, self.loss_module)



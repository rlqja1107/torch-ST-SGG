# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.config import cfg

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        loss_type="CrossEntropy",
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()

        self.criterion_loss = nn.CrossEntropyLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, gt_or_pseudo_list):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """

        refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)
        if gt_or_pseudo_list is not None:
                loss_pseudo = F.cross_entropy(relation_logits[gt_or_pseudo_list == 1], rel_labels[gt_or_pseudo_list == 1], reduction='sum') * (cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.BETA)
                loss_bg = F.cross_entropy(relation_logits[(gt_or_pseudo_list == 0) & (rel_labels == 0)], rel_labels[(gt_or_pseudo_list == 0) & (rel_labels == 0)], reduction='sum')
                loss_gt_rel = F.cross_entropy(relation_logits[(gt_or_pseudo_list == 0) & (rel_labels > 0)], rel_labels[(gt_or_pseudo_list == 0) & (rel_labels > 0)], reduction='sum')
                n = (rel_labels >= 0).sum()
                loss_relation = (loss_pseudo + loss_bg + loss_gt_rel) / n
        else:
            loss_relation = self.criterion_loss(relation_logits[rel_labels != -1], rel_labels[rel_labels!=-1].long())
            
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


class WRelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        cfg,
        rel_loss='bce',
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.rel_loss = rel_loss
        self.obj_criterion_loss = nn.CrossEntropyLoss()
        print(self.rel_loss)
        if "STSGG" in cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR:
            self.st_coef = cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.BETA
        else:
            self.st_coef = 1.0
            
        if self.rel_loss == 'bce':
            self.rel_criterion_loss = nn.BCEWithLogitsLoss()
        elif self.rel_loss == 'ce':
            self.rel_criterion_loss = CEForSoftLabel()
        elif self.rel_loss == "ce_rwt":
            self.rel_criterion_loss = ReweightingCE()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, pos_weight=None, one_hot_gt_or_pseudo=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        refine_obj_logits = refine_logits
        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)
        if one_hot_gt_or_pseudo is not None:
            if self.rel_loss in ['ce_rwt', 'ce']:
                loss_relation = self.rel_criterion_loss(relation_logits, rel_labels, self.st_coef, one_hot_gt_or_pseudo)
        else:
            loss_relation = self.rel_criterion_loss(relation_logits, rel_labels)

        # self.obj_criterion_loss.to(fg_labels.device)
        loss_refine_obj = self.obj_criterion_loss(refine_obj_logits, fg_labels.long())
        return loss_relation, loss_refine_obj



class CEForSoftLabel(nn.Module):
    """
    Given a soft label, choose the class with max class as the GT.
    converting label like [0.1, 0.3, 0.6] to [0, 0, 1] and apply CrossEntropy
    """
    def __init__(self, reduction="mean"):
        super(CEForSoftLabel, self).__init__()
        self.reduction=reduction
        
    def forward(self, input, target, st_coef = 1.0, one_hot_gt_or_pseudo=None):
        final_target = torch.zeros_like(target)
        final_target[torch.arange(0, target.size(0)), target.argmax(1)] = 1.
        target = final_target
        x = F.log_softmax(input, 1)
        loss = torch.sum(- x * target, dim=1)

        if one_hot_gt_or_pseudo is None:
            loss = torch.sum(- x * target, dim=1)
        else:
            gt_label = target.max(1).indices
            gt_pred_bool = (one_hot_gt_or_pseudo == 0)&(gt_label > 0)
            pseudo_pred_bool = (one_hot_gt_or_pseudo == 1)&(gt_label > 0)
            bg_bool = (one_hot_gt_or_pseudo == 0)&(gt_label == 0)
            
            gt_label_loss = torch.sum(-x[gt_pred_bool] * target[gt_pred_bool], dim=1)
            pseudo_label_loss = torch.sum(-x[pseudo_pred_bool] * target[pseudo_pred_bool], dim=1) * st_coef
            bg_loss = torch.sum(-x[bg_bool] * target[bg_bool], dim=1)
            
            loss = torch.cat([gt_label_loss, pseudo_label_loss, bg_loss])
            
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class ReweightingCE(nn.Module):
    """
    Given a soft label, choose the class with max class as the GT.
    converting label like [0.1, 0.3, 0.6] to [0, 0, 1] and apply CrossEntropy
    """
    def __init__(self, reduction="mean"):
        super(ReweightingCE, self).__init__()
        self.reduction=reduction
        self.pseudo_loss_type = "ce"

    def forward(self, input, target, st_coef = 1.0, one_hot_gt_or_pseudo=None):
        """
        Args:
            input: the prediction
            target: [N, N_classes]. For each slice [weight, 0, 0, 1, 0, ...]
                we need to extract weight.
        Returns:

        """
        final_target = torch.zeros_like(target)
        final_target[torch.arange(0, target.size(0)), target.argmax(1)] = 1.
        idxs = torch.nonzero(target[:, 0] != 1).squeeze()
        weights = torch.ones_like(target[:, 0])
        weights[idxs] = -target[:, 0][idxs]
        target = final_target
        x = F.log_softmax(input, 1)
        if one_hot_gt_or_pseudo is None:
            loss = torch.sum(- x * target, dim=1)*weights
        elif self.pseudo_loss_type == 'ce':
            gt_label = target.max(1).indices
            gt_pred_bool = (one_hot_gt_or_pseudo == 0)&(gt_label > 0)
            pseudo_pred_bool = (one_hot_gt_or_pseudo == 1)&(gt_label > 0)
            bg_bool = (one_hot_gt_or_pseudo == 0)&(gt_label == 0)
            
            gt_label_loss = torch.sum(-x[gt_pred_bool] * target[gt_pred_bool], dim=1) * weights[gt_pred_bool]
            pseudo_label_loss = torch.sum(-x[pseudo_pred_bool] * target[pseudo_pred_bool], dim=1) * st_coef
            bg_loss = torch.sum(-x[bg_bool] * target[bg_bool], dim=1)
            
            loss = torch.cat([gt_label_loss, pseudo_label_loss, bg_loss])
        else:
            raise ValueError('Unrecognized loss function')

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def make_weaksup_relation_loss_evaluator(cfg):
    loss_evaluator = WRelationLossComputation(
        cfg,
        cfg.WSUPERVISE.LOSS_TYPE,
    )

    return loss_evaluator


def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )

    return loss_evaluator

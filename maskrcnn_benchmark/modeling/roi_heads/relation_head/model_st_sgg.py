
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import pickle
import os

class ST_SGG(nn.Module):
    def __init__(self, 
                 cfg
                 ):
        super(ST_SGG, self).__init__()
        self.cfg = cfg
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.alpha = cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_INC
        self.alpha_decay = cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_DEC
        self.save_period = cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.SAVE_CUMULATIVE_PSEUDO_LABEL_INFO_PERIOD
        self.use_gsl_output = cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.USE_GSL_OUTPUT
        
        # statistics: information of N predicate class
        statistics = torch.load("initial_data/obj_pred_info/obj_pred_info_1800") if self.num_rel_cls > 200 else torch.load("initial_data/obj_pred_info/obj_pred_info_50")
        fg_matrix = statistics['fg_matrix']
        pred_count = fg_matrix.sum(0).sum(0)[1:]
        pred_count = torch.hstack([torch.LongTensor([0]), pred_count])
        self.pred_count =pred_count.cuda()
        num_max = pred_count.max()
        
        # Calculate the lambda^{inc}
        temp = (1/(np.exp((np.log(1.0)/self.alpha))))*num_max - num_max
        self.lambda_inc = (pred_count / (num_max+temp)) ** (self.alpha)
        
        # Calculate the lambda^{dec}
        self.lambda_dec = torch.zeros(len(self.lambda_inc)).cuda()
        val, ind = torch.sort(self.lambda_inc)
        temp = (1/(np.exp((np.log(1.0)/self.alpha_decay))))*num_max - num_max
        temp_lambda_inc = (pred_count / (num_max+temp)) ** (self.alpha_decay)
        for i, (_, idx) in enumerate(zip(val, ind)):
            if idx == 0: continue
            self.lambda_dec[idx] = temp_lambda_inc[ind[len(temp_lambda_inc)-i]]
                                       
        
        self.pred_threshold = (torch.zeros(self.num_rel_cls) + 1e-6).cuda()
        
        # For saving the threshold info
        self.n_pseudo_info = {}
        self.n_cumulative_class = torch.zeros(self.num_rel_cls, dtype=torch.int).cuda()
        self.forward_time = 0
        self.pseudo_label_info = {i: [] for i in np.arange(self.num_rel_cls)}
        self.batch_decrease_conf_dict = {i:[] for i in range(self.num_rel_cls)}
        self.batch_increase_conf_dict = {i:[] for i in range(self.num_rel_cls)}
        
    def box_filter(self, boxes):
        overlaps = self.box_overlaps(boxes, boxes) > 0
        overlaps.fill_diagonal_(0)
        return torch.stack(torch.where(overlaps)).T
    
    def box_overlaps(self, box1, box2):
        num_box1 = box1.shape[0]
        num_box2 = box2.shape[0]
        lt = torch.maximum(box1.reshape([num_box1, 1, -1])[:,:,:2], box2.reshape([1, num_box2, -1])[:,:,:2])
        rb = torch.minimum(box1.reshape([num_box1, 1, -1])[:,:,2:], box2.reshape([1, num_box2, -1])[:,:,2:])
        wh = (rb - lt).clip(min=0)
        inter = wh[:,:,0] * wh[:,:,1]
        return inter
    
    
    def update_class_threshold(self, rel_pesudo_labels, one_hot_gt_or_pseudo):
        """
        Adaptive Thresholding: EMA
        """
        concat_pseudo_labels = torch.cat(rel_pesudo_labels)
        pseudo_label_set = torch.unique(concat_pseudo_labels[torch.nonzero(one_hot_gt_or_pseudo)])
        for p in np.arange(self.num_rel_cls):
            if p == 0: continue
            # Descent
            if p not in pseudo_label_set:
                if len(self.batch_decrease_conf_dict[p]) > 0:
                    decay_pred_conf = np.mean(self.batch_decrease_conf_dict[p])
                    self.pred_threshold[p] = self.pred_threshold[p] * (1-self.lambda_dec[p]) + decay_pred_conf * self.lambda_dec[p]
            # Ascent
            else:
                if len(self.batch_increase_conf_dict[p]) > 0:
                    mean_conf = np.mean(self.batch_increase_conf_dict[p])
                    self.pred_threshold[p] = (1-self.lambda_inc[p])* self.pred_threshold[p] + self.lambda_inc[p] * mean_conf
                    
        # Clear the list        
        for i in range(self.num_rel_cls):
            self.batch_increase_conf_dict[i].clear()
            self.batch_decrease_conf_dict[i].clear()
        
        
    def save_threshold_info(self):
        """
        Save the pseudo-label info: threshold, class
        """
        if self.save_period > 0:
            self.n_pseudo_info[self.forward_time] = {}
            self.n_pseudo_info[self.forward_time]['n_class'] = np.array(self.n_cumulative_class.cpu(), dtype=np.int32)
            self.n_pseudo_info[self.forward_time]['threshold'] = np.array(self.pred_threshold.cpu(), dtype=np.float16)

            if self.forward_time % self.save_period == 0:
                previous_path = f"{self.cfg.OUTPUT_DIR}/pseudo_info_{self.forward_time-self.save_period}.pkl"
                if os.path.isfile(previous_path):
                    os.remove(previous_path)
                    
                with open(f"{self.cfg.OUTPUT_DIR}/pseudo_info_{self.forward_time}.pkl", 'wb') as f:
                    pickle.dump(self.n_pseudo_info, f)
        self.forward_time += 1
    
    
    def forward(self, rel_pair_idxs, inst_proposals, rel_labels, pred_rel_logits, gsl_outputs=None):
        rel_pseudo_labels = []
        n_class = torch.zeros(self.num_rel_cls, dtype=torch.float)
        gt_or_pseudo_list = [] # 1: assign pseudo-label, 0: No assign
        
        for i, (rel_pair, pred_rel_logit) in enumerate(zip(rel_pair_idxs, pred_rel_logits)):
            rel_pair = rel_pair.long()
            n_annotate_label = torch.nonzero(rel_labels[i]).shape[0]
            pred_rel_logit = F.softmax(pred_rel_logit, -1).detach()
            
            # Filter the non-overlapped pair
            overlap_idx = self.box_filter(inst_proposals[i].bbox)
            overlap_idx = ((overlap_idx.T[...,None][...,None] == rel_pair[None,...][None,...]).sum(0).sum(-1) == 2).any(0)[n_annotate_label:]

            rel_confidence, pred_rel_class = pred_rel_logit[:,1:].max(1)
            pred_rel_class += 1
            rel_confidence_threshold = self.pred_threshold[pred_rel_class]
            
            if gsl_outputs is not None and self.use_gsl_output:
                # use graph strcuture learner to give the confident pseudo-labels
                gsl_output = gsl_outputs[i].detach()
                gsl_output = gsl_output[rel_pair[:,0], rel_pair[:,1]][n_annotate_label:]
                valid_pseudo_label_idx = (rel_confidence >= rel_confidence_threshold)[n_annotate_label:]
                valid_pseudo_label_idx = valid_pseudo_label_idx & (gsl_output == 1)
                
                no_valid_pseudo_label_idx = (rel_confidence < rel_confidence_threshold)[n_annotate_label:]
                no_valid_pseudo_label_idx = no_valid_pseudo_label_idx & (gsl_output == 1)
            else:
                valid_pseudo_label_idx = (rel_confidence >= rel_confidence_threshold)[n_annotate_label:]
                no_valid_pseudo_label_idx = (rel_confidence < rel_confidence_threshold)[n_annotate_label:]
            
            
            # Filter the non-overlap
            valid_pseudo_label_idx = valid_pseudo_label_idx & overlap_idx
            no_valid_pseudo_label_idx = no_valid_pseudo_label_idx & overlap_idx
                    
            
            # For pseudo-labeling and increasing the threshold
            max_class_thres = torch.zeros(self.num_rel_cls)
            valid_rel_confidence = rel_confidence[n_annotate_label:][valid_pseudo_label_idx]
            valid_rel_confidence, sort_ind = torch.sort(valid_rel_confidence, descending=True)
            valid_pseudo_label = pred_rel_class[n_annotate_label:][valid_pseudo_label_idx][sort_ind]
            relative_idx = torch.nonzero(valid_pseudo_label_idx).view(-1)[sort_ind]
            
            for p, c, rel_idx in zip(valid_pseudo_label, valid_rel_confidence, relative_idx):
                    if (self.pred_threshold[p] <= c).item():
                        # Constraint to the number of pseudo-label per image for preventing the confirmation bias
                        max_class_thres[p.item()] += 1
                        self.batch_increase_conf_dict[p.item()].append(c.item())
                        
                        if max_class_thres[p.item()] > 3:
                            valid_pseudo_label_idx[rel_idx] = False
                            continue
                        n_class[p] += 1
                    else:
                        valid_pseudo_label_idx[rel_idx] = False 

            # For decaying the threshold  
            no_valid_pseudo_label = pred_rel_class[n_annotate_label:][no_valid_pseudo_label_idx]
            no_valid_confidence = rel_confidence[n_annotate_label:][no_valid_pseudo_label_idx]
            
            for p, c in zip(no_valid_pseudo_label, no_valid_confidence):
                self.batch_decrease_conf_dict[p.item()].append(c.item())
                
            rel_pseudo_label = deepcopy(rel_labels[i].clone())
            rel_pseudo_label[n_annotate_label:][valid_pseudo_label_idx] = pred_rel_class[n_annotate_label:][valid_pseudo_label_idx]
            rel_pseudo_labels.append(rel_pseudo_label) 

            gt_or_pseudo = torch.zeros((len(rel_pair)), dtype = torch.long)
            gt_or_pseudo[n_annotate_label:][valid_pseudo_label_idx] = 1
            gt_or_pseudo_list.append(gt_or_pseudo)
            
        if len(rel_pseudo_labels) == 0:
            rel_pseudo_labels = None
            
        for i in range(self.num_rel_cls):
            if i == 0 or n_class[i].item() == 0: continue
            self.n_cumulative_class[i] += int(n_class[i].item())

        return rel_pseudo_labels, torch.cat(gt_or_pseudo_list).cuda()
  
        
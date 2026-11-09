
import torch

class DROLoss:
    # def __init__(self, criterion, is_robust, dataset, alpha=None, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False):
    def __init__(self, criterion, is_robust, dataset, alpha=0.1, gamma=0.1, adj=None, min_var_weight=0.05, step_size=0.01, normalize_loss=True, btl=False):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = dataset.n_groups
        # self.group_counts = dataset.group_counts().cuda()
        self.group_counts = torch.tensor(dataset.group_counts).cuda()
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_str = dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        # import pdb; pdb.set_trace()
        per_sample_losses = self.criterion(yhat, y)

        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        # import pdb; pdb.set_trace()
        if yhat.size(-1)==1:
            pred = (yhat >= 0).float()  # logit 기준
            acc_tensor = (pred == y).float()
            group_acc, group_count = self.compute_group_avg(acc_tensor, group_idx)
            # print(group_acc)

        else:
            group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            # import pdb; pdb.set_trace()
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
             actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        # return group_loss.mean()
        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())
        
        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        weights = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        # import pdb; pdb.set_trace()
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        # import pdb; pdb.set_trace()
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()




# from __future__ import print_function
import torch.nn as nn

import math

class FairSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(FairSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    
    def forward(self, features, labels, sensitive_labels,group_norm, method, epoch, mask=None, device=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: target classes of shape [bsz].
            sensitive_labels: sensitive attributes of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if device == None:
            device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            sensitive_labels = sensitive_labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            sensitive_mask = torch.eq(sensitive_labels, sensitive_labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        sensitive_mask = sensitive_mask.repeat(anchor_count, contrast_count)
        n_sensitive_mask=(~sensitive_mask.bool()).float()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # import pdb; pdb.set_trace()

        # compute log_prob
        if method=="FSCL":
            mask = mask * logits_mask
            logits_mask_fair=logits_mask*(~mask.bool()).float()*sensitive_mask
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))

        elif method=="SupCon":
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        elif method=="FSCL*":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
                
            logits_mask_fair=logits_mask*sensitive_mask
            logits_mask_fair = logits_mask_fair.to(device)
            exp_logits_fair = torch.exp(logits) * logits_mask_fair
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            log_prob = logits - torch.log(exp_logits_sum+((exp_logits_sum==0)*1))

        elif method=="SimCLR":
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask=mask*logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

              
        # compute mean of log-likelihood over positive
        #apply group normalization
        if group_norm==1:
            mean_log_prob_pos = ((mask*log_prob)/((mask*sensitive_mask).sum(1))).sum(1)
           
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        #apply group normalization
        if group_norm==1:
            C=loss.size(0)/8
            norm=(1/(((mask*sensitive_mask).sum(1)+1).float()))
            loss=(loss*norm)*C
            
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

from geomloss import SamplesLoss

class Fair_Loss_Scaler(nn.Module):
    def __init__(self, level='individual', fair_scaling_group_weights=None, 
                    fair_scaling_temperature=1., fair_scaling_coef=.5, sinkhorn_blur=0.1, right_asymptote=2.):
        super().__init__()
        self.level = level
        self.fair_scaling_temperature = fair_scaling_temperature
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=sinkhorn_blur)
        self.fair_scaling_coef = fair_scaling_coef
        #self.fair_scaling_coef = nn.Parameter(torch.tensor(fair_scaling_coef))
        self.right_asymptote = right_asymptote

    def forward(self, x, smp_xs, attr):

        individual_weights = torch.softmax(x/self.fair_scaling_temperature, dim=0) * self.right_asymptote # * x.shape[0]


        ttl_smp_x = torch.cat(smp_xs)
        distances_distributions = torch.zeros(len(smp_xs)).type_as(x)
        for i, smp_x in enumerate(smp_xs):
            # import pdb; pdb.set_trace()
            distances_distributions[i] = self.sinkhorn_loss(smp_x[:, None], ttl_smp_x[:, None])
        # import pdb; pdb.set_trace()
        
        group_weights = torch.softmax(distances_distributions[attr.type(torch.long)]/self.fair_scaling_temperature, dim=0) * self.right_asymptote # * x.shape[0]

        loss = (((1-self.fair_scaling_coef)*individual_weights + self.fair_scaling_coef*group_weights )* x).mean()
         
        return loss



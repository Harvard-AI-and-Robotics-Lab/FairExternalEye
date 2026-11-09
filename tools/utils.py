import os
import os.path as osp
import random

import PIL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from data_loaders import cnn, external_eye
import tools.imutils as imutils
from timm.data import create_transform


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

class PolyOptimizer_adam(torch.optim.Adam):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class PolyOptimizer_cls(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                if i == 4:
                    self.param_groups[i]['lr'] = self.__initial_lr[i]
                else:
                    self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

def make_path(args):

    exp_path = osp.join('./experiments', args.name)
    ckpt_path = osp.join(exp_path, 'ckpt')
    train_path = osp.join(exp_path, 'train')
    val_path = osp.join(exp_path, 'val')
    infer_path = osp.join(exp_path, 'infer')
    dict_path = osp.join(exp_path, 'dict')
    crf_path = osp.join(exp_path, 'crf')
        
    for path in [exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path]:
        os.makedirs(path, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for alpha in args.alphas:
        crf_alpha_path = osp.join(crf_path, str(alpha).zfill(2))
        if not os.path.exists(crf_alpha_path):
            os.makedirs(crf_alpha_path)

    return exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path


def make_path_with_log(args):

    exp_path = osp.join('./experiments', args.name)
    ckpt_path = osp.join(exp_path, 'ckpt')
    train_path = osp.join(exp_path, 'train')
    val_path = osp.join(exp_path, 'val')
    infer_path = osp.join(exp_path, 'infer')
    dict_path = osp.join(exp_path, 'dict')
    crf_path = osp.join(exp_path, 'crf')
    log_path = osp.join(exp_path, 'log.txt')
        
    for path in [exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path]:
        os.makedirs(path, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for alpha in args.alphas:
        crf_alpha_path = osp.join(crf_path, str(alpha).zfill(2))
        if not os.path.exists(crf_alpha_path):
            os.makedirs(crf_alpha_path)

    return exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, log_path


def build_dataset_moco(args, phase='train'):
    resize = args.resize
    color_jitter = args.cj

    if phase == 'train':
        dataset = cnn.ExternalEyeCNNDataset(
            args, resize=resize, color_jitter=color_jitter, train=True
        )

    elif phase == 'val':
        dataset = cnn.ExternalEyeCNNDataset(
            args, resize=resize, color_jitter=color_jitter, train=False
        )

    return dataset




def build_dataset_trm(is_train, args, gen_attn=False):
    dataset = None
    nb_classes = None
    

    if args.data_set == 'exEYE':
        if gen_attn:
            # args.gen_attention_maps = False
            args.gen_attention_maps = True
            args.aa = None
            args.color_jitter = None
            
            transform = build_transform(is_train, args)
            # transform = None
            dataset = external_eye.ExternalEyeDataset(args, train=is_train, gen_attn=gen_attn, transform=transform)
        else: 
            args.gen_attention_maps = False

            args.aa = None
            args.color_jitter = None
            args.center_crop = True
            transform = build_transform(is_train, args)

            args.aa = 'rand-m9-mstd0.5-inc1'
            args.color_jitter = 0.4
            args.center_crop = False
            args.scale = (0.7,1.1)
            requires_two_views = is_train and args.model == "ViT_fscl"
            transform2 = build_transform(is_train, args) if requires_two_views else None
      

            dataset = external_eye.ExternalEyeDataset(
                args, train=is_train, gen_attn=gen_attn, transform=transform, transform2=transform2
            )
        
        # import pdb; pdb.set_trace()
    
        return dataset
    elif args.data_set == 'exEYE_LLaVA':
        if gen_attn:
            # args.gen_attention_maps = False
            args.gen_attention_maps = True
            args.aa = None
            args.color_jitter = None
            
            transform = transforms.Compose([transforms.CenterCrop((1024,1024)),transforms.ToTensor()])
            dataset = external_eye.ExternalEyeDataset(args, train=is_train, gen_attn=gen_attn, transform=transform)
        else: 
            args.gen_attention_maps = False

            args.aa = None
            args.color_jitter = None
            args.center_crop = True
            transform = build_transform(is_train, args)

            args.aa = 'rand-m9-mstd0.5-inc1'
            args.color_jitter = 0.4
            args.center_crop = False
            args.scale = (0.7,1.1)
            # transform2 = build_transform(is_train, args)
            transform2 = None
      

            dataset = external_eye.ExternalEyeDataset(
                args, train=is_train, gen_attn=gen_attn, transform=transform, transform2=transform2
            )
        
        # import pdb; pdb.set_trace()
    
        return dataset
    
        # nb_classes = 20


from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.functional import InterpolationMode

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train

        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=args.scale,
            ratio=(1.0,1.0)
        
        )
        if not resize_im:
            if args.center_crop:
                transform.transforms[0] = transforms.CenterCrop(args.input_size)
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            else:
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)
        return transform
    

    t = []
    if resize_im and not args.gen_attention_maps:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    #ADDED
    else:
        t.append(
            transforms.Resize(size=(512,512), interpolation=3))  # to maintain same ratio w.r.t. 224 images
        

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from sklearn.metrics import auc, confusion_matrix

# def compute_group_auc(targets, preds, infos, groups, group_name, logger, multi_class=False):
#     logger.info("#"*10)
#     logger.info(f"{group_name} Group AUC")
#     for group in groups:
#         idx = np.array(infos) == group
#         if idx.sum() > 0:
#             if multi_class:
#                 group_auc = roc_auc_score(targets[idx], preds[idx], multi_class='ovo')
#             else:
#                 group_auc = roc_auc_score(targets[idx], preds[idx])
#             logger.info(f"{group_name} {group}: {group_auc:.4f}")
#         else:
#             logger.info(f"{group_name} {group}: No samples")

from sklearn.preprocessing import label_binarize
import pdb

def compute_group_auc(targets, preds, infos, groups, group_name, logger, global_auc, multi_class=False):
    """
    Compute AUC for each demographic group, ensuring that the number of unique classes in
    targets matches the number of probability columns in preds.

    Args:
        targets: (batch_size,) - Ground truth labels (class indices: 0,1,2,...)
        preds: (batch_size, num_classes) - Model predicted probabilities per class
        infos: (batch_size,) - Group identifiers for each sample
        groups: (list) - Unique group labels
        group_name: (str) - Name of the group
        logger: Logger object
        multi_class: (bool) - Whether the classification is multi-class
    """
    group_auc_list = []
    logger.info("#" * 30)
    logger.info(f"{group_name} Group AUC")

    for group in groups:
        # import pdb; pdb.set_trace()
        idx = np.array(infos) == group  # Find indices for the group
        if idx.sum() > 0:
            try:
                # Extract targets and predictions for the group
                targets_group = targets[idx]
                preds_group = preds[idx]

                # Get unique classes in this group (convert to int)
                unique_classes = np.unique(targets_group).astype(int)

                # Skip if only one class exists in the group (AUC not defined)
                if len(unique_classes) < 2:
                    logger.info(f"{group_name} {group}: Only one class present, AUC not defined")
                    continue

                if multi_class:
                    # Apply softmax if necessary
                    # preds_group = torch.nn.functional.softmax(torch.tensor(preds_group), dim=1).numpy()

                    # Binarize targets to match available classes
                    targets_group_bin = label_binarize(targets_group, classes=np.arange(preds.shape[1]))

                    # Ensure preds_group only includes relevant classes
                    preds_group = preds_group[:, unique_classes]  # Now it uses integer indices
                    # pdb.set_trace()
                    # Compute multi-class AUC
                    group_auc = roc_auc_score(targets_group_bin[:, unique_classes], preds_group, multi_class='ovr')
                    # print("multi_class-AUC")

                else:
                    # Compute binary AUC
                    group_auc = roc_auc_score(targets_group, preds_group)

                logger.info(f"{group_name} {group}: {group_auc:.4f}")
                group_auc_list.append(np.abs(group_auc - global_auc))

            except Exception as e:
                logger.info(f"{group_name} {group}: AUC computation failed - {e}")

        else:
            logger.info(f"{group_name} {group}: No samples")
    es_auc = global_auc / (1 + sum(group_auc_list)) if group_auc_list else np.nan

    logger.info(f"ES-AUC {group_name}: {es_auc:.4f}")

    return group_auc_list

def log_sensitivity_at_specificity(targets, preds, logger):
    fpr, tpr, _ = roc_curve(targets, preds)
    # pdb.set_trace()
    specificities = [0.8, 0.85, 0.9, 0.95]
    es_auc_values = []

    for spec in specificities:
        idx_candidates = np.where(fpr <= (1 - spec))[0]  # Find indices where FPR is low enough
        if len(idx_candidates) > 0:
            idx = idx_candidates[-1]  # Select the highest available TPR
            sensitivity_at_spec = tpr[idx]
            es_auc_values.append(sensitivity_at_spec)
            logger.info(f"Sensitivity at Specificity {spec * 100:.0f}%: {sensitivity_at_spec:.4f}")
        else:
            logger.info(f"Sensitivity at Specificity {spec * 100:.0f}%: Not available (FPR too high)")
    # logger.info(f"Mean ES-AUC: {sum(es_auc_values)/len(es_auc_values)}")

def log_multiclass_sensitivity_at_specificity(targets, preds, logger, num_classes=4):
    """
    Compute ES-AUC for multi-class classification using One-Versus-Rest (OvR) method.

    Args:
        targets: (batch_size,) - Ground truth class indices (e.g., [0, 1, 2, 3])
        preds: (batch_size, num_classes) - Model predicted probabilities per class
        num_classes: (int) - Number of classes

    Returns:
        Mean ES-AUC score
    """
    specificities = [0.8, 0.85, 0.9, 0.95]
    es_auc_values = []

    for class_idx in range(num_classes):
        # Convert targets to binary for One-Versus-Rest strategy
        binary_targets = (targets == class_idx).astype(int)  # One class vs. rest (1 for class_idx, 0 otherwise)
        class_preds = preds[:, class_idx]  # Probability for this class

        fpr, tpr, _ = roc_curve(binary_targets, class_preds)

        es_auc_per_class = []
        for spec in specificities:
            idx_candidates = np.where(fpr <= (1 - spec))[0]  # Find indices where FPR is low enough
            if len(idx_candidates) > 0:
                idx = idx_candidates[-1]  # Select the highest available TPR
                sensitivity_at_spec = tpr[idx]
                es_auc_per_class.append(sensitivity_at_spec)
                logger.info(f"Sensitivity at Specificity {spec * 100:.0f}%: {sensitivity_at_spec:.4f}")
            else:
                es_auc_per_class.append(0.0)  # If no valid threshold, assume sensitivity is 0
                # logger.info(f"Sensitivity at Specificity {spec * 100:.0f}%: {sensitivity_at_spec:.4f}")
        
        # Average ES-AUC across specificities for this class
        mean_es_auc_class = np.mean(es_auc_per_class)
        # logger.info(f"Sensitivity at Specificity {spec * 100:.0f}%: {sensitivity_at_spec:.4f}")
        es_auc_values.append(mean_es_auc_class)

    # Mean ES-AUC across all classes
    # mean_es_auc = np.mean(es_auc_values)
    # logger.info(f"Mean ES-AUC: {mean_es_auc:.4f}")
    # return mean_es_auc


from torch.utils.data import Sampler
class ImprovedBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, dr_ratio=0.05):
        """
        dataset: PyTorch Dataset 객체
        batch_size: mini-batch 크기 (32)
        dr_ratio: mini-batch 내 DR(1) 포함 비율 (default: 5% = 2개 DR, 30개 non-DR)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.dr_ratio = dr_ratio

        # ✅ Dataset에서 미리 레이블 불러오기
        labels = dataset.labels  # NumPy 배열
        print(f"Total dataset size: {len(dataset)}")
        print(f"Total DR (1) samples: {np.sum(dataset.labels == 1)}")
        print(f"Total non-DR (0) samples: {np.sum(dataset.labels == 0)}")

        # ✅ DR(1)과 non-DR(0) 샘플 인덱스를 NumPy 벡터 연산으로 처리
        self.dr_indices = np.where(labels == 1)[0]  # DR 샘플 인덱스 (200개)
        self.non_dr_indices = np.where(labels == 0)[0]  # non-DR 샘플 인덱스 (9800개)

        # ✅ NumPy 기반 셔플링
        np.random.shuffle(self.dr_indices)
        np.random.shuffle(self.non_dr_indices)

        self.dr_per_batch = max(1, int(self.batch_size * self.dr_ratio))  
        self.non_dr_per_batch = self.batch_size - self.dr_per_batch  

    def __iter__(self):
        batch_sample_list = []
        dr_len = len(self.dr_indices)
        non_dr_len = len(self.non_dr_indices)

        dr_pointer = 0
        non_dr_pointer = 0  

        while non_dr_pointer + self.non_dr_per_batch <= non_dr_len:
            batch = []

            selected_dr = self.dr_indices[dr_pointer: dr_pointer + self.dr_per_batch].tolist()
            selected_non_dr = self.non_dr_indices[non_dr_pointer: non_dr_pointer + self.non_dr_per_batch].tolist()

            if len(selected_dr) < self.dr_per_batch:
                selected_dr += np.random.choice(self.dr_indices, self.dr_per_batch - len(selected_dr)).tolist()

            batch.extend(selected_dr)
            batch.extend(selected_non_dr)
            np.random.shuffle(batch)
            batch_sample_list.append(batch)

            dr_pointer = (dr_pointer + self.dr_per_batch) % dr_len
            non_dr_pointer = (non_dr_pointer + self.non_dr_per_batch) % non_dr_len  

        np.random.shuffle(batch_sample_list)
        # print(f"DEBUG: batch_list type before return: {type(batch_sample_list)}, length: {len(batch_sample_list)}")
        return iter(batch_sample_list)

    def __len__(self):
        batch_count = len(self.non_dr_indices) // self.non_dr_per_batch
        # print(f"DEBUG: batch_count type: {type(batch_count)}, value: {batch_count}")  
        return int(batch_count)  
    


def endless_loader(dataloader):
    while True:
        for data in dataloader:
            yield data
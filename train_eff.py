from __future__ import division
from __future__ import print_function

# Base
import os
import os.path as osp
from tqdm import tqdm
import random
import importlib
import argparse
import logging
import pdb

from matplotlib import pyplot as plt

# DL
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

# Custom
import tools.imutils as imutils
import tools.utils as utils
import tools.pyutils as pyutils

from tools.utils import log_sensitivity_at_specificity, compute_group_auc, ImprovedBalancedBatchSampler, log_multiclass_sensitivity_at_specificity

# from evaluation import eval_in_script

from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from sklearn.metrics import auc, confusion_matrix
import torch
import numpy as np
import random

# CUDA_VISIBLE_DEVICES = 0 python train_eff.py --name eff_[task]_[DRO-group] --task task --group_type 0-3

if __name__ == '__main__':

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--num_workers", default=8, type=int)
    # parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--batch_size", default=32, type=int) #best setting

    # Augmentation
    
    parser.add_argument("--resize", default=484, type=float)
    parser.add_argument("--cj", default=[0.4, 0.4, 0.4, 0.1], nargs='+', type=float)
    parser.add_argument("--CRF", default=4, type=int)

    # Learning rate
    parser.add_argument("--lr", default=0.001, type=float) #DR/AMD/Glaucoma
    # parser.add_argument("--lr", default=1e-4, type=float) #DRO loss?
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--max_epochs", default=20, type=int)

    # Experiments
    parser.add_argument("--model", default='EfficientNet', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--seed", default=5137, type=int)
    parser.add_argument("--task", required=True, choices=["amd", "dr", "glaucoma"])
    parser.add_argument("--group_type", type=int, choices=range(4), default=None) #Sex, race, ethnic, language

    # Output
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dict", action='store_false')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--out_num", default=100, type=int)
    parser.add_argument("--alphas", default=[6, 10, 24], nargs='+', type=int)

    args = parser.parse_args()
    pyutils.same_seeds(args.seed)

    exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, log_path = utils.make_path_with_log(args)
    
    # if args.task in ['dr','glaucoma']:
    #     args.C = 1
    # elif args.task == 'amd':
    #     args.C = 4
    args.C = 1



    # Logger
    if osp.isfile(log_path):
        os.remove(log_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    logger.info('-'*52 + ' SETUP ' + '-'*52)
    for arg in vars(args):
        logger.info(arg + ' : ' + str(getattr(args, arg)))
    logger.info('-'*111)
    
    logger.info('Start experiment ' + args.name + '!')
    
    train_dataset = utils.build_dataset_moco(args, phase='train')
    val_dataset = utils.build_dataset_moco(args, phase='val')

    # WeightedRandomSampler 생성
    # batch_sampler = ImprovedBalancedBatchSampler(train_dataset,batch_size=args.batch_size,dr_ratio=0.05)
    # train_data_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,num_workers=args.num_workers,pin_memory=True)

    sampler = torch.utils.data.RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, drop_last=True, sampler=sampler, num_workers=args.num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    logger.info('Train dataset is loaded from data/splits/train_final.npy')
    logger.info('Validation dataset is loaded from data/splits/val_final.npy')

    train_num_img = len(train_dataset)
    train_num_batch = len(train_data_loader)

    max_step = (train_num_img // args.batch_size * args.max_epochs)
    args.max_step = max_step

    model = getattr(importlib.import_module('models.model_'+args.model), 'model_WSSS')(args, logger)
    model.train_setup()

    logger.info('-' * 111)
    logger.info(('-' * 49) + ' Start Train ' + ('-' * 49))

    Auc_list = []
    max_auc = 0

    for epo in range(args.max_epochs):
        epo_str = str(epo).zfill(3)

        # Train
        logger.info('-' * 111)
        logger.info('Epoch ' + epo_str + ' train')
        model.set_phase('train')

        # print("lr_mult:",model.opt_main.lr_mult)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
        # batch_sampler = ImprovedBalancedBatchSampler(train_dataset,batch_size=args.batch_size,dr_ratio=0.1)
        # train_data_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,num_workers=args.num_workers,pin_memory=True)
        
        train_num_batch = len(train_data_loader)

        for idx, pack in enumerate(tqdm(train_data_loader,dynamic_ncols=False)):
            model.unpack(pack)
            model.update(epo)
            if idx % args.print_freq == 0 and idx != 0:
                model.print_log(epo + 1, idx / train_num_batch)
        model.scheduler.step()

        logger.info('Epoch ' + epo_str + ' model is saved!')
        model.save_model(epo, ckpt_path)

    
        # Validation
        logger.info('-' * 111)
        logger.info('Epoch ' + epo_str + ' validation')
        model.set_phase('eval')
        preds, targets, infos = [], [], {'sex': [], 'race': [], 'ethnic': [], 'language': []}
        
        for _, pack in enumerate(tqdm(val_data_loader,dynamic_ncols=False)):
            model.unpack(pack)
            pred, target, info = model.infer_multi(epo, val_path, dict_path, crf_path, vis=False, dict=args.dict, crf=args.crf)
            preds.extend(pred.cpu())
            targets.extend(target.cpu())
            for key in infos.keys():
                infos[key].extend(info[key])
        
        preds, targets = np.array(preds), np.array(targets)
        
        if args.C ==1:
            auc_score = roc_auc_score(targets, preds)
        else:
            auc_score = roc_auc_score(targets, preds, multi_class='ovr')
        logger.info(f"AUC: {auc_score:.4f} and Best AUC is {max_auc:.4f}")
        
        if args.C ==1:
            log_sensitivity_at_specificity(targets, preds, logger)
        else:
            log_multiclass_sensitivity_at_specificity(targets, preds, logger)
        
        for group, labels in zip(['Sex', 'Race', 'Ethnic', 'Language'], [['0', '1'],
                                                                         ["Asian", "Black or African American", "White","Other or Unknown"],
                                                                         ["0", "1", "Unknown or Not Reported"],
                                                                         ["English", "Spanish","Other or Unknown"]]):
            # pdb.set_trace()
            compute_group_auc(targets, preds, infos[group.lower()], labels, group, logger, auc_score, args.C>1)
        
        if auc_score > max_auc:
            max_auc = auc_score
            logger.info("New best AUC!")

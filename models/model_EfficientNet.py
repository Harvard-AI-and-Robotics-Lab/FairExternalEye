import os, time
import os.path as osp
import argparse
import glob
import random
import pdb
from turtle import pos

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import itertools

# Image tools
import cv2
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from torchvision import transforms

from tools import utils, pyutils
from tools.imutils import save_img, denorm, _crf_with_alpha

# import resnet38d
from networks import efficientnet

from tools.losses import DROLoss



def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class model_WSSS():

    def __init__(self, args, logger):

        self.args = args
        if self.args.C == 1:
            self.categories = ['DR','non-DR']

        # Common things
        self.phase = 'train'
        self.dev = 'cuda'
        if self.args.C == 1:
            self.bce = nn.BCEWithLogitsLoss()#pos_weight=torch.tensor(5.0))
        elif self.args.C > 1:
            self.bce = nn.CrossEntropyLoss()
        self.bs = args.batch_size
        self.logger = logger

        # Model attributes
        self.net_names = ['net_main']
        self.base_names = ['cls']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]

        self.nets = []
        self.opts = []

        # Evaluation-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.num_count = 0

        self.val_wrong = 0
        self.val_right = 0

        # Define networks
        # self.net_main = efficientnet.EfficientNet.from_name('efficientnet-b5',num_classes=self.args.C)
        self.net_main = efficientnet.EfficientNet.from_pretrained('efficientnet-b5',num_classes=self.args.C)
        # self.net_main = othermodels.create_model(model_type='efficientnet_v2',in_dim=3,out_dim=args.C)
        # self.net_main.apply(self.freeze_bn)
        # Initialize networks with ImageNet pretrained weight
        # self.net_main.load_state_dict(resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params'), strict=False)
 

    # Save networks
    def save_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        torch.save(self.net_main.module.state_dict(), ckpt_path + '/' + epo_str + 'net_main.pth')

    # Load networks
    def load_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        self.net_main.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_main.pth'), strict=True)

        self.net_main = torch.nn.DataParallel(self.net_main.to(self.dev))


    # Set networks' phase (train/eval)
    def set_phase(self, phase):

        if phase == 'train':
            self.phase = 'train'
            for name in self.net_names:
                getattr(self, name).train()
            self.logger.info('Phase : train')

        else:
            self.phase = 'eval'
            for name in self.net_names:
                getattr(self, name).eval()
            self.logger.info('Phase : eval')

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args

        # param_main = self.net_main.get_parameter_groups()

        self.opt_main = optim.AdamW([
            {"params": [p for n, p in self.net_main.named_parameters() if "_fc" not in n], "lr": args.lr},  # 나머지 부분
            {"params": [p for n, p in self.net_main.named_parameters() if "_fc" in n], "lr": 10*args.lr}  # _fc 레이어
        ], weight_decay=args.wt_dec)



        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_main, T_max=args.max_epochs)


        self.logger.info('SGD-optimizer for net_main is defined.')
        self.logger.info('* Base learning rate : ' + str(args.lr))
        self.logger.info('* Weight decaying : ' + str(args.wt_dec) + ', max step : ' + str(args.max_step))

        self.net_main = torch.nn.DataParallel(self.net_main.to(self.dev))
        
        self.nets.append(self.net_main)

        #############Dataset#########
        if self.args.group_type is not None:
            dataset = np.load("data/splits/train_final.npy", allow_pickle=True)
            #sex, race, ethnic, language (2,3,4,5)
            group_id = int(self.args.group_type) # "0 sex, 1 race, 2 ethnic, 3 language"
            dataset_group = dataset[:,group_id+2]
            # groups = [['0', '1', "Unknown or Not Reported"],["Asian", "Black or African American", "White","Other or Unknown"],
                    # ["0", "1", "Unknown or Not Reported"],["English", "Spanish","Other or Unknown"]]
            
            # RACE 537/702/16730/2338
            # Ethnic 18454/1043/810
            # Language 17956/1629/722

            groups = [['0', '1'],["Asian", "Black or African American", "White","Other or Unknown"],
                    ["0", "1", "Unknown or Not Reported"],["English", "Spanish","Other or Unknown"]]
            
            
            self.mapping = {}
            for idx, group in enumerate(groups[group_id]):
                self.mapping[group] = idx
            class Dummy: pass
            self.dataset = Dummy()
            self.dataset.n_groups = len(groups[group_id])
            self.dataset.group_counts = [(dataset_group==group).sum() for group in groups[group_id]]
            self.dataset.group_str = ['Sex', 'Race', 'Ethnic', 'Language'][group_id]
            
            if self.args.C == 1:
                # self.bce_dro = DROLoss(criterion=nn.BCEWithLogitsLoss(reduction='none'),is_robust=True,dataset=self.dataset)
                self.bce_dro = DROLoss(criterion=nn.BCEWithLogitsLoss(reduction='none'),is_robust=False,dataset=self.dataset)
            else:
                self.bce_dro = DROLoss(criterion=nn.CrossEntropyLoss(reduction='none'),is_robust=False,dataset=self.dataset)


    def unpack(self, pack):

        if self.phase == 'train':
            self.img = pack[0].to(self.dev)  # B x 3 x H x W
            self.id = pack[0].to(self.dev)  # B x 3 x H x W
            self.name = pack[2]  # list of image names
            self.info = {"sex": pack[3], "race": pack[4], "ethnic": pack[5], "language":pack[6]}
            self.label = pack[-1].to(self.dev)  # B x 20

        if self.phase == 'eval':
            self.img = pack[0].to(self.dev)  # B x 3 x H x W
            self.id = pack[0].to(self.dev)  # B x 3 x H x W
            self.name = pack[2]  # list of image names
            self.info = {"sex": pack[3], "race": pack[4], "ethnic": pack[5], "language":pack[6]}
            self.label = pack[-1].to(self.dev)  # B x 20

        # self.split_label()

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo):

        # Tensor dimensions
        B = self.img.shape[0]
        H = self.img.shape[2]
        W = self.img.shape[3]

        self.B = B
        self.C = self.args.C

        self.opt_main.zero_grad()

        outputs, _ = self.net_main(self.img)
        self.cls_logits = outputs

        self.cls_pred = torch.sigmoid(self.cls_logits)
      
        self.label[self.label==255] = 0
        if self.C == 1:
            # print(self.info[self.dataset.group_str.lower()])
            if self.args.group_type is not None:
                group_idx = torch.tensor([
                    # self.mapping.get(s)  # fallback to 0 if key not found
                    self.mapping.get(s, -1)  # fallback to 0 if key not found
                    for s in self.info[self.dataset.group_str.lower()]
                ]).cuda()
                # print(group_idx)
                self.loss_cls = self.bce_dro.loss(self.cls_logits,self.label.unsqueeze(-1).float(),group_idx,is_training=True)
            else:
                self.loss_cls = 1*(
                    self.bce(self.cls_logits,self.label.unsqueeze(-1).float()))
        else: #Class > 1
            if self.args.group_type is not None:
                group_idx = torch.tensor([
                    self.mapping.get(s)  # fallback to 0 if key not found
                    # self.mapping.get(s, 0)  # fallback to 0 if key not found
                    for s in self.info[self.dataset.group_str.lower()]
                ]).cuda()
                self.loss_cls = self.bce_dro.loss(self.cls_logits,self.label.long(),group_idx,is_training=True)
            
            else:
                self.loss_cls = 1*(
                    self.bce(self.cls_logits,self.label.long())
                    )
        
        # self.loss_dro = DROLoss(self.bce,)
        loss = self.loss_cls 

        positive_preds = self.cls_pred[self.label == 1]
        # print(f"양성 샘플 예측 평균값: {(positive_preds).mean():.4f} and {(self.label==1).sum()} DR images exist")  # 너무 낮다면 문제

        # pdb.set_trace()
        # print(self.label.sum())
        
        loss.backward()
        self.opt_main.step()
              
       
        ################################################### Export ###################################################


        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()
        self.count += 1

        self.count_rw(self.label, self.cls_logits, 0)
    

    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict
    def infer_multi(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False):

        if self.phase != 'eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        with torch.no_grad():

            # self.img = F.interpolate(self.img,size=(512,512),mode='bilinear',align_corners=True)

            logits, _ = self.net_main.module.forward(self.img.cuda())

            if logits.size(-1)>1:
                pred = F.softmax(logits,dim=-1)
            else:
                pred = torch.sigmoid(logits)
        
        self.label[self.label==255]=0
        
        return pred, self.label, self.info

    # Print loss/accuracy (and re-initialize them)
    def print_log(self, epo, iter):

        loss_str = ''
        acc_str = ''

        for i in range(len(self.loss_names)):
            loss_str += self.loss_names[i] + ' : ' + str(round(self.running_loss[i] / self.count, 5)) + ', '

        for i in range(len(self.acc_names)):
            if self.right_count[i] != 0:
                acc = 100 * self.right_count[i] / (self.right_count[i] + self.wrong_count[i])
                acc_str += self.acc_names[i] + ' : ' + str(round(acc, 2)) + ', '
                self.accs[i] = acc

        plt.imshow(denorm(self.img[0]).permute(1,2,0).cpu().detach().numpy())
        plt.savefig("outputs/image_eff.png")

        self.logger.info(loss_str[:-2])
        self.logger.info(acc_str[:-2])

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.count = 0

    def count_rw(self, label, out, idx):
        if self.args.C == 1:
            prob = F.sigmoid(out)
            pred = (prob>=0.5).float()
            
            # pdb.set_trace()
            self.right_count[idx] += (pred==label).sum().float().item()

            self.wrong_count[idx] += (pred!=label).sum().float().item()
        
        else:
            for b in range(self.bs):  # 8
                gt = label[b].cpu().detach().numpy()
                gt_cls = np.nonzero(gt)[0]
                num = len(np.nonzero(gt)[0])
                pred = out[b].cpu().detach().numpy()
                pred_cls = pred.argsort()[-num:][::-1]

                for c in gt_cls:
                    if c in pred_cls:
                        self.right_count[idx] += 1
                    else:
                        self.wrong_count[idx] += 1
 
        

    # Max_norm
    def max_norm(self, cam_cp):
        N, C, H, W = cam_cp.size()
        cam_cp = F.relu(cam_cp)
        max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam_cp

    def cam_l1(self, cam1, cam2):
        return torch.mean(torch.abs(cam2.detach() - cam1))



class FocalSigmoidLoss(torch.nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):
        super(FocalSigmoidLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Compute sigmoid activation
        probs = torch.sigmoid(logits)
        
        # Compute focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)  # True class probabilities
        focal_weight = (1 - p_t) ** self.gamma

        # Apply weighting factor α for class imbalance
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_weight * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

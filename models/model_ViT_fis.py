from cProfile import label
from http.client import NON_AUTHORITATIVE_INFORMATION
import os, time
from statistics import mode
import os.path as osp
import argparse
import glob
import random
import pdb
from turtle import pos

import numpy as np
from numpy.core.fromnumeric import size
# from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

# Image tools
import cv2
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from torchvision import transforms
import torchvision

from tools import utils, pyutils, trmutils
from tools.imutils import save_img, denorm, _crf_with_alpha, cam_on_image
# import tools.visualizer as visualizer
from networks import mctformer

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from tools.losses import DROLoss, Fair_Loss_Scaler


import sys
sys.path.append("..") 

def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


##########
# CUDA_VISIBLE_DEVICES=0 python train_trm.py --name vit_baseline --model ViT --task dr
##########

class model_WSSS():

    def __init__(self, args, logger):

        self.args = args

        if self.args.C == 20:
            self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        elif self.args.C == 1:
            self.categories = ['DR','non-DR']
       
        # Common things
        self.phase = 'train'
        self.dev = 'cuda'
        if self.args.C == 1:
            self.bce = nn.BCEWithLogitsLoss(reduction='none')#pos_weight=torch.tensor(5.0))
        elif self.args.C > 1:
            self.bce = nn.CrossEntropyLoss()

        self.bce_fis = Fair_Loss_Scaler()


        self.bs = args.batch_size
        self.logger = logger
        self.writer = args.writer

        # Model attributes
        self.net_names = ['net_trm']
        self.base_names = ['cls','fis']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]

        self.ebd_memory_t = []
        self.ebd_memory_s = []
        self.ctk_global = [[] for _ in range(self.args.C)]

        # self.is_empty_memory = [True for i in range(len(self.categories))]

        self.nets = []
        self.opts = []

        # Evaluation-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.num_count = 0
        
        #Tensorboard
        self.global_step = 0

        self.val_wrong = 0
        self.val_right = 0

        self.num_class = args.C

        groups = [['0', '1'],["Asian", "Black or African American", "White","Other or Unknown"],
                    ["0", "1", "Unknown or Not Reported"],["English", "Spanish","Other or Unknown"]]
        
        self.num_groups = len(groups[int(args.group_type)])

        # Define networks
        self.net_trm = create_model(
            'deit_small_MCTformerPlus',
            pretrained=False,
            num_classes=args.C,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            num_group = self.num_groups
        )

            
        if args.finetune:
           
            checkpoint = (
                torch.hub.load_state_dict_from_url(args.finetune, map_location="cpu", check_hash=True)
                if args.finetune.startswith("https://")
                else torch.load(args.finetune, map_location="cpu")
            )

            try:
                checkpoint_model = checkpoint['model']
            except:
                checkpoint_model = checkpoint
            state_dict = self.net_trm.state_dict()

            if 'head.bias' in state_dict.keys():
                for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
            else:
                for k in ['head.weight', 'head_dist.weight', 'head_dist.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = self.net_trm.patch_embed.num_patches
        if args.finetune.startswith('https'):
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.net_trm.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

        # if args.finetune.startswith('https') and 'MCTformer' in args.model:
        if True:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1,args.C,1)
        # else:
            # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = pos_tokens[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(pos_tokens.size(0), pos_tokens.size(1), pos_tokens.size(2)*2, pos_tokens.size(3) *2)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        # pdb.set_trace()

        checkpoint_model['pos_embed_cls'] = extra_tokens
        checkpoint_model['pos_embed_pat'] = pos_tokens

        # if args.finetune.startswith('https') and 'MCTformer' in args.model:
        if True:
            cls_token_checkpoint = checkpoint_model['cls_token']
            new_cls_token = cls_token_checkpoint.repeat(1,args.C,1)
            checkpoint_model['cls_token'] = new_cls_token

        self.net_trm.load_state_dict(checkpoint_model, strict=False)

        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.KD = nn.KLDivLoss(reduction='batchmean')
        
    # Save networks
    def save_model(self, epo, ckpt_path):
        print("Current learning rate is ",self.opt_trm.param_groups[0]['lr'])
        epo_str = str(epo).zfill(3)
        torch.save(self.net_trm.module.state_dict(), ckpt_path + '/' + epo_str + 'net_trm.pth')
    
    def save_model_coco(self, epo, iter, ckpt_path):
        epo_str = str(epo).zfill(3)
        iter_str = str(iter).zfill(5)
        torch.save(self.net_trm.module.state_dict(), ckpt_path + '/' + epo_str +"_"+ iter_str + 'net_trm.pth')

    # Load networks
    def load_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        self.net_trm.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_trm.pth'), strict=True)

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))
    
    def load_pretrained_model(self, pretrained_path):
        self.net_trm.load_state_dict(torch.load(pretrained_path), strict=True)

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))

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
        # self.net_sup.eval()

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args


        linear_scaled_lr = args.lr * args.batch_size * trmutils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

        self.opt_trm = create_optimizer(args, self.net_trm)
        self.lr_scheduler, _ = create_scheduler(args, self.opt_trm)


        self.logger.info('Poly-optimizer for trm is defined.')

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))
        self.logger.info('Networks are uploaded on multi-gpu.')

        self.nets.append(self.net_trm)

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
            

    # Unpack data pack from data_loader
    def unpack(self, pack, group_pack=None):

        if self.phase == 'train':
            self.img = pack[0].to(self.dev)  # B x 3 x H x W
            self.id = pack[0].to(self.dev)  # B x 3 x H x W
            self.name = pack[2]  # list of image names
            self.info = {"sex": pack[3], "race": pack[4], "ethnic": pack[5], "language":pack[6]}
            self.label = pack[-1].to(self.dev)  # B x 20
            for idx in range(len(group_pack)):
                setattr(self,"img_group_%d"%idx,group_pack[idx][0].to(self.dev))
                setattr(self,"label_group_%d"%idx,group_pack[idx][-1].to(self.dev))

            # self.data_group = group_pack.to(self.dev)

        if self.phase == 'eval':
            self.img = pack[0].to(self.dev)  # B x 3 x H x W
            self.id = pack[0].to(self.dev)  # B x 3 x H x W
            self.name = pack[2]  # list of image names
            self.info = {"sex": pack[3], "race": pack[4], "ethnic": pack[5], "language":pack[6]}
            self.label = pack[-1].to(self.dev)  # B x 20

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo):
        # Tensor dimensions
        B = self.img.shape[0]
        H = self.img.shape[2]
        W = self.img.shape[3]

        h,w = H//16, W//16
        C = self.num_class  # Number of cls

        self.B = B
        self.C = C

        ################################################### Update TRM ###################################################
        self.opt_trm.zero_grad()
        self.net_trm.train()
        # print(self.opt_trm.param_groups[0]['lr'])
        ######################################################################
        outputs = self.net_trm(self.img)
        self.x_cls_logits = outputs[0]
        self.x_patch_logits = outputs[2]

        if self.C == 1:
            
            group_losses = []
            with torch.no_grad():
                for idx in range(self.dataset.n_groups):
                    group_output = self.net_trm(getattr(self,"img_group_%d"%idx))
                    group_x_patch_logits = group_output[2]
                    group_loss = self.bce(group_x_patch_logits,getattr(self,"label_group_%d"%idx).unsqueeze(-1).float())
                    group_losses.append(group_loss.squeeze(-1))


            group_idx = torch.tensor([
                # self.mapping.get(s)  # fallback to 0 if key not found
                self.mapping.get(s, 0)  # fallback to 0 if key not found
                for s in self.info[self.dataset.group_str.lower()]
            ]).cuda()
            self.loss_cls = self.bce(self.x_cls_logits,self.label.unsqueeze(-1).float()).mean()
            loss_cls_patch = self.bce(self.x_patch_logits,self.label.unsqueeze(-1).float())
            # pdb.set_trace()
            # self.loss_fis = self.bce_fis(loss_cls_patch,group_losses,group_idx[self.args.group_type])
            self.loss_fis = self.bce_fis(loss_cls_patch.squeeze(-1),group_losses,group_idx)
                
        # else: #Class > 1
        #     self.loss_cls = (
        #         self.bce(self.x_cls_logits,self.label.unsqueeze(-1).float())+
        #         self.bce(self.x_patch_logits,self.label.unsqueeze(-1).float()))
        # pdb.set_trace()
            
        loss_trm = self.loss_cls + 10*self.loss_fis
        # print(self.loss_cls,self.loss_adv)

        loss_trm.backward()

        self.opt_trm.step()
        
        ################################################### Export ###################################################


        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()

        self.count += 1
        #Tensorboard
        self.global_step +=1

        self.count_rw(self.label, self.x_patch_logits, 0)     # FG get rw except fg

    # Initialization for msf-infer
    def infer_init(self,epo):
        n_gpus = torch.cuda.device_count()
        self.net_trm.eval()
        # self.net_trm_replicas = torch.nn.parallel.replicate(self.net_trm.module, list(range(n_gpus)))

    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict
    def infer_multi(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False):

        if self.phase != 'eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        
        B, _, H, W = self.img.shape
        n_gpus = torch.cuda.device_count()

        with torch.no_grad():
            # self.img = F.interpolate(self.img,size=(768,768),mode='bicubic')

            pred, cams, _ = self.net_trm.module.forward(self.img.cuda(),return_att=True,n_layers= 12)
            if pred.size(-1)>1:
                pred = F.softmax(pred,dim=-1)
            else:
                pred = torch.sigmoid(pred)

        self.label[self.label==255]=0
        # pdb.set_trace()
        return pred, self.label, self.info
 
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
        plt.savefig("outputs/image_vit.png")

        self.logger.info(loss_str[:-2])
        self.logger.info(acc_str[:-2])

        ###Tensorboard###
        for i in range(len(self.loss_names)):
            self.writer.add_scalar("Loss/%s"%self.loss_names[i],(self.running_loss[i] / self.count),self.global_step)

        # for c in self.gt_cls:
        # temp = cam_on_image(denorm(self.img[0]).cpu().detach().numpy(), self.rcams_bg[0][0].detach().cpu().numpy())
        # self.writer.add_image('000000/bg', temp, epo)
        # temp = cam_on_image(denorm(self.img[0]).cpu().detach().numpy(), self.rcams_fg[0][0].detach().cpu().numpy())
        # self.writer.add_image('000000/fg', temp, epo)

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.count = 0

    def count_rw(self, label, out, idx):
        for b in range(out.size(0)):  # 8
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



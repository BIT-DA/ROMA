import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import timm
import time
import torch.nn.functional as F
import sys
from functools import partial
import torch.nn as nn
import math

from torchvision.transforms import transforms as tfs

class ROMAModel(BaseModel):
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--adj_size_list', type=list, default=[2, 4, 6, 8, 12], help='different scales of perception field')
        parser.add_argument('--lambda_mlp', type=float, default=1.0, help='weight of lr for discriminator')
        parser.add_argument('--lambda_motion', type=float, default=1.0, help='weight for Temporal Consistency')
        parser.add_argument('--lambda_D_ViT', type=float, default=1.0, help='weight for discriminator')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_global', type=float, default=1.0, help='weight for Global Structural Consistency')
        parser.add_argument('--lambda_spatial', type=float, default=1.0, help='weight for Local Structural Consistency')
        parser.add_argument('--atten_layers', type=str, default='1,3,5', help='compute Cross-Similarity on which layers')
        parser.add_argument('--local_nums', type=int, default=256)
        parser.add_argument('--which_D_layer', type=int, default=-1)
        parser.add_argument('--side_length', type=int, default=7)

        parser.set_defaults(pool_size=0) 

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)


        self.loss_names = ['G_GAN_ViT', 'D_real_ViT', 'D_fake_ViT', 'global', 'spatial', 'motion']
        self.visual_names = ['real_A0', 'real_A1', 'fake_B0', 'fake_B1', 'real_B0', 'real_B1']
        self.atten_layers = [int(i) for i in self.opt.atten_layers.split(',')]


        if self.isTrain:
            self.model_names = ['G', 'D_ViT']
        else:  # during test time, only load G
            self.model_names = ['G']


        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)


        if self.isTrain:

            self.netD_ViT = networks.MLPDiscriminator().to(self.device)
            self.netPreViT = timm.create_model("vit_base_patch16_384",pretrained=True).to(self.device)
            

            self.norm = F.softmax

            self.resize = tfs.Resize(size=(384,384))

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for atten_layer in self.atten_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_ViT = torch.optim.Adam(self.netD_ViT.parameters(), lr=opt.lr * opt.lambda_mlp, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_ViT)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        pass
   

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD_ViT, True)
        self.optimizer_D_ViT.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D_ViT.step()

        # update G
        self.set_requires_grad(self.netD_ViT, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A0 = input['A0' if AtoB else 'B0'].to(self.device)
        self.real_A1 = input['A1' if AtoB else 'B1'].to(self.device)
        self.real_B0 = input['B0' if AtoB else 'A0'].to(self.device)
        self.real_B1 = input['B1' if AtoB else 'A1'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B0 = self.netG(self.real_A0) 
        self.fake_B1 = self.netG(self.real_A1)
        
        if self.opt.isTrain:
            real_A0 = self.real_A0
            real_A1 = self.real_A1
            real_B0 = self.real_B0
            real_B1 = self.real_B1
            fake_B0 = self.fake_B0
            fake_B1 = self.fake_B1
            self.real_A0_resize = self.resize(real_A0)
            self.real_A1_resize = self.resize(real_A1)
            real_B0 = self.resize(real_B0)
            real_B1 = self.resize(real_B1)
            self.fake_B0_resize = self.resize(fake_B0)
            self.fake_B1_resize = self.resize(fake_B1)
            self.mutil_real_A0_tokens = self.netPreViT(self.real_A0_resize, self.atten_layers, get_tokens=True)
            self.mutil_real_A1_tokens = self.netPreViT(self.real_A1_resize, self.atten_layers, get_tokens=True)
            self.mutil_real_B0_tokens = self.netPreViT(real_B0, self.atten_layers, get_tokens=True)
            self.mutil_real_B1_tokens = self.netPreViT(real_B1, self.atten_layers, get_tokens=True)
            self.mutil_fake_B0_tokens = self.netPreViT(self.fake_B0_resize, self.atten_layers, get_tokens=True)
            self.mutil_fake_B1_tokens = self.netPreViT(self.fake_B1_resize, self.atten_layers, get_tokens=True)

    def tokens_concat(self, origin_tokens, adjacent_size):
        adj_size = adjacent_size
        B, token_num, C = origin_tokens.shape[0], origin_tokens.shape[1], origin_tokens.shape[2]
        S = int(math.sqrt(token_num))
        if S * S != token_num:
            print('Error! Not a square!')
        token_map = origin_tokens.clone().reshape(B,S,S,C)
        cut_patch_list = []
        for i in range(0, S, adj_size):
            for j in range(0, S, adj_size):
                i_left = i
                i_right = i + adj_size + 1 if i + adj_size <= S else S + 1
                j_left = j
                j_right = j + adj_size if j + adj_size <= S else S + 1

                cut_patch = token_map[:, i_left:i_right, j_left: j_right, :]
                cut_patch= cut_patch.reshape(B,-1,C)
                cut_patch = torch.mean(cut_patch, dim=1, keepdim=True)
                cut_patch_list.append(cut_patch)


        result = torch.cat(cut_patch_list,dim=1)
        return result


    def cat_results(self, origin_tokens, adj_size_list):
        res_list = [origin_tokens]
        for ad_s in adj_size_list:
            cat_result = self.tokens_concat(origin_tokens, ad_s)
            res_list.append(cat_result)
        
        result = torch.cat(res_list, dim=1)

        return result



    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""


        lambda_D_ViT = self.opt.lambda_D_ViT
        fake_B0_tokens = self.mutil_fake_B0_tokens[self.opt.which_D_layer].detach()
        fake_B1_tokens = self.mutil_fake_B1_tokens[self.opt.which_D_layer].detach()

        real_B0_tokens = self.mutil_real_B0_tokens[self.opt.which_D_layer]
        real_B1_tokens = self.mutil_real_B1_tokens[self.opt.which_D_layer]


        fake_B0_tokens = self.cat_results(fake_B0_tokens, self.opt.adj_size_list)
        fake_B1_tokens = self.cat_results(fake_B1_tokens, self.opt.adj_size_list)



        real_B0_tokens = self.cat_results(real_B0_tokens, self.opt.adj_size_list)
        real_B1_tokens = self.cat_results(real_B1_tokens, self.opt.adj_size_list)

        pre_fake0_ViT = self.netD_ViT(fake_B0_tokens)
        pre_fake1_ViT = self.netD_ViT(fake_B1_tokens)

        self.loss_D_fake_ViT = (self.criterionGAN(pre_fake0_ViT, False).mean() + self.criterionGAN(pre_fake1_ViT, False).mean()) * 0.5 * lambda_D_ViT

        pred_real0_ViT = self.netD_ViT(real_B0_tokens)
        pred_real1_ViT = self.netD_ViT(real_B1_tokens)
        self.loss_D_real_ViT = (self.criterionGAN(pred_real0_ViT, True).mean() + self.criterionGAN(pred_real1_ViT, True).mean()) * 0.5 * lambda_D_ViT

        self.loss_D_ViT = (self.loss_D_fake_ViT + self.loss_D_real_ViT) * 0.5


        return self.loss_D_ViT

    def compute_G_loss(self):

        if self.opt.lambda_GAN > 0.0:

            fake_B0_tokens = self.mutil_fake_B0_tokens[self.opt.which_D_layer]
            fake_B1_tokens = self.mutil_fake_B1_tokens[self.opt.which_D_layer]
            fake_B0_tokens = self.cat_results(fake_B0_tokens, self.opt.adj_size_list)
            fake_B1_tokens = self.cat_results(fake_B1_tokens, self.opt.adj_size_list)
            pred_fake0_ViT = self.netD_ViT(fake_B0_tokens)
            pred_fake1_ViT = self.netD_ViT(fake_B1_tokens)
            self.loss_G_GAN_ViT = (self.criterionGAN(pred_fake0_ViT, True) + self.criterionGAN(pred_fake1_ViT, True)) * 0.5 * self.opt.lambda_GAN
        else:
            self.loss_G_GAN_ViT = 0.0

        if self.opt.lambda_global > 0.0 or self.opt.lambda_spatial > 0.0:
            self.loss_global, self.loss_spatial = self.calculate_attention_loss()
        else:
            self.loss_global, self.loss_spatial = 0.0, 0.0

        if self.opt.lambda_motion > 0.0:
            self.loss_motion = 0.0
            for real_A0_tokens, real_A1_tokens, fake_B0_tokens, fake_B1_tokens in zip(self.mutil_real_A0_tokens, self.mutil_real_A1_tokens, self.mutil_fake_B0_tokens, self.mutil_fake_B1_tokens):
                A0_B1 = real_A0_tokens.bmm(fake_B1_tokens.permute(0,2,1))
                B0_A1 = fake_B0_tokens.bmm(real_A1_tokens.permute(0,2,1))
                cos_dis_global =  F.cosine_similarity(A0_B1, B0_A1, dim=-1)
                self.loss_motion += self.criterionL1(torch.ones_like(cos_dis_global), cos_dis_global).mean()
        else:
            self.loss_motion = 0.0

        self.loss_G = self.loss_G_GAN_ViT + self.loss_global + self.loss_spatial + self.loss_motion
        return self.loss_G

    def calculate_attention_loss(self):
        n_layers = len(self.atten_layers)
        mutil_real_A0_tokens = self.mutil_real_A0_tokens
        mutil_real_A1_tokens = self.mutil_real_A1_tokens
        mutil_fake_B0_tokens = self.mutil_fake_B0_tokens
        mutil_fake_B1_tokens = self.mutil_fake_B1_tokens

            
        if self.opt.lambda_global > 0.0:
            loss_global = self.calculate_similarity(mutil_real_A0_tokens, mutil_fake_B0_tokens) + self.calculate_similarity(mutil_real_A1_tokens, mutil_fake_B1_tokens)
            loss_global *= 0.5

        else:
            loss_global = 0.0

        if self.opt.lambda_spatial > 0.0:
            loss_spatial = 0.0
            local_nums = self.opt.local_nums
            tokens_cnt = 576
            local_id = np.random.permutation(tokens_cnt)
            local_id = local_id[:int(min(local_nums, tokens_cnt))]

            mutil_real_A0_local_tokens = self.netPreViT(self.real_A0_resize, self.atten_layers, get_tokens=True, local_id=local_id, side_length = self.opt.side_length)
            mutil_real_A1_local_tokens = self.netPreViT(self.real_A1_resize, self.atten_layers, get_tokens=True, local_id=local_id, side_length = self.opt.side_length)

            mutil_fake_B0_local_tokens = self.netPreViT(self.fake_B0_resize, self.atten_layers, get_tokens=True, local_id=local_id, side_length = self.opt.side_length)
            mutil_fake_B1_local_tokens = self.netPreViT(self.fake_B1_resize, self.atten_layers, get_tokens=True, local_id=local_id, side_length = self.opt.side_length)

            loss_spatial = self.calculate_similarity(mutil_real_A0_local_tokens, mutil_fake_B0_local_tokens) + self.calculate_similarity(mutil_real_A1_local_tokens, mutil_fake_B1_local_tokens)
            loss_spatial *= 0.5
        
        else:
            loss_spatial = 0.0

            

        return loss_global * self.opt.lambda_global, loss_spatial * self.opt.lambda_spatial

    def calculate_similarity(self, mutil_src_tokens, mutil_tgt_tokens):
        loss = 0.0
        n_layers = len(self.atten_layers)

        for src_tokens, tgt_tokens in zip(mutil_src_tokens, mutil_tgt_tokens):

            src_tgt = src_tokens.bmm(tgt_tokens.permute(0,2,1))
            tgt_src = tgt_tokens.bmm(src_tokens.permute(0,2,1))
            cos_dis_global =  F.cosine_similarity(src_tgt, tgt_src, dim=-1)
            loss += self.criterionL1(torch.ones_like(cos_dis_global), cos_dis_global).mean()

        loss = loss / n_layers
        return loss


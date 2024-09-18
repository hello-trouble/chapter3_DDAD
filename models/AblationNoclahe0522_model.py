########### for stage1 without recon, the loss about recon in G are deleted, and the update of net_recon are deleted
import os
# import numpy as np
import torch
import torch.nn as nn

# import torch.nn.functional as F
# import os
# import itertools
# import util.util as util
from .base_model import BaseModel

from TVLoss.TVLossL1 import L1_TVLoss_Charbonnier,CharbonnierLoss
from . import losses
from models import networks, networks_ori

from networks_msbdn.MSBDN_DFF_Dbranch04043 import Net as Net_domain_concat
from ECLoss.ECLoss import  DCLoss


from skimage.color import rgb2hsv

from util.util import np_to_torch,torch_to_np

import random
try:
    xrange  # Python2
except NameError:
    xrange = range  # Python 3

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
class AblationNoclahe0522Model(BaseModel):
    def name(self):
        return 'AblationNoclahe0522Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:

            parser.add_argument('--lambda_syn_Dehazing', type=float, default=0.0, help='weight for dehazing loss of synthetic data')
            parser.add_argument('--lambda_syn_vgg', type=float, default=0, help='whether use the vgg_loss in real_haze_img')


            parser.add_argument('--lambda_gan_un', type=float, default=0.0,help='the weight of real_haze_img')
            parser.add_argument('--lambda_recon', type=float, default=0.0, help='weight for reconstruction loss of real data')
            parser.add_argument('--lambda_recon_vgg', type=float, default=0.0,help='weight for reconstruction loss of real data')
            parser.add_argument('--lambda_rs_Dehazing_DC', type=float, default=0, help='weight for dark channel loss')
            parser.add_argument('--lambda_rs_Dehazing_TV', type=float, default=0, help='weight for TV loss')
            parser.add_argument('--use_patch', action='store_true', help='use the patch to constraint the reconstruction')
            parser.add_argument("--patchSize", type=int, default=0,help='whether use the patchNCE Loss')
            parser.add_argument('--patchD_3', type=int, default=0,help='whether use the transform loss in real_haze_img')


            parser.add_argument('--lambda_real_Dehazing', type=float, default=0.0,help='weight for dehazing loss of synthetic data')
            parser.add_argument('--lambda_real_vgg', type=float, default=0.0,help='whether use the vgg_loss in real_haze_img')
            parser.add_argument('--lambda_real_Dehazing_TV', type=float, default=0.0001, help='weight for TV loss')

            parser.add_argument('--end2end', action='store_true',help='pretrained dehazing model')


            parser.add_argument("--global_imageD", action='store_true',
                                help='whether use global discriminator to enhance the real dehaze results')

        return parser
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.end2end=self.opt.end2end
        if self.isTrain:
            visual_names_S = ['syn_haze_img', 'clear_img', 'syn_dehazing_img']  # , 's_rec_img'] ,
            visual_names_R = ['real_haze_img',  'r_clahe_img', 'rs_dehazing_img','real_dehazing_img']  # ,'clahe_img'
            self.visual_names = visual_names_S + visual_names_R
        else:
            visual_names_R = ['real_haze_img', 'r_dehazing_img', 'r_clahe_img']
            # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names += ['R_Dehazing','R_recon' ]#,'D_clear' ,'R_recon'
            if self.opt.global_imageD==True:
                self.model_names += ['D_A']
        else:
            self.model_names = ['R_Dehzaing']


        self.netR_Dehazing = networks.init_net( Net_domain_concat(), init_type='normal', gpu_ids=self.gpu_ids)
        self.netR_recon = networks.define_G(3, 3, 64, 'unet_256', 'instance', 'False', init_type='normal',
                                            init_gain=0.02, gpu_ids=self.gpu_ids)

        #define the adversarial network to discriminate the r_dehazed image whether similar to the clear image
        if self.isTrain:
            if self.opt.global_imageD==True:
                # self.netD_A = networks.define_D(3, 64, 'basic', 3, 'instance', init_gain=0.02, gpu_ids=self.gpu_ids)
                self.netD_A = networks_ori.define_D(3, 64, 'no_norm_4', 4, 'instance', False,  gpu_ids=self.gpu_ids, use_parallel=True)
            self.criterionGAN = losses.GANLoss(use_ls=not opt.no_lsgan).to(self.device) #use_ls = True
            self.criterionDehazing = CharbonnierLoss().to(self.device)
            self.criterionRecon = CharbonnierLoss().to(self.device)
            self.TVLoss = L1_TVLoss_Charbonnier().to(self.device)
            self.criterionVGG = networks.VGGLoss().to(self.device)

            # if self.opt.epoch_count<=45:
            #     self.opt.lr=opt.lr
            # elif self.opt.epoch_count>45 and self.opt.epoch_count<90:
            #     self.opt.lr = 0.000075
            # elif self.opt.epoch_count>=90 and self.opt.epoch_count<120:
            #     self.opt.lr=0.00005
            # elif self.opt.epoch_count >= 120:
            #     self.opt.lr = 0.000025
####### changed as below:
            if self.opt.epoch_count <= 45:
                self.opt.lr = self.opt.lr
            elif self.opt.epoch_count > 45 and self.opt.epoch_count < 90:
                self.opt.lr = 0.000075
            elif self.opt.epoch_count >= 90:
                self.opt.lr = 0.00005

            self.optimizer_G_task = torch.optim.Adam(self.netR_Dehazing.parameters(), lr=self.opt.lr, betas=(0.9, 0.999))
            self.optimizer_R_recon = torch.optim.Adam(self.netR_recon.parameters(),lr=self.opt.lr,betas=(0.9,0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_task)
            self.optimizers.append(self.optimizer_R_recon)
            if self.opt.global_imageD == True:
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
                self.optimizers.append(self.optimizer_D_A)

            # self.loss_D_fake_un = torch.tensor(0.0, dtype=torch.float)
            # self.loss_D_real_un = torch.tensor(0.0, dtype=torch.float)
            # self.loss_G_un = torch.tensor(0.0, dtype=torch.float)
            # self.loss_recon = torch.tensor(0.0, dtype=torch.float)
            # self.loss_G_un = torch.tensor(0.0, dtype=torch.float)
            # self.loss_R_Dehazing_DC = torch.tensor(0.0, dtype=torch.float)
            # self.loss_R_syn_Dehazing_TV = torch.tensor(0.0, dtype=torch.float)

            self.loss_real_Dehazing = torch.tensor(0.0, dtype=torch.float).to(self.device)
            self.loss_real_vgg = torch.tensor(0.0, dtype=torch.float).to(self.device)
            self.loss_R_Dehazing_TV = torch.tensor(0.0, dtype=torch.float).to(self.device)
            self.loss_Syn_vgg= torch.tensor(0.0, dtype=torch.float).to(self.device)
            self.loss_vgg_A = torch.tensor(0.0, dtype=torch.float).to(self.device)
            self.loss_real_Dehazing_feature  = torch.tensor(0.0, dtype=torch.float).to(self.device)
            self.loss_G_un = torch.tensor(0.0, dtype=torch.float).to(self.device)
            self.loss_D_real_un = torch.tensor(0.0, dtype=torch.float32).to(self.device)
            self.loss_D_fake_un = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        if self.isTrain and self.opt.continue_train:
            self.load_networks(self.opt.which_epoch)
    def load_networks(self, which_epoch):
        if int(which_epoch)<=90:
            self.model_names=['R_Dehazing','R_recon' ]
        elif int(which_epoch)>90 and self.opt.global_imageD == True:
            self.model_names = ['R_Dehazing', 'R_recon','D_A']

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                # if isinstance(net, torch.nn.DataParallel):
                #     net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def set_input(self, input):

        if self.isTrain:
            AtoB = self.opt.which_direction == 'AtoB'
            input_A = input['A' if AtoB else 'B']#haze image
            input_B = input['B' if AtoB else 'A'] #ground image

            input_D = input['D' ]  # _real_haze image
            input_E = input['E']  # real_enhance image
            # input_F = input['F']  # unpaired_clear img

            self.syn_haze_img = input_A.to(self.device)
            self.clear_img = input_B.to(self.device)
            # self.unpaired_clear_img = input_F.to(self.device)


            self.real_haze_img  = input_D.to(self.device)
            # self.real_clahe_img = input_E.to(self.device)

            self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        _,_,_,self.syn_dehazing_img= self.netR_Dehazing(self.syn_haze_img,data_type='syn')

        _,_,_,self.real_syn_dehazing_img = self.netR_Dehazing(self.real_haze_img,data_type='real_syn')
        _,_,_,self.real_dehazing_img= self.netR_Dehazing(self.real_haze_img, data_type='real') # data_type='real_syn' is changed in 20220614
        self.rs_recon_img = self.netR_recon(self.real_syn_dehazing_img)

        # if self.opt.use_patch:
        #     w = self.clear_img.size(3)
        #     h = self.clear_img.size(2)
        #     w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
        #     h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
        #     # self.dehazing_patch = self.real_dehazing_img[:, :, h_offset:h_offset + self.opt.patchSize,
        #     #                   w_offset:w_offset + self.opt.patchSize]
        #     self.input_patch = self.real_clahe_img[:, :, h_offset:h_offset + self.opt.patchSize,
        #                        w_offset:w_offset + self.opt.patchSize]
        #     self.recon_patch = self.rs_recon_img[:, :, h_offset:h_offset + self.opt.patchSize,
        #                        w_offset:w_offset + self.opt.patchSize]
        #     if self.opt.patchD_3 > 0:
        #         self.dehazing_patch_1 = []
        #
        #         self.input_patch_1 = []
        #         self.recon_patch_1 = []
        #         w = self.clear_img.size(3)
        #         h = self.clear_img.size(2)
        #         for i in range(self.opt.patchD_3):
        #             w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
        #             h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
        #             # self.dehazing_patch_1.append(self.real_dehazing_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #             #                          w_offset_1:w_offset_1 + self.opt.patchSize])
        #             self.input_patch_1.append(self.real_clahe_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                                       w_offset_1:w_offset_1 + self.opt.patchSize])
        #             self.recon_patch_1.append(self.rs_recon_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                                       w_offset_1:w_offset_1 + self.opt.patchSize])

    def forward3(self):
        _ , _ , _ ,self.syn_dehazing_img = self.netR_Dehazing(self.syn_haze_img, data_type='syn')

        _, _, self.real_syn_features, self.real_syn_dehazing_img = self.netR_Dehazing(self.real_haze_img, data_type='real_syn')
        _, _, self.real_features, self.real_dehazing_img = self.netR_Dehazing(self.real_haze_img, data_type='real')
        self.rs_recon_img = self.netR_recon(self.real_syn_dehazing_img)

        # if self.opt.use_patch:
        #     w = self.clear_img.size(3)
        #     h = self.clear_img.size(2)
        #     w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
        #     h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))
        #     # self.dehazing_patch = self.real_dehazing_img[:, :, h_offset:h_offset + self.opt.patchSize,
        #     #                   w_offset:w_offset + self.opt.patchSize]
        #     self.input_patch = self.real_clahe_img[:, :, h_offset:h_offset + self.opt.patchSize,
        #                        w_offset:w_offset + self.opt.patchSize]
        #     self.recon_patch = self.rs_recon_img[:, :, h_offset:h_offset + self.opt.patchSize,
        #                        w_offset:w_offset + self.opt.patchSize]
        #     if self.opt.patchD_3 > 0:
        #         self.dehazing_patch_1 = []
        #
        #         self.input_patch_1 = []
        #         self.recon_patch_1 = []
        #         w = self.clear_img.size(3)
        #         h = self.clear_img.size(2)
        #         for i in range(self.opt.patchD_3):
        #             w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
        #             h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
        #             # self.dehazing_patch_1.append(self.real_dehazing_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #             #                          w_offset_1:w_offset_1 + self.opt.patchSize])
        #             self.input_patch_1.append(self.real_clahe_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                                       w_offset_1:w_offset_1 + self.opt.patchSize])
        #             self.recon_patch_1.append(self.rs_recon_img[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
        #                                       w_offset_1:w_offset_1 + self.opt.patchSize])
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G_stage1(self):

        lambda_syn_Dehazing = self.opt.lambda_syn_Dehazing
        lambda_syn_vgg = self.opt.lambda_syn_vgg

        lambda_rs_Dehazing_DC = self.opt.lambda_rs_Dehazing_DC
        lambda_rs_Dehazing_TV = self.opt.lambda_rs_Dehazing_TV
        lambda_gan_un = self.opt.lambda_gan_un
        lambda_recon = self.opt.lambda_recon
        # lambda_recon_vgg = self.opt.lambda_recon_vgg

        # lambda_real_Dehazing = self.opt.lambda_real_Dehazing
        # lambda_real_vgg = self.opt.lambda_real_vgg
        # lambda_real_Dehazing_TV = self.opt.lambda_real_Dehazing_TV

        #synthetic_haze_images
        self.loss_Syn_Dehazing = self.criterionDehazing(self.syn_dehazing_img, self.clear_img)* lambda_syn_Dehazing
        self.loss_Syn_vgg = self.criterionVGG(self.syn_dehazing_img, self.clear_img,data_type='syn')*lambda_syn_vgg

        #real_clahe_images
        self.loss_R_syn_Dehazing_DC = DCLoss((self.real_syn_dehazing_img + 1) / 2,self.opt.patch_size) * lambda_rs_Dehazing_DC
        self.loss_R_syn_Dehazing_TV = self.TVLoss(self.real_syn_dehazing_img) * lambda_rs_Dehazing_TV
        self.loss_recon = self.criterionRecon(self.rs_recon_img,self.real_haze_img)*lambda_recon
        # self.loss_vgg_A = self.criterionVGG(self.rs_recon_img,self.real_clahe_img,data_type='syn') * lambda_recon_vgg
        # if self.opt.use_patch:
        #     loss_vgg_patch = self.criterionVGG(self.recon_patch,  self.input_patch,data_type='syn')* lambda_recon_vgg
        #     loss_patch_recon = self.criterionRecon(self.recon_patch, self.input_patch) * lambda_recon
        #     if self.opt.patchD_3 > 0:
        #         for i in range(self.opt.patchD_3):
        #             loss_vgg_patch += self.criterionVGG(self.recon_patch_1[i],self.input_patch_1[i],data_type='syn') * lambda_recon_vgg
        #             loss_patch_recon += self.criterionRecon(self.recon_patch_1[i],self.input_patch_1[i]) * lambda_recon
        #         self.loss_vgg_A += loss_vgg_patch / (self.opt.patchD_3 + 1)
        #         self.loss_recon += loss_patch_recon / (self.opt.patchD_3 + 1)
        # considering  multi_patch loss
        # if self.opt.global_imageD:
        #     pred_fake_un =self.netD_A(self.real_syn_dehazing_img)
        #     self.loss_G_un = self.criterionGAN(pred_fake_un,True)*lambda_gan_un
        # else:
        #     self.loss_G_un= torch.tensor(0.0,dtype=torch.float).to(self.device)
        #     self.loss_D_real_un = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        #     self.loss_D_fake_un = torch.tensor(0.0, dtype=torch.float32).to(self.device)


        # real_haze_images

        # self.loss_real_Dehazing = self.criterionDehazing(self.real_dehazing_img,self.real_syn_dehazing_img.detach())* lambda_real_Dehazing
        # self.loss_real_vgg = self.criterionVGG(self.real_dehazing_img, self.real_syn_dehazing_img.detach(),data_type='syn')*lambda_real_vgg
        # self.loss_R_Dehazing_TV = self.TVLoss(self.real_dehazing_img) * lambda_real_Dehazing_TV
        # for i in range(len(self.real_out)-1):
        #     self.loss_real_Dehazing += self.criterionDehazing(self.real_out[i],self.real_syn_out[i].detach())* lambda_real_Dehazing*0.25

        self.loss_G = self.loss_Syn_Dehazing+self.loss_Syn_vgg+ self.loss_R_syn_Dehazing_DC + self.loss_R_syn_Dehazing_TV + self.loss_recon
        self.loss_G.backward()

    def backward_G_stage2(self):

            lambda_syn_Dehazing = self.opt.lambda_syn_Dehazing
            lambda_syn_vgg = self.opt.lambda_syn_vgg

            lambda_rs_Dehazing_DC = self.opt.lambda_rs_Dehazing_DC
            lambda_rs_Dehazing_TV = self.opt.lambda_rs_Dehazing_TV
            lambda_gan_un = self.opt.lambda_gan_un
            lambda_recon = self.opt.lambda_recon
            lambda_recon_vgg = self.opt.lambda_recon_vgg

            lambda_real_Dehazing = self.opt.lambda_real_Dehazing
            lambda_real_vgg = self.opt.lambda_real_vgg
            lambda_real_Dehazing_TV = self.opt.lambda_real_Dehazing_TV

            # synthetic_haze_images
            self.loss_Syn_Dehazing = self.criterionDehazing(self.syn_dehazing_img, self.clear_img) * lambda_syn_Dehazing
            self.loss_Syn_vgg = self.criterionVGG(self.syn_dehazing_img, self.clear_img,data_type='syn')*lambda_syn_vgg

            # real_clahe_images
            self.loss_R_syn_Dehazing_DC = DCLoss((self.real_syn_dehazing_img + 1) / 2,
                                                 self.opt.patch_size) * lambda_rs_Dehazing_DC
            self.loss_R_syn_Dehazing_TV = self.TVLoss(self.real_syn_dehazing_img) * lambda_rs_Dehazing_TV
            self.loss_recon = self.criterionRecon(self.rs_recon_img, self.real_haze_img) * lambda_recon
            # self.loss_vgg_A = self.criterionVGG(self.rs_recon_img,self.real_clahe_img,data_type='syn') * lambda_recon_vgg
            # if self.opt.use_patch:
            #     loss_vgg_patch = self.criterionVGG(self.recon_patch,  self.input_patch,data_type='syn')* lambda_recon_vgg
            #     loss_patch_recon = self.criterionRecon(self.recon_patch, self.input_patch) * lambda_recon
            #     if self.opt.patchD_3 > 0:
            #         for i in range(self.opt.patchD_3):
            #             loss_vgg_patch += self.criterionVGG(self.recon_patch_1[i],self.input_patch_1[i],data_type='syn') * lambda_recon_vgg
            #             loss_patch_recon += self.criterionRecon(self.recon_patch_1[i],self.input_patch_1[i]) * lambda_recon
            #         self.loss_vgg_A += loss_vgg_patch / (self.opt.patchD_3 + 1)
            #         self.loss_recon += loss_patch_recon / (self.opt.patchD_3 + 1)
            # considering  multi_patch loss
            if self.opt.global_imageD:
                pred_fake_un = self.netD_A(self.real_syn_dehazing_img)
                self.loss_G_un = self.criterionGAN(pred_fake_un, True) * lambda_gan_un
            else:
                self.loss_G_un =  torch.tensor(0.0, dtype=torch.float).to(self.device)

            # real_haze_images

            # self.loss_real_Dehazing = self.criterionDehazing(self.real_dehazing_img,self.real_syn_dehazing_img.detach())* lambda_real_Dehazing
            # self.loss_real_vgg = self.criterionVGG(self.real_dehazing_img, self.real_syn_dehazing_img.detach(),data_type='syn')*lambda_real_vgg
            # self.loss_R_Dehazing_TV = self.TVLoss(self.real_dehazing_img) * lambda_real_Dehazing_TV
            # for i in range(len(self.real_out)-1):
            #     self.loss_real_Dehazing += self.criterionDehazing(self.real_out[i],self.real_syn_out[i].detach())* lambda_real_Dehazing*0.25

            self.loss_G = self.loss_Syn_Dehazing +self.loss_Syn_vgg+ self.loss_R_syn_Dehazing_DC + self.loss_R_syn_Dehazing_TV + self.loss_recon + self.loss_G_un
            self.loss_G.backward()

    def backward_G_stage3(self):

        lambda_syn_Dehazing = self.opt.lambda_syn_Dehazing
        lambda_syn_vgg = self.opt.lambda_syn_vgg

        lambda_rs_Dehazing_DC = self.opt.lambda_rs_Dehazing_DC
        lambda_rs_Dehazing_TV = self.opt.lambda_rs_Dehazing_TV
        lambda_gan_un = self.opt.lambda_gan_un
        lambda_recon = self.opt.lambda_recon
        lambda_recon_vgg = self.opt.lambda_recon_vgg

        lambda_real_Dehazing = self.opt.lambda_real_Dehazing
        lambda_real_vgg = self.opt.lambda_real_vgg
        lambda_real_Dehazing_TV = self.opt.lambda_real_Dehazing_TV

        # synthetic_haze_images
        self.loss_Syn_Dehazing = self.criterionDehazing(self.syn_dehazing_img, self.clear_img) * lambda_syn_Dehazing
        self.loss_Syn_vgg = self.criterionVGG(self.syn_dehazing_img, self.clear_img,data_type='syn')*lambda_syn_vgg

        # real_clahe_images
        self.loss_R_syn_Dehazing_DC = DCLoss((self.real_syn_dehazing_img + 1) / 2,
                                             self.opt.patch_size) * lambda_rs_Dehazing_DC
        self.loss_R_syn_Dehazing_TV = self.TVLoss(self.real_syn_dehazing_img) * lambda_rs_Dehazing_TV
        self.loss_recon = self.criterionRecon(self.rs_recon_img, self.real_haze_img) * lambda_recon
        # self.loss_vgg_A = self.criterionVGG(self.rs_recon_img,self.real_clahe_img,data_type='syn') * lambda_recon_vgg
        # if self.opt.use_patch:
        #     loss_vgg_patch = self.criterionVGG(self.recon_patch,  self.input_patch,data_type='syn')* lambda_recon_vgg
        #     loss_patch_recon = self.criterionRecon(self.recon_patch, self.input_patch) * lambda_recon
        #     if self.opt.patchD_3 > 0:
        #         for i in range(self.opt.patchD_3):
        #             loss_vgg_patch += self.criterionVGG(self.recon_patch_1[i],self.input_patch_1[i],data_type='syn') * lambda_recon_vgg
        #             loss_patch_recon += self.criterionRecon(self.recon_patch_1[i],self.input_patch_1[i]) * lambda_recon
        #         self.loss_vgg_A += loss_vgg_patch / (self.opt.patchD_3 + 1)
        #         self.loss_recon += loss_patch_recon / (self.opt.patchD_3 + 1)
        # considering  multi_patch loss
        if self.opt.global_imageD:
            pred_fake_un = self.netD_A(self.real_syn_dehazing_img)
            self.loss_G_un = self.criterionGAN(pred_fake_un, True) * lambda_gan_un
        else:
            self.loss_G_un = torch.tensor(0.0, dtype=torch.float).to(self.device)

        # real_haze_images

        self.loss_real_Dehazing = self.criterionDehazing(self.real_dehazing_img,self.real_syn_dehazing_img.detach())* lambda_real_Dehazing

        # self.loss_real_vgg = self.criterionVGG(self.real_dehazing_img, self.real_syn_dehazing_img.detach(),data_type='syn')*lambda_real_vgg
        self.loss_real_Dehazing_feature = self.criterionDehazing(self.real_features,self.real_syn_features.detach())* lambda_real_Dehazing
        self.loss_R_Dehazing_TV = self.TVLoss(self.real_dehazing_img) * lambda_real_Dehazing_TV
        # for i in range(len(self.real_out)-1):
        #     self.loss_real_Dehazing += self.criterionDehazing(self.real_out[i],self.real_syn_out[i].detach())* lambda_real_Dehazing*0.25

        self.loss_G = self.loss_Syn_Dehazing  +self.loss_Syn_vgg+ self.loss_R_syn_Dehazing_DC + self.loss_R_syn_Dehazing_TV + self.loss_recon + self.loss_G_un +self.loss_real_Dehazing +  self.loss_real_Dehazing_feature
        self.loss_G.backward()
    def optimize_parameters_stage(self):
        self.forward()
        # if self.opt.global_imageD==True:
        #     self.set_requires_grad([self.netD_A], False)  # self.netD_clear,
        self.optimizer_G_task.zero_grad()
        self.optimizer_R_recon.zero_grad()
        self.backward_G_stage1()
        self.optimizer_G_task.step()
        self.optimizer_R_recon.step()

        # if self.opt.global_imageD==True:
        #     self.set_requires_grad([self.netD_A], True)  # self.netD_Rfeat,
        #     self.optimizer_D_A.zero_grad()
        #     self.backward_D_depth()
        #     self.optimizer_D_A.step()

    def optimize_parameters_stage2(self):
        self.forward()
        if self.opt.global_imageD == True:
            self.set_requires_grad([self.netD_A], False)  # self.netD_clear,
        self.optimizer_G_task.zero_grad()
        self.optimizer_R_recon.zero_grad()
        self.backward_G_stage2()
        self.optimizer_G_task.step()
        self.optimizer_R_recon.step()
        if self.opt.global_imageD == True:
            self.set_requires_grad([self.netD_A], True)  # self.netD_Rfeat,
            self.optimizer_D_A.zero_grad()
            self.backward_D_depth()
            self.optimizer_D_A.step()

    def optimize_parameters_stage3(self):
        self.forward3()
        if self.opt.global_imageD == True:
            self.set_requires_grad([self.netD_A], False)  # self.netD_clear,
        self.optimizer_G_task.zero_grad()
        self.optimizer_R_recon.zero_grad()
        self.backward_G_stage3()
        self.optimizer_G_task.step()
        self.optimizer_R_recon.step()

        if self.opt.global_imageD == True:
            self.set_requires_grad([self.netD_A], True)  # self.netD_Rfeat,
            self.optimizer_D_A.zero_grad()
            self.backward_D_depth()
            self.optimizer_D_A.step()

    def update_learning_rate(self,epoch):

        if epoch <= 45:
            self.opt.lr = self.opt.lr
        elif epoch > 45 and epoch < 90:
            self.opt.lr = 0.000075
        elif epoch >= 90:
            self.opt.lr = 0.00005

        for param_group in self.optimizer_G_task.param_groups:
            param_group['lr']=self.opt.lr
        if self.opt.global_imageD:
            for param_group in self.optimizer_D_A.param_groups:
                param_group['lr'] =self.opt.lr
        for param_group in self.optimizer_R_recon.param_groups:
            param_group['lr'] = self.opt.lr
        print('update learning rate: %f ' % (self.opt.lr))


    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD(real.detach())
        pred_fake = netD(fake.detach())
        if use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_depth(self):

        pred_real_Dun = self.netD_A(self.clear_img)
        pred_fake_Dun = self.netD_A(self.real_syn_dehazing_img.detach())
        self.loss_D_real_un = self.criterionGAN(pred_real_Dun,True)
        self.loss_D_fake_un = self.criterionGAN(pred_fake_Dun,False)
        # else:
        #     self.loss_D_real_un = torch.tensor(0.0,dtype=torch.float32)
        #     self.loss_D_fake_un = torch.tensor(0.0,dtype=torch.float32)
        self.loss_D =(self.loss_D_real_un + self.loss_D_fake_un)/2.0
        self.loss_D.backward()
# # save models to the disk
    def save_networks(self, which_epoch):


        self.model_names= ['R_Dehazing','R_recon'] #,'D_P'
        if self.opt.global_imageD == True:
            self.model_names += ['D_A']

        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                # net_dict = net.cpu().state_dict()
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

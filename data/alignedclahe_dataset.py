
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import util.util as util


class AlignedClaheDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_AB = os.path.join(opt.dataroot, opt.phase)


        self.dir_D = os.path.join(opt.dataroot, 'real_train_resize')
        self.dir_E = os.path.join(opt.dataroot, 'real_train_resize_widthclahe') #changed at 0501

        # self.dir_F = os.path.join(opt.dataroot, 'unpaired_clear')  # 930

        self.AB_paths = sorted(make_dataset(self.dir_AB))
        # self.C_paths = sorted(make_dataset(self.dir_C))
        self.D_paths = sorted(make_dataset(self.dir_D))
        self.E_paths = sorted(make_dataset(self.dir_E))
        # self.F_paths = sorted(make_dataset(self.dir_F))

        self.transformPIL = transforms.ToPILImage()
        transform_list1 = [transforms.ToTensor()]
        transform_list2 = [transforms.Normalize([0.5],
                                               [0.5])]


        self.transform1 = transforms.Compose(transform_list1)
        self.transform2 = transforms.Compose(transform_list2)


    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        if self.opt.phase == 'train':
            AB_path = self.AB_paths[index]

            # and C is the unlabel hazy image
            r_ind = random.randint(0, int((len(self.D_paths)-1)))
            D_path = self.D_paths[r_ind]
            E_path = self.E_paths[r_ind]

            # unpaired_ind = random.randint(0, int(len(self.F_paths) - 1))
            # F_path = self.F_paths[unpaired_ind]

            # C_path = self.C_paths[random.randint(0, len(self.AB_paths)-2200)]
            AB = Image.open(AB_path).convert('RGB')  # image
            # C = Image.open(C_path).convert('RGB')   #image
            D = Image.open(D_path).convert('RGB')
            E = Image.open(E_path).convert('RGB')
            # F = Image.open(F_path).convert('RGB')

            width_D = D.width
            height_D = D.height
            if width_D < self.opt.fineSize or height_D < self.opt.fineSize:
                if width_D <= height_D:
                    width_new= self.opt.fineSize
                    height_new = width_new * int(height_D/width_D)
                elif height_D < width_D:
                    height_new = self.opt.fineSize
                    width_new = height_new * int(width_D/height_new)
                D = D.resize((width_new,height_new),resample=Image.BICUBIC)
                E = E.resize((width_new, height_new), resample=Image.BICUBIC)

            # width_F = F.width
            # height_F = F.height
            # if width_F < self.opt.fineSize or height_F < self.opt.fineSize:
            #     if width_F <= height_F:
            #         width_new = self.opt.fineSize
            #         height_new = width_new * int(height_F / width_F)
            #     elif height_F < width_F:
            #         height_new = self.opt.fineSize
            #         width_new = height_new * int(width_F / height_new)
            #     F = F.resize((width_new, height_new), resample=Image.BICUBIC)

            AB = self.transform1(AB)
            # C = self.transform1(C)
            D = self.transform1(D)
            E = self.transform1(E)
            # F = self.transform1(F)


            ######### crop the training image into fineSize ########
            w_total = AB.size(2)
            w = int(w_total / 2)
            h = AB.size(1)
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

            A = AB[:, h_offset:h_offset + self.opt.fineSize,
                   w_offset:w_offset + self.opt.fineSize]
            B = AB[:, h_offset:h_offset + self.opt.fineSize,
                   w + w_offset:w + w_offset + self.opt.fineSize]
            # C = C[:, h_offset:h_offset + self.opt.fineSize,
            #     w_offset:w_offset + self.opt.fineSize]

            w = D.size(2)
            h = D.size(1)

            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            D = D[:, h_offset:h_offset + self.opt.fineSize,w_offset:w_offset + self.opt.fineSize]
            E = E[:, h_offset:h_offset+self.opt.fineSize, w_offset:w_offset+self.opt.fineSize]

            # w = F.size(2)
            # h = F.size(1)
            # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            # F = F[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(2, idx)
                B = B.index_select(2, idx)
                D = D.index_select(2, idx)
                #C = C.index_select(2, idx)
                E = E.index_select(2, idx)
                # F = F.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx =[i for i in range(A.size(1)-1,-1,-1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(1,idx)
                B = B.index_select(1,idx)
                D = D.index_select(1,idx)
                #C = C.index_select(1,idx)
                E= E.index_select(1, idx)
                # F = F.index_select(1, idx)
            A = self.transform2(A)
            B = self.transform2(B)
            # C= self.transform2(C)

            D = self.transform2(D)
            E = self.transform2(E)
            # F = self.transform2(F)
            if random.random()<0.5:
                noise = torch.randn(3, self.opt.fineSize, self.opt.fineSize) / 100
                A = A + noise
            return {'A': A, 'B': B,  'D': D,  'E':E, 'D_paths': D_path,
                    'A_paths': AB_path, 'B_paths': AB_path} #  'C': C, A:haze,B:gt, C:real_haze, D: depth of A, E:depth of C

        # elif self.opt.phase == 'test':
        #     if self.opt.test_type == 'syn':
        #         AB_path = self.AB_paths[index]
        #         # F_path = self.F_paths[index]
        #         # F = Image.open(F_path)
        #         AB = Image.open(AB_path).convert('RGB')
        #         ori_w = AB.width
        #         ori_h = AB.height
        #         # ori_fw = F.width
        #         # ori_fh = F.height
        #
        #
        #         # new_w = int(np.floor(ori_w/32)*32)
        #         # new_h = int(np.floor(ori_h/16)*16)
        #         # new_w = ori_w
        #         # new_h = ori_h
        #         new_w = 1024
        #         new_h = 512
        #         AB = AB.resize((int(new_w), int(new_h)), Image.BICUBIC)
        #         AB = self.transform1(AB)
        #         AB = self.transform2(AB)
        #         # F = F.resize((ori_fw, ori_fh), Image.BICUBIC)
        #         # F = self.transform1(F)
        #         # F = self.transform2(F)
        #         A = AB[:,:,0:int(new_w/2)]
        #         B = AB[:,:,int(new_w/2):new_w]
        #         return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}#'F': F,
        #
        #
        #     elif self.opt.test_type == 'real':
        #         C_path = self.D_paths[index]
        #         F_path=self.F_paths[index]
        #         C = Image.open(C_path).convert('RGB')
        #         F=Image.open(F_path).convert('RGB')
        #         C_w = C.width
        #         C_h = C.height
        #         # C = C.resize((C_w, C_h), Image.BICUBIC)
        #
        #         new_w = int(np.floor(C_w / 16) * 16)
        #         new_h = int(np.floor(C_h / 16) * 16)
        #         C = C.resize((int(new_w), int(new_h)), Image.BICUBIC)
        #         F=F.resize((int(new_w), int(new_h)), Image.BICUBIC)
        #         C = self.transform1(C)
        #         C = self.transform2(C)
        #         F=self.transform1(F)
        #         F=self.transform2(F)
        #         return {'C': C, 'C_paths': C_path,'F': F, 'F_paths': F_path}
        #A = self.transformPIL(A)
        #B = self.transformPIL(B)
        #A_half = self.transformPIL(A_half)
        #A.save('A.png')
        #B.save('B.png')
        #A_half.save('A_half.png')



    def __len__(self):

        if self.opt.phase == 'train':
            return len(self.AB_paths)
        # elif self.opt.phase == 'test':
        #     if self.opt.test_type == 'syn':
        #         return len(self.AB_paths)
        #     elif self.opt.test_type == 'real':
        #         return len(self.C_paths)

    def name(self):
        return 'AlignedclaheDataset'




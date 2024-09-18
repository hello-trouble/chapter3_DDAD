import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms 
from .grad_conv import grad_conv_hor, grad_conv_vet
from torch.nn.functional import l1_loss
import torch.nn as nn

# img must be variable with grad and of dim N*C*W*H
def TVLossL1(img):
    hor = grad_conv_hor()(img)
    vet = grad_conv_vet()(img)
    target = torch.autograd.Variable(torch.FloatTensor(img.shape).zero_().cuda())
    loss_hor = l1_loss(hor, target, size_average=False)
    loss_vet = l1_loss(vet, target, size_average=False)
    loss = loss_hor+loss_vet
    return loss

def CORAL_loss(src, tar):
    d=src.data.shape[1]
    src_norm=torch.mean(src, dim=0,keepdim=True)-src
    src_cov=src_norm.t() @ src_norm

    tar_norm=torch.mean(tar,dim=0,keepdim=True)-tar
    tar_cov=tar_norm.t()@ tar_norm

    coral=torch.mean(torch.mul((src_cov-tar_cov),(src_cov-tar_cov)))
    coral_loss=coral/(4*d*d)
    return coral_loss
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
class L1_TVLoss_Charbonnier(nn.Module):

    def __init__(self):

        super(L1_TVLoss_Charbonnier, self).__init__()

        self.e = 0.000001 ** 2



    def forward(self, x):

        batch_size = x.size()[0]

        h_tv = torch.abs((x[:, :, 1:, :]-x[:, :, :-1, :]))

        h_tv = torch.mean(torch.sqrt(h_tv ** 2 + self.e))

        w_tv = torch.abs((x[:, :, :, 1:]-x[:, :, :, :-1]))

        w_tv = torch.mean(torch.sqrt(w_tv ** 2 + self.e))

        return h_tv + w_tv

if __name__ == "__main__":
    img = Image.open('1.jpg')
    img = transforms.ToTensor()(img)[None, :, :, :]
    img = torch.autograd.Variable(img, requires_grad=True)
    
    loss = TVLossL1(img)

from torchvision.utils import make_grid
import os
import numpy as np
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from collections import OrderedDict

#from Experiments.pretrained_model.MSBDN.MSBDN_DFF import Net as MSBDN_ori
from Experiments.pretrained_model.SSID import SSID_networks
from Experiments.pretrained_model.DAD import networks_DAD
from Experiments.pretrained_model.PSD.PSD_MSBDN import MSBDNNet as PSD_MSBDN
from Experiments.pretrained_model.Ours.MSBDN_DFF_Dbranch04043 import Net as NetDAB
import models.networks as networks
abs=os.getcwd()+"/"
os.environ['CUDA_VISIBLE_DEVICES']='0'
def tensor_show(tensors,titles=['haze']):
    fig=plt.figure()
    for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
        img=make_grid(tensor)
        npimg=img.numpy()
        ax=fig.add_subplot(221+i)
        ax.imshow(np.transpose(npimg,(1,2,0)))
        ax.set_title(tit)
    plt.show()
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy=image_tensor[0].cpu().float().numpy()
    image_numpy=(np.transpose(image_numpy, (1, 2, 0))+1)/2.0*255.0
    image_numpy=np.maximum(image_numpy, 0)
    image_numpy=np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def tensor2im2(image_tensor, imtype=np.uint8):
    image_numpy=image_tensor[0].cpu().float().numpy()
    image_numpy=np.transpose(image_numpy, (1, 2, 0))*255.0
    image_numpy=np.maximum(image_numpy, 0)
    image_numpy=np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)
def pad_tensor(input):
    height_org, width_org=input.shape[2], input.shape[3]
    divide=64

    if width_org%divide!=0 or height_org%divide!=0:

        width_res=width_org%divide
        height_res=height_org%divide
        if width_res!=0:
            width_div=divide-width_res
            pad_left=int(width_div/2)
            pad_right=int(width_div-pad_left)
        else:
            pad_left=0
            pad_right=0

        if height_res!=0:
            height_div=divide-height_res
            pad_top=int(height_div/2)
            pad_bottom=int(height_div-pad_top)
        else:
            pad_top=0
            pad_bottom=0

        padding=nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))  # 注意这里的参数，是左右，上下，先宽后高。tensor的维度则是先高后宽。
        input=padding(input)
    else:
        pad_left=0
        pad_right=0
        pad_top=0
        pad_bottom=0

    height, width=input.data.shape[2], input.data.shape[3]
    assert width%divide==0, 'width cant divided by stride'
    assert height%divide==0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom
def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width=input.shape[2], input.shape[3]
    return input[:, :, pad_top: height-pad_bottom, pad_left: width-pad_right]

parser=argparse.ArgumentParser()
parser.add_argument("--gpu_ids",type=str,default="0",help= 'use which gpu_card,used in defination of the network')
parser.add_argument("--test_type",type=str,default='real', help = " syn with gt, real without gt")
parser.add_argument("--syn_dir",type=str,default="/home/qiaoyingxv/datasets/domain_dehazing/HazeRD/haze_resized")#test_haze
parser.add_argument("--gt_dir",type = str, default= "/home/qiaoyingxv/datasets/domain_dehazing/HazeRD/clear_resized")#test_gt
parser.add_argument("--real_dir",type=str,default="/home/qiaoyingxv/datasets/RTTS_inPSD/JPEGImages") #domain_dehazing/realcoal_haze   real_haze_resized UnannotatedHazyImages #real_haze_resized
parser.add_argument("--out_dir",type=str,default="Experiments/Results/compare_visual/HazeRD") #realcoal_haze_test0906
parser.add_argument("---pretrained_DA_syn",type=str,default="Experiments/pretrained_model/DAD/30_netS_Dehazing.pth")
parser.add_argument("---pretrained_DA_real",type=str,default="Experiments/pretrained_model/DAD/30_netR_Dehazing.pth")
parser.add_argument("---pretrained_msbdn_origin",type=str,default="Experiments/pretrained_model/MSBDN/model.pkl")
parser.add_argument("---pretrained_SLDD",type=str,default="Experiments/pretrained_model/SSID/latest_net_G.pth")
opt=parser.parse_args()
str_ids =opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
	id = int(str_id)
	if id >= 0:
	    opt.gpu_ids.append(id)
if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
device="cuda" if torch.cuda.is_available() else 'cpu'

# net_msbdn_ori =MSBDN_ori().cuda()
# netSSLD_Dehazing = SSID_networks.define_G(3, 3, 64, 'EDskipconn', 'instance','True', opt.gpu_ids, 'False', 'True')
# netS_Dehazing = networks_DAD.define_Gen(3, 3, 64, 4, 'batch','PReLU', 'UNet', 'kaiming', 0,False, 0, 0.1)
# netR_Dehazing = networks_DAD.define_Gen(3, 3, 64, 4, 'batch', 'PReLU', 'UNet', 'kaiming', 0,False, 0, 0.1)
netPSD = PSD_MSBDN().to(device)
net_Ours = NetDAB().to(device)

# netSSLD_Dehazing = netSSLD_Dehazing.cuda()
# netS_Dehazing=nn.DataParallel(netS_Dehazing)
# netR_Dehazing=nn.DataParallel(netR_Dehazing)
netPSD = nn.DataParallel(netPSD)

net_msbdn_ori = torch.load('networks/model.pkl', map_location=str(device))
ckpt_syn = torch.load(opt.pretrained_DA_syn, map_location=str(device))
ckpt_real = torch.load(opt.pretrained_DA_real,map_location=str(device))
ckpt_SSLD = torch.load("Experiments/pretrained_model/SSID/latest_net_G.pth")
ckpt_ours = torch.load('Experiments/pretrained_model/Ours/186_netR_Dehazing.pth')
new_state_dict = OrderedDict()
for k, v in ckpt_ours.items():
    name = k[7:]
    new_state_dict[name] = v

# net_msbdn_ori.load_state_dict(ckpt_msbdn_ori)
netS_Dehazing.load_state_dict(ckpt_syn)
netR_Dehazing.load_state_dict(ckpt_real)
netSSLD_Dehazing.load_state_dict(ckpt_SSLD)
netPSD.load_state_dict(torch.load('Experiments/pretrained_model/PSD/PSB-MSBDN'))
net_Ours.load_state_dict(new_state_dict)


netS_Dehazing.eval()
netR_Dehazing.eval()
net_msbdn_ori.eval()
netSSLD_Dehazing.eval()
netPSD.eval()
net_Ours.eval()

if opt.test_type=='syn':
    test_dir= opt.syn_dir
else:
    test_dir = opt.real_dir
for img in os.listdir(test_dir):
    image_dir=os.path.join(test_dir,img)
    if opt.test_type == 'syn':

        gt_img_path = os.path.join(opt.gt_dir, img)
        gt_img = Image.open(gt_img_path)
        gt_img = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])(gt_img)[None,::]
        gt_img, pad_left, pad_right, pad_top, pad_bottom=pad_tensor(gt_img)
        gt_img= pad_tensor_back(gt_img, pad_left, pad_right, pad_top, pad_bottom)
        gt_img = gt_img.cuda()
    output_image_dir=os.path.join(opt.out_dir,img)
    haze_img=Image.open(image_dir).convert("RGB")
    haze_img=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])(haze_img)[None,::]
    haze_img,pad_left, pad_right, pad_top, pad_bottom=pad_tensor(haze_img)
    haze_img=haze_img.cuda()
    with torch.no_grad():
       netS_img=netS_Dehazing(haze_img)
       netR_img=netR_Dehazing(haze_img)
       msbdn_ori_img = net_msbdn_ori(haze_img)
       netSSLD_Dehazing_img = netSSLD_Dehazing(haze_img)
       psd_img = netPSD(haze_img)
       if opt.test_type == 'syn':
            _,_,_,dehazing_img = net_Ours(haze_img,data_type='syn')
       elif opt.test_type == 'real':
           _, _, _, dehazing_img = net_Ours(haze_img, data_type='real')
       haze_img = pad_tensor_back(haze_img,pad_left,pad_right, pad_top, pad_bottom)
       netS_img=pad_tensor_back(netS_img[-1], pad_left, pad_right, pad_top, pad_bottom)
       netR_img=pad_tensor_back(netR_img[-1], pad_left, pad_right, pad_top, pad_bottom)
       msbdn_ori_img= pad_tensor_back(msbdn_ori_img, pad_left, pad_right, pad_top, pad_bottom)
       netSSLD_Dehazing_img = pad_tensor_back(netSSLD_Dehazing_img, pad_left, pad_right, pad_top, pad_bottom)
       psd_img = pad_tensor_back(psd_img, pad_left, pad_right, pad_top, pad_bottom)
       dehazing_img = pad_tensor_back(dehazing_img, pad_left, pad_right, pad_top, pad_bottom)
       psd_img_new = (psd_img-0.5)/0.5
       msbdn_ori_img_new= (msbdn_ori_img-0.5)/0.5
    if opt.test_type == 'syn':

        ts = torch.cat([haze_img, netS_img,netR_img, msbdn_ori_img_new, netSSLD_Dehazing_img, psd_img_new, dehazing_img,gt_img], 3)
        ts_img = tensor2im(ts)
        ts_img = Image.fromarray(ts_img)
        ts_img.save(os.path.join(opt.out_dir, 'all_'+ img))

        gt_img = tensor2im(gt_img)
        gt_img = Image.fromarray(gt_img)
        gt_img.save(os.path.join(opt.out_dir, 'gt_' + img))

        haze_img = tensor2im(haze_img)
        haze_img = Image.fromarray(haze_img)
        haze_img.save(os.path.join(opt.out_dir, 'haze_' + img))

        netS_img = tensor2im(netS_img)
        netS_img = Image.fromarray(netS_img)
        netS_img.save(os.path.join(opt.out_dir, 'netS_'+img))


        netR_img = tensor2im(netR_img)
        netR_img = Image.fromarray(netR_img)
        netR_img.save(os.path.join(opt.out_dir, 'netR_' + img))
        #
        msbdn_ori_img = tensor2im2(msbdn_ori_img)
        msbdn_ori_img = Image.fromarray(msbdn_ori_img)
        msbdn_ori_img.save(os.path.join(opt.out_dir, 'msbdn_ori_'+ img))
        #
        ssld_img = tensor2im(netSSLD_Dehazing_img)
        ssld_img = Image.fromarray(ssld_img)
        ssld_img.save(os.path.join(opt.out_dir, 'ssid_'+ img))
        #
        psd_img = tensor2im2(psd_img)
        psd_img = Image.fromarray(psd_img)
        psd_img.save(os.path.join(opt.out_dir, 'psd_' + img))
        #
        dehazing_img = tensor2im(dehazing_img)
        dehazing_img = Image.fromarray(dehazing_img)
        dehazing_img.save(os.path.join(opt.out_dir, 'dehazing_' + img))

    else:

        ts = torch.cat([haze_img,  netR_img, msbdn_ori_img_new,netSSLD_Dehazing_img,psd_img_new,dehazing_img], 3)
        ts_img=tensor2im(ts)
        ts_img=Image.fromarray(ts_img)
        ts_img.save(os.path.join(opt.out_dir, 'all_' + img))

        haze_img = tensor2im(haze_img)
        haze_img = Image.fromarray(haze_img)
        haze_img.save(os.path.join(opt.out_dir, 'haze_' + img))

        netS_img = tensor2im(netS_img)
        netS_img = Image.fromarray(netS_img)
        netS_img.save(os.path.join(opt.out_dir, 'netS_' + img))

        netR_img = tensor2im(netR_img)
        netR_img = Image.fromarray(netR_img)
        netR_img.save(os.path.join(opt.out_dir, 'netR_'+img))

        #
        msbdn_ori_img = tensor2im2(msbdn_ori_img)
        msbdn_ori_img = Image.fromarray(msbdn_ori_img)
        msbdn_ori_img.save(os.path.join(opt.out_dir, 'msbdn_ori_'+ img))
        #
        ssld_img = tensor2im(netSSLD_Dehazing_img)
        ssld_img = Image.fromarray(ssld_img)
        ssld_img.save(os.path.join(opt.out_dir, 'ssid_'+ img))
        #
        psd_img = tensor2im2(psd_img)
        psd_img = Image.fromarray(psd_img)
        psd_img.save(os.path.join(opt.out_dir, 'psd_' + img))
        #
        dehazing_img = tensor2im(dehazing_img)
        dehazing_img = Image.fromarray(dehazing_img)
        dehazing_img.save(os.path.join(opt.out_dir, 'dehazing_' + img))

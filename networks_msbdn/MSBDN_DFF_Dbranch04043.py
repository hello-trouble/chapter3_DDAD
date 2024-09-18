import torch
import torch.nn as nn
import torch.nn.functional as F
from networks_msbdn.base_networks import Encoder_MDCBlock1, Decoder_MDCBlock1

def make_model(args, parent=False):
    return Net()

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class ResidualBlock_new(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_new, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = torch.add(out, residual)
        return out
class ResidualDAttentionBlock(torch.nn.Module):
    def __init__(self,channels,reduction):
        super(ResidualDAttentionBlock,self).__init__()
        self.conv1 = ConvLayer(channels,channels,kernel_size=3,stride=1)
        self.conv2 = ConvLayer(channels,channels,kernel_size=3,stride=1)
        self.relu = nn.PReLU()
        self.db_atten =DBatten(channels, reduction)
    # def foward(self):
    #     pass
    def forward(self,z,data_type):
        residual =z
        y=self.relu(self.conv1(z))
        y= self.conv2(y)
        atten = self.db_atten.forward_datatype(y,data_type)
        out=residual + atten
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class DBatten(nn.Module):
    def __init__(self, inplanes, ratio):
        super(DBatten, self).__init__()
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)

        # self.width =width_size
        self.conv_h = nn.Conv2d(inplanes, self.planes, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(inplanes, self.planes, kernel_size=1, stride=1, padding=0)
        self.conv_s = nn.Conv2d(self.planes, self.inplanes, kernel_size=1, stride=1, padding=0)
        # self.conv1_s = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=0)
        self.conv_t = nn.Conv2d(self.planes, self.inplanes, kernel_size=1, stride=1, padding=0)
        # self.conv1_t = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1, stride=1, padding=0)
        # self.layerNormh = nn.BatchNorm2d(self.planes)  # 64
        # self.layerNormw = nn.BatchNorm2d(self.planes)  # 128
        # self.layerNorm = nn.BatchNorm2d(inplanes)  # 64 128
        self.reLU = nn.ReLU(inplace=True)  # yapf: disable
        self.poolh = nn.AdaptiveAvgPool2d((None, 1))
        self.max_poolh = nn.AdaptiveMaxPool2d((None,1))
        self.poolw = nn.AdaptiveAvgPool2d((1, None))
        self.max_poolw= nn.AdaptiveMaxPool2d((1,None))
        self.sigmoid = nn.Sigmoid()
        self.act = h_swish()


    def channel_shuffle(self,x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # reshape
        # b, c, h, w =======>  b, g, c_per, h, w
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x
    def forward_datatype(self, x,data_type):
        residual = x
        x_h_avg = self.poolh(x)  # [batch_size, c, h, 1]
        x_h_max = self.max_poolh(x)
        x_h = x_h_avg + x_h_max

        x_w_avg = self.poolw(x)  # [batch_size, c, 1, w]
        x_w_max = self.max_poolw(x)
        x_w = x_w_avg + x_w_max


        # x_h = self.poolh(x)
        # x_w1 = self.poolw(x)  # [batch_size, c, 1, w]
        # x_w = x_w1.permute(0, 1, 3, 2)  # [batch_size, c, w, 1]
        x_w = x_w.permute(0, 1, 3, 2)  # [batch_size, c, w, 1]
        x_h = self.conv_h(x_h)
        x_h = self.act(x_h)
        x_w = self.conv_w(x_w)
        x_w = self.act(x_w)
        attention_w = F.normalize(x_w, p=2, dim=1, eps=1e-12)
        attention_w = attention_w.permute(0, 1, 3, 2)  # [batch_size, c, 1, w]
        y = x_h * attention_w
        if data_type=='syn':
            src=  self.conv_s(y)
            src = self.channel_shuffle(src,16)
            # src = self.act(src)
            # src = self.conv1_s(src)
            atten= self.sigmoid(src)
        elif data_type=='real_syn':
            trg = self.conv_t(y)
            trg = self.channel_shuffle(trg,16)
            # trg = self.act(trg)
            # trg = self.conv1_t(trg)
            atten = self.sigmoid(trg)
        elif data_type=='real':
            trg = self.conv_t(y)
            trg = self.channel_shuffle(trg,16)
            # trg = self.act(trg)
            # trg = self.conv1_t(trg)
            atten= self.sigmoid(trg)
        out=atten*residual
        return out

class Net(nn.Module):
    def __init__(self, res_blocks=18):
        super(Net, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)

        self.dense0 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.dense0_t= ResidualBlock_new(16)

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.fusion1 = Encoder_MDCBlock1(32, 2, mode='iter2')
        self.dense1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.dense1_t =ResidualBlock_new(32)
        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.fusion2 = Encoder_MDCBlock1(64, 3, mode='iter2')
        self.dense2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.dense2_t = ResidualBlock_new(64)

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.fusion3 = Encoder_MDCBlock1(128, 4, mode='iter2')
        self.dense3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.fusion4 = Encoder_MDCBlock1(256, 5, mode='iter2')

        # self.dehaze = nn.Sequential()
        # for i in range(0, res_blocks):
        #     self.dehaze.add_module('res%d' % i,ResidualDAttentionBlock(256, reduction=32))
        self.dehaze_1 =  ResidualBlock(256)
        self.dehaze_2 =  ResidualBlock(256)
        self.dehaze_3 =  ResidualDAttentionBlock(256, reduction=16)
        self.dehaze_4 =  ResidualBlock(256)
        self.dehaze_5 =  ResidualBlock(256)
        self.dehaze_6 =  ResidualDAttentionBlock(256, reduction=16)
        self.dehaze_7 =  ResidualBlock(256)
        self.dehaze_8 =  ResidualBlock(256)
        self.dehaze_9 =  ResidualDAttentionBlock(256, reduction=16)
        self.dehaze_10 = ResidualBlock(256)
        self.dehaze_11 = ResidualBlock(256)
        self.dehaze_12 = ResidualDAttentionBlock(256, reduction=16)
        self.dehaze_13 = ResidualBlock(256)
        self.dehaze_14 = ResidualBlock(256)
        self.dehaze_15 = ResidualDAttentionBlock(256, reduction=16)
        self.dehaze_16 = ResidualBlock(256)
        self.dehaze_17 = ResidualBlock(256)
        self.dehaze_18 = ResidualDAttentionBlock(256, reduction=16)

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.fusion_4 = Decoder_MDCBlock1(128, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.fusion_3 = Decoder_MDCBlock1(64, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.fusion_2 = Decoder_MDCBlock1(32, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.fusion_1 = Decoder_MDCBlock1(16, 5, mode='iter2')
        # self.conv_output8= ConvLayer(128,3,kernel_size=3,stride=1)
        # self.conv_output4 = ConvLayer(64, 3, kernel_size = 3, stride = 1)
        # self.conv_output2 = ConvLayer(32,3, kernel_size = 3, stride =1)
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)
        self.act=nn.Tanh()


    def forward(self, x,data_type):

        res1x = self.conv_input(x)
        feature_mem = [res1x]
        if data_type =='syn' or data_type=='real_syn':
            x = self.dense0(res1x) + res1x
        elif data_type =='real':
            x = self.dense0_t(self.dense0(res1x)) +res1x
        results=[x]

        res2x = self.conv2x(x)
        res2x = self.fusion1(res2x, feature_mem)
        feature_mem.append(res2x)
        #origin_code  res2x =self.dense1(res2x) + res2x
        if data_type =='syn' or data_type=='real_syn':
            res2x = self.dense1(res2x) + res2x
        elif data_type =='real':
            res2x = self.dense1_t(self.dense1(res2x)) + res2x
        results.append(res2x)

        res4x =self.conv4x(res2x)
        res4x = self.fusion2(res4x, feature_mem)
        feature_mem.append(res4x)
        #origin_code res4x = self.dense2(res4x) + res4x
        if data_type =='syn'or data_type=='real_syn':
            res4x = self.dense2(res4x) + res4x
        elif data_type =='real':
            res4x = self.dense2_t(self.dense2(res4x)) + res4x
        results.append(res4x)

        res8x = self.conv8x(res4x)
        res8x = self.fusion3(res8x, feature_mem)
        feature_mem.append(res8x)
        ###########################3
        #############################
        #################################
        #####################################333
        ############################################
        ##################################################
        ###########################################################3   problem here
        #original_code res8x = self.dense3(res8x) + res8x
        # res8x_1 = self.dense3(res8x)
        # ###################
        # res8x = self.dense3_db(res8x_1,data_type)  + res8x

        res8x = self.dense3(res8x) + res8x


        res16x = self.conv16x(res8x)
        res16x = self.fusion4(res16x, feature_mem)
        res_dehaze = res16x #torch.Size([1, 256, 16, 16])
        in_ft = res16x*2

        res16x = self.dehaze_1(in_ft)
        res16x = self.dehaze_2(res16x)
        res16x = self.dehaze_3(res16x, data_type)
        res16x = self.dehaze_4(res16x)
        res16x = self.dehaze_5(res16x)
        res16x = self.dehaze_6(res16x, data_type)
        res16x = self.dehaze_7(res16x)
        res16x = self.dehaze_8(res16x)
        res16x = self.dehaze_9(res16x, data_type)
        res16x = self.dehaze_10(res16x)
        res16x = self.dehaze_11(res16x)
        res16x = self.dehaze_12(res16x, data_type)
        res16x = self.dehaze_13(res16x)
        res16x = self.dehaze_14(res16x)
        res16x = self.dehaze_15(res16x, data_type)
        res16x = self.dehaze_16(res16x)
        res16x = self.dehaze_17(res16x)
        res16x = self.dehaze_18(res16x, data_type)+ in_ft - res_dehaze
        feature_mem_up = [res16x] #torch.Size([1, 256, 16, 16])


        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        res8x = self.fusion_4(res8x, feature_mem_up)
        feature_mem_up.append(res8x) #torch.Size([1, 128, 32, 32])

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x = self.fusion_3(res4x, feature_mem_up)
        feature_mem_up.append(res4x) #torch.Size([1, 64, 64, 64])

        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x 
        res2x = self.fusion_2(res2x, feature_mem_up)
        feature_mem_up.append(res2x) #torch.Size([1, 32, 128, 128])

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x 
        x = self.fusion_1(x, feature_mem_up) #torch.Size([1, 16, 256, 256])

        x = self.conv_output(x)
        x = self.act(x)
        results.append(x)
        return results

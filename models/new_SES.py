import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
#from utils import save_net,load_net
from models.cbam_model import ChannelAttention,SpatialAttention
from models.deform_conv import DeformConv2D


from models.ses_conv import SESMaxProjection
from models.ses_conv import SESConv_Z2_H, SESConv_H_H


class MNIST_SES_Scalar(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0], basis_type='A', ses_channels = [32,63,95,128]):
        super().__init__()
        #C1, C2, C3 = 32, 63, 95
        C1 = ses_channels[0]
        C2 = ses_channels[1]
        C3 = ses_channels[2]
        C4 = ses_channels[3]
        
        
            
        self.main1 = nn.Sequential(
            SESConv_Z2_H(3, C1, kernel_size, 3, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type),
            SESMaxProjection(),
            nn.ReLU(True),
            #nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),
            
            SESConv_Z2_H(C1, C1, kernel_size, 3, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type),
            SESMaxProjection(),
            nn.ReLU(True),
        )
        
        self.main2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),
            SESConv_Z2_H(C1, C2, kernel_size, 3, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type),
            SESMaxProjection(),
            nn.ReLU(True),
            #nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),
            SESConv_Z2_H(C2, C2, kernel_size, 3, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type),
            SESMaxProjection(),
            nn.ReLU(True),
            
        )
        self.main3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),
            SESConv_Z2_H(C2, C3, kernel_size, 3, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type),
            SESMaxProjection(),
            nn.ReLU(True),
            #nn.MaxPool2d(pool_size, padding=2),
            #nn.MaxPool2d(2),
            nn.BatchNorm2d(C3),
            SESConv_Z2_H(C3, C3, kernel_size, 3, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type),
            SESMaxProjection(),
            nn.ReLU(True),
            #nn.MaxPool2d(pool_size, padding=2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C3),
        )
        self.main4 = nn.Sequential(
            SESConv_Z2_H(C3, C4, kernel_size, 3, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.BatchNorm2d(C4),
            SESConv_Z2_H(C4, C4, kernel_size, 3, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type),
            SESMaxProjection(),
            nn.ReLU(True),
            nn.BatchNorm2d(C4),
        )


    def forward(self, x):
        x1 = self.main1(x)
        
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        x3 = self.main4(x3)
        #x = x.view(x.size(0), -1)
        #x = self.linear(x)
        return x1,x2,x3


class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            #torch.nn.LeakyReLU(negative_slope=0.1,inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #torch.nn.Dropout(0.5),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )

        self.relu=torch.nn.ReLU(inplace=True)

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)
class ASRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(ASRNet, self).__init__()
        
        ####scale equivariance    I have not added dropout in the network
        self.ses_channels = [32,64,128,256]
        ses_num_scales = 4
        ses_factor = 2.0##3.0
        ses_min_scale = 1.2##1.5
        ses_mult = 1.4   ##now I need to change it in ses_basis
        ses_max_order = 4 ##now I need to change it in ses_basis
        ses_size = 7 ###13
        ses_dropout = 0.7
        ses_q = ses_factor ** (1 / (ses_num_scales - 1))
        ses_scales = [ses_min_scale * ses_q**i for i in range(ses_num_scales)]
        ses_scales = [round(s, 2) for s in ses_scales]
        self.ses = MNIST_SES_Scalar(pool_size=2, kernel_size=ses_size, scales=ses_scales,
                             basis_type='B', ses_channels=self.ses_channels)

        ###########backbone     
        self.seen = 0
        ## frontend feature extraction
        #self.frontend_feat1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]
        #self.frontend_feat1 = [64, 64]
        #self.frontend_feat2 = ['M', 128, 128]
        #self.frontend_feat3 = ['M', 256, 256, 256, 'M', 512, 512, 512]
        self.mid_feat  = [512,512,512]

        '''
        # CBAM module (convolution block attention module)
        cite as "CBAM: Convolutional Block Attention Module, 2018 ECCV"
        '''
        self.planes = self.ses_channels[-1]
        #self.planes = self.ses_channels[-1]
        self.ca = ChannelAttention(self.planes)
        self.sa = SpatialAttention()

        '''
        dilation convolution (Spatial Pyramid Module)
        cite as "Scale Pyramid Network for Crowd Counting, 2019 WACV"
        '''
        self.conv4_3_1 = nn.Conv2d(self.planes, 512, kernel_size=3, padding=2, dilation=2)
        self.conv4_3_2 = nn.Conv2d(self.planes, 512, kernel_size=3, padding=4, dilation=4)
        self.conv4_3_3 = nn.Conv2d(self.planes, 512, kernel_size=3, padding=8, dilation=8)
        self.conv4_3_4 = nn.Conv2d(self.planes, 512, kernel_size=3, padding=12, dilation=12)
        # self.conv4 = [self.conv4_3_1, self.conv4_3_2, self.conv4_3_3, self.conv4_3_4]
        self.conv5 = nn.Conv2d(2048, 512, kernel_size=1)
        self.drop1 = nn.Dropout(0.3)
        '''
        convolution layers
        '''
        self.mid_end = make_layers(self.mid_feat,in_channels = 512,)
        self.drop2 = nn.Dropout(0.4)

        '''
        deformable convolution network
        cite as "Deformable Convolutional Networks, 2017 ICCV"
        '''
        self.offset1 = nn.Conv2d(512, 18, kernel_size=3, padding=1)
        self.conv6_1 = DeformConv2D(512, 256, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(256)
        self.drop3 = nn.Dropout(0.3)

        self.offset2 = nn.Conv2d(256, 18, kernel_size=3, padding=1)
        self.conv6_2 = DeformConv2D(256, 128, kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)
        self.drop4 = nn.Dropout(0.3)

        self.offset3 = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv6_3 = DeformConv2D(128, 64, kernel_size=3, padding=1)
        self.bn6_3 = nn.BatchNorm2d(64)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        self.sr_decoder = build_sr_decoder('vggnet',nn.BatchNorm2d)
        
        
        
        

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            mm = 0
            #for sub_model in [self.frontend1, self.frontend2, self.frontend3]:
                #print('mm', mm)
                #for i in range(len(sub_model.state_dict().items())):
                #    print(len(sub_model.state_dict().items()))
                #    list(sub_model.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i+mm][1].data[:]
                #mm += len(sub_model.state_dict().items())
            #for i in range(len(self.frontend.state_dict().items())):
            #    list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        f1, f2, x = self.ses(x)### 32 63 95
        
        #print(f1.shape)
        #print(f2.shape)
        #print(x.shape)
        residual = x
        x = self.ca(x) * x
        x = self.sa(x) * x
        x += residual#### += will not change the cannel numbers.
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
        x1 = self.conv4_3_1(x)
        x2 = self.conv4_3_2(x)
        x3 = self.conv4_3_2(x)
        x4 = self.conv4_3_2(x)
        x = torch.cat((x1, x2, x3, x4), 1) 
        x = self.conv5(x)
        #x = self.drop1(x)
        
        seperate_point = x###seperate point

        # x = self.backend(x)
        x = self.mid_end(x)
        #x = self.drop2(x)
        

        offset1 = self.offset1(x)
        x = F.relu(self.conv6_1(x, offset1))
        x = self.bn6_1(x)
        #x = self.drop3(x)

        offset2 = self.offset2(x)
        x = F.relu(self.conv6_2(x, offset2))
        x = self.bn6_2(x)
        #x = self.drop4(x)

        offset3 = self.offset3(x)
        x = F.relu(self.conv6_3(x, offset3))
        x = self.bn6_3(x)
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
        x = self.output_layer(x)
        
        
        ######sr
        x_sr= self.sr_decoder(f2, f1)
        
        
        
        
        return torch.abs(x), x_sr, f1, f2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1

    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)






class Decoder(nn.Module):
    def __init__(self, backbone, BatchNorm):
        super(Decoder, self).__init__()
            
        low_level_inplanes = 32 ##f1
        high_level_inplanes = 64 ##f2

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        self.c_input = high_level_inplanes + 48
        
        #self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
        self.last_conv = nn.Sequential(nn.Conv2d(self.c_input, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       #nn.Dropout(0.3),
                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.3),
                                       nn.Conv2d(128, 64, kernel_size=1, stride=1))
                                       
                                       
        self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2) 
        self.up_edsr_1 = EDSRConv(64,64)
        self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2) 
        self.up_edsr_2 = EDSRConv(32,32)
        #self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2) 
        #self.up_edsr_3 = EDSRConv(16,16)
        #self.up_sr_4 = nn.ConvTranspose2d(16, 16, 2, stride=2) 
        #self.up_edsr_4 = EDSRConv(16,16)
        self.up_conv_last = nn.Conv2d(32,3,1)
        
        
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        
        x_sr_up = self.up_sr_1(x)
        x_sr_up=self.up_edsr_1(x_sr_up)

        x_sr_up = self.up_sr_2(x_sr_up)
        x_sr_up=self.up_edsr_2(x_sr_up)
            
            

        #x_sr_up = self.up_sr_3(x_sr_up)
        #x_sr_up=self.up_edsr_3(x_sr_up)
            
            #
            
        #x_sr_up = self.up_sr_4(x_sr_up)
        #x_sr_up=self.up_edsr_4(x_sr_up)
            
            
        x_sr_up=self.up_conv_last(x_sr_up)

        return x_sr_up
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_sr_decoder(backbone, BatchNorm):
    return Decoder(backbone, BatchNorm)
    
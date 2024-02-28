import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
#from utils import save_net,load_net
from models.cbam_model import ChannelAttention,SpatialAttention
from models.deform_conv import DeformConv2D
import time
import numpy as np
from models.FourierEncoding import (BasicEncoding, PositionalEncoding, GaussianEncoding, PosEncoding)

from models.siren import Siren

from torch.autograd import Variable


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



class New_bay_Net(nn.Module):
    def __init__(self, downsample_ratio,input_size, load_weights=False):
        super(New_bay_Net, self).__init__()
        self.seen = 0
        self.downsample_ratio = downsample_ratio
        self.input_size = input_size
        
        self.frontend_feat1 = [64, 64]
        self.frontend_feat2 = ['M', 128, 128]
        self.frontend_feat3 = ['M', 256, 256, 256]
        self.frontend_feat4 = ['M', 512, 512, 512]
        
        self.ASRNet = ASRNet()
        
        
        self.m = 64 ## pos = [b 1024 4*m]  
        self.z_feature_size = 16   ##z = [b, 4*m, zf,zf]  
        
        
        ####sr spatial vae
        self.conv_z1 = nn.Conv2d(self.frontend_feat2[-1], 1, 1, bias=False)
        self.conv_z2 = nn.Conv2d(self.frontend_feat3[-1], 1, 1, bias=False)
        #self.relu_z1 = nn.ReLU(inplace=True)
        
        self.sr_decoder = build_sr_decoder('vggnet',nn.BatchNorm2d, 16)
        
        
        ####cc spatial vae
        #self.conv_c1 = nn.Conv2d(self.frontend_feat4[-1], 1, 1, bias=False)
        
        #self.Encoder2z = Encoder2z(self.downsample_ratio, self.m + 2*4 + 1, self.z_feature_size)
        
        #self.Encoder2z = Encoder2z(self.downsample_ratio, 8+3*(4*self.m)+4, self.z_feature_size)
        self.pos_encode_layer = PositionalEncoding(0.5, self.m) ##0.5
        #self.pos_encode_layer = PosEncoding(2, self.m)
        #self.pos_encode_layer = GaussianEncoding(10.0, 2, self.m)
        pos_out_dim = 2*2*self.m
        #pos_out_dim = 2*self.m
        
        weight_dim = pos_out_dim + 3*self.z_feature_size*self.z_feature_size + 4
        #weight_dim = pos_out_dim + 4*self.z_feature_size*self.z_feature_size + 5
        
        self.Encoder2z = Encoder2z(self.input_size, weight_dim, self.z_feature_size)
        
        #self.pos_encode_layer = PositionalEncoding(0.5, 2)
        #self.pos_encode_layer = BasicEncoding()
        
        self.cc_decoder = build_cc_decoder(self.z_feature_size, self.m, pos_out_dim)   #### output_layer's size
        
        #self.linear_c = nn.Linear(2, 2, bias=False)
        
        self.kl_div = 0
        
        #self.grad_layer = GradLayer()


    def forward(self, x, grid_c, grid_sr, use_sr, mode, epoch):
        
        x,f1,f2 = self.ASRNet(x)  ## [16 1 32 32]
        
        #print(grid_c.shape, grid_sr.shape)[16, 1024, 2])
        
        
        if use_sr:
        
            f2_ = self.conv_z1(f2)##32 32 1
            f3_ = self.conv_z2(f3)##16 16 1
            f2_ = F.interpolate(f2_, size=f3_.size()[2:], mode='bilinear', align_corners=True)
            z = torch.cat((f2_, f3_), dim=1)
            #z = self.relu_z1(z)
        
            x_sr= self.sr_decoder(z, grid_sr)
            #print('3', time.time())
        
        
            #return torch.abs(x), x_sr, f1, f2
            return x, f1, f2
        else:
            #print(x.shape) 
            #grid_c = self.linear_c(grid_c)
            grid_c = self.pos_encode_layer(grid_c)
            #print(grid_c.shape) ##[16, 1024, 40]
            f1 = x
            x = self.Encoder2z(x)
            f2 = x
            #print(x.shape) ##[16 1 8 8] ### [16 64 16 16]
            
            if mode == 'train': #and epoch>500:
                mod = True
            else:
                mod = False
                

            
            
            x, out_sigma, kl_div = self.cc_decoder(x, grid_c, mod) #[16, 1024]
            #print(x.shape)
            self.kl_div = kl_div
            
            #grad = self.grad_layer(x)
        
            #return torch.abs(x), f1, f2
            return x, f1, f2, out_sigma
        

class ASRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(ASRNet, self).__init__()
        self.seen = 0
        
        
        
        
        ## frontend feature extraction
        #self.frontend_feat1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]
        self.frontend_feat1 = [64, 64]
        self.frontend_feat2 = ['M', 128, 128]
        self.frontend_feat3 = ['M', 256, 256, 256]
        self.frontend_feat4 = ['M', 512, 512, 512]
        self.mid_feat  = [512,512,512]
        
        self.resnet_backbone = Res_Backbone()
        
        '''
        # CBAM module (convolution block attention module)
        cite as "CBAM: Convolutional Block Attention Module, 2018 ECCV"
        '''
        self.planes = self.frontend_feat4[-1]
        self.ca = ChannelAttention(self.planes)
        self.sa = SpatialAttention()

        '''
        dilation convolution (Spatial Pyramid Module)
        cite as "Scale Pyramid Network for Crowd Counting, 2019 WACV"
        '''
        self.conv4_3_1 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv4_3_2 = nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4)
        self.conv4_3_3 = nn.Conv2d(512, 512, kernel_size=3, padding=8, dilation=8)
        self.conv4_3_4 = nn.Conv2d(512, 512, kernel_size=3, padding=12, dilation=12)
        # self.conv4 = [self.conv4_3_1, self.conv4_3_2, self.conv4_3_3, self.conv4_3_4]
        self.conv5 = nn.Conv2d(2048, 512, kernel_size=1)
        #self.drop1 = nn.Dropout(0.3)
        '''
        convolution layers
        '''
        self.mid_end = make_layers(self.mid_feat,in_channels = 512,)
        #self.drop2 = nn.Dropout(0.4)

        '''
        deformable convolution network
        cite as "Deformable Convolutional Networks, 2017 ICCV"
        '''
        self.offset1 = nn.Conv2d(512, 18, kernel_size=3, padding=1)
        self.conv6_1 = DeformConv2D(512, 256, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(256)
        #self.drop3 = nn.Dropout(0.3)

        self.offset2 = nn.Conv2d(256, 18, kernel_size=3, padding=1)
        self.conv6_2 = DeformConv2D(256, 128, kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)
        #self.drop4 = nn.Dropout(0.3)

        self.offset3 = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv6_3 = DeformConv2D(128, 64, kernel_size=3, padding=1)
        self.bn6_3 = nn.BatchNorm2d(64)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        self.output_layer2 = nn.Conv2d(512, 512, kernel_size=1)
        
        
        

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()       
            for i in range(len(self.resnet_backbone.state_dict().items())):
                list(self.resnet_backbone.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self, x):
        #print('1', time.time())
        #x = self.frontend1(x)
        x, f1, f2, f3 = self.resnet_backbone(x)
        #x = self.drop1(x)
        #print(x.shape)
        residual = x
        x = self.ca(x) * x
        x = self.sa(x) * x
        x += residual
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
        x1 = self.conv4_3_1(x)
        x2 = self.conv4_3_2(x)
        x3 = self.conv4_3_2(x)
        x4 = self.conv4_3_2(x)
        x = torch.cat((x1, x2, x3, x4), 1) 
        x = self.conv5(x)
        #x = self.drop2(x)
        
        seperate_point = x###seperate point

        if False:
            # x = self.backend(x)
            #x = self.mid_end(x)
            for i in range(len(self.mid_end)):
                x = self.mid_end[i](x)
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
            
            #return torch.abs(x), f1, f2
        else:    
        
            x = self.output_layer2(x)
            
            
            #return x, f1, f2
            
            
        
        return torch.abs(x), f1, f2
        #return x, f1, f2


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class Swish(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super(Swish, self).__init__()
        
        self.beta = nn.Parameter(torch.tensor([0.5]))


    def forward(self, input):
        """
        Forward pass of the function.
        """
        #return input * torch.sigmoid(input)
        return (input * torch.sigmoid_(input * F.softplus(self.beta))).div_(1.1)

class ResLinear(nn.Module):
    def __init__(self, n):
        """
        Init method.
        """
        super(ResLinear, self).__init__()
        self.linear = nn.Linear(n, n, bias=False)##146 = 12*12 + 2
        #self.act = Swish()
        self.act = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        """
        Forward pass of the function.
        """
        return self.act(self.linear(x)) + x
 

class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(ResidLinear, self).__init__()

        #self.linear = nn.Linear(n_in, n_out, bias=False)
        self.linear = nn.Linear(n_in, n_out)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.linear(x) + x)
        




class Decoder(nn.Module):
    def __init__(self, backbone, BatchNorm, feature_size):
        super(Decoder, self).__init__()
        
            
        n_features = 2*feature_size*feature_size+2
        #inplanes = inplanes
        #self.conv1 = nn.Conv2d(inplanes, 1, 1, bias=False)
        
        self.res1 = ResLinear(n_features)
        self.res2 = ResLinear(n_features)
        self.last = nn.Linear(n_features, 3)
        
        


    def forward(self, x1, x2):### x1:z x2:coordinates

        #x1 = self.conv1(x1) 
        
        b, n_query_pts = x2.shape[0], x2.shape[1]
        
        x = torch.cat([torch.unsqueeze(x1.view(b, -1), dim=1).repeat(1, n_query_pts, 1), x2], dim=2)
        return torch.squeeze(self.last(self.res2(self.res1(x))))
        
        
        


def build_sr_decoder(backbone, BatchNorm, feature_size):
    return Decoder(backbone, BatchNorm, feature_size)
    



class Res_Backbone(nn.Module):
    def __init__(self, load_weights=False):
        super(Res_Backbone, self).__init__()
        
        self.seen = 0
        ## frontend feature extraction
        #self.frontend_feat1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]
        self.frontend_feat1 = [64, 64]
        self.frontend_feat2 = ['M', 128, 128]
        self.frontend_feat3 = ['M', 256, 256, 256]
        self.frontend_feat4 = ['M', 512, 512, 512]
        self.mid_feat  = [512,512,512]
        self.frontend1 = make_layers_2(self.frontend_feat1)
        self.frontend2 = make_layers_2(self.frontend_feat2, in_channels = 64)
        self.frontend3 = make_layers_2(self.frontend_feat3, in_channels = 128)
        self.frontend4 = make_layers_2(self.frontend_feat4, in_channels = 256)
        
        
        
    def forward(self, x):
        x = self.frontend1(x)
        #for i in range(len(self.frontend1)):
        #    x = self.frontend1[i](x)
        
        f1 = x
        
        x = self.frontend2(x)
        #for i in range(len(self.frontend2)):
        #    x = self.frontend2[i](x)
        
        f2 = x
        x = self.frontend3(x)
        #for i in range(len(self.frontend3)):
        #    x = self.frontend3[i](x)
        f3 = x
        x = self.frontend4(x)
        
        
        
        return x, f1, f2, f3

    
class Encoder2z(nn.Module):
    def __init__(self, input_size, m_size, z_feature_size, load_weights=False):
        super(Encoder2z, self).__init__()
        
        self.seen = 0

        self.m_size = m_size
        self.z_feature_size = z_feature_size
        
        self.ratio = int(np.log2(input_size/8 / self.z_feature_size)) #### input size should be changed
        self.frontend_feat = []
        
        for i in range(self.ratio):
            #self.frontend_feat += ['M',64]
            self.frontend_feat += ['M',self.m_size]
            #self.frontend_feat += [16]
        #self.frontend_feat += [64,self.m_size]  ## [8 8 m]
        self.frontend_feat += [self.m_size,self.m_size]  ## [8 8 m]
        #self.frontend_feat += [self.m_size,self.m_size]
        
       
        
            
            
        self.frontend = make_layers_3(self.frontend_feat, in_channels = 512, batch_norm = True)
        #self.conv = nn.Conv2d(16, 1, 1, bias=False)
        
        #self.bn = nn.BatchNorm2d(self.m_size)
        #self.res1 = ResidLinear(2,2)
        #self.res2 = nn.Linear(2, 1)
        #self.res3 = nn.Linear(1024, 1024)
        
        self._initialize_weights() 
        
    def forward(self, x):
        #print(x.shape)
        x = self.frontend(x)
        #x = self.bn(x)
        #x = self.conv(x)
        #print(x.shape)
        
        #b = x.shape[0]
        #x = x.reshape([b, -1, 2])
        #x = self.res1(x)
        #x = self.res2(x)
        #x = self.res3(x)
        #print(x.shape)
        #x = x.reshape([b, 1,32,32])
        #x = torch.squeeze(x)
        
        #print(x.shape)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.001)
        
class CC_Decoder(nn.Module):
    def __init__(self, feature_size, m_size, pos_out_dim):
        super(CC_Decoder, self).__init__()
        self.m_size = m_size
        self.n_features = int(feature_size*feature_size)
        self.pos_dim = pos_out_dim
        
        self.weight_dim = self.pos_dim + 3*self.n_features + 4
        
        #n_features = feature_size*feature_size#+2*2*m_size
        #inplanes = inplanes
        #self.conv1 = nn.Conv2d(inplanes, 1, 1, bias=False)
        
        self.res1 = ResidLinear(self.n_features, self.n_features)
        self.res2 = ResidLinear(self.n_features, self.n_features)
        self.res3 = ResidLinear(self.n_features, self.n_features)
        #self.res4 = ResidLinear(self.n_features, self.n_features)
        #self.res5 = ResidLinear(self.n_features, self.n_features)
        self.last1 = nn.Linear(self.n_features, 1)
        #self.last2 = nn.Linear(self.n_features, 1)
        self.last2 = torch.nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.PReLU(),
            nn.Linear(self.n_features, 1),
            )
        
        self.act = nn.PReLU()#nn.SiLU()
        #self.act = nn.LeakyReLU()
        #self.act = nn.Tanh()
        
        #self.map_cor1 = nn.Linear(4*self.m_size, 4*self.m_size, bias=False)
        #self.map_cor2 = nn.Linear(4*self.m_size, 4*self.m_size)
        #self.map_act = nn.Tanh()
        
        self.act1 = nn.PReLU()#nn.Tanh()#nn.LeakyReLU()
        self.act2 = nn.PReLU()#nn.Tanh()#nn.LeakyReLU()
        self.act3 = nn.PReLU()#nn.Tanh()#nn.LeakyReLU()
        self.act4 = nn.PReLU()#nn.Tanh()#nn.LeakyReLU()
        self.act5 = nn.LeakyReLU()
        
        self.act6 = nn.LeakyReLU()
        
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        
        self.learn_prior = Variable(torch.rand(1).type(torch.FloatTensor), requires_grad=True).cuda()
        

        #self.bn1 = nn.BatchNorm1d(self.n_features)
        #self.bn2 = nn.BatchNorm1d(self.n_features)
        #self.bn3 = nn.BatchNorm1d(self.n_features)
        #self.bn4 = nn.BatchNorm1d(self.n_features)
        
        self.W_fine = nn.Linear(self.n_features, self.n_features)
        #self.W_fine2 = nn.Linear(self.weight_dim, self.weight_dim)
        
        self._initialize_weights() 
        self.omega_0 = 30.0
        self.kl = 0
        
        #with torch.no_grad():
        #    self.last.weight.uniform_(-np.sqrt(6 / self.n_features) / self.omega_0, 
        #                                      np.sqrt(6 / self.n_features) / self.omega_0)
        

    def forward(self, x1, x2, mode):### x1:z x2:coordinates

        #x1 = self.conv1(x1) 

        
        b, n_query_pts = x2.shape[0], x2.shape[1] #x1 [b 1 8 8 ]x2 [b 1024 2*2*m]
        
        #x2 = self.map_cor1(x2)
        #x2 = self.map_cor2(x2)
        #x2 = self.map_act(x2)
        

        #x = torch.cat([torch.unsqueeze(x1.view(b, -1), dim=1).repeat(1, n_query_pts, 1), x2], dim=2)
        #x = torch.unsqueeze(x1.view(b, -1), dim=1).repeat(1, n_query_pts, 1)
        #print(x1.shape)
        #W1 = torch.reshape(x1[:, :self.m_size], (b, self.m_size*2*2, self.n_features)) 
        #W2 = torch.reshape(x1[:, self.m_size:self.m_size+4], (b, self.n_features, self.n_features)) 
        #W3 = torch.reshape(x1[:, self.m_size+4:self.m_size+8], (b, self.n_features, self.n_features)) 
        #W4 = torch.reshape(x1[:, self.m_size+8:], (b, self.n_features, 4))
        
        #weight_dim = self.pos_dim + 3*self.n_features + 4
        
        W = torch.reshape(x1, (b, self.weight_dim, self.n_features))
        W = self.W_fine(W)

        
        #W = self.bn2(W)
        #W = torch.reshape(x1, (b, -1, 1)) 
        #out = torch.einsum("bij, bjk -> bik", x2, W)  ##[b 1024 self.n_features]
        #out = torch.squeeze(self.act(self.last(self.res3(self.res2(self.res1(out))))))
        #out = torch.squeeze(self.act(self.last(self.res1(out))))
        ####experiment of hydranetwork
        
        W1 = W[:,:self.pos_dim,:] 
        b1 = W[:,self.pos_dim:self.pos_dim+1,:].repeat(1, n_query_pts, 1)
        
        
        W2 = W[:,(self.pos_dim+1):(self.pos_dim+self.n_features+1),:]
        b2 = W[:,(self.pos_dim+self.n_features+1):(self.pos_dim+self.n_features+2),:].repeat(1, n_query_pts, 1)
        #print(W1.shape, W2.shape,b1.shape,b2.shape)
        
        W3 = W[:,(self.pos_dim+self.n_features+2):(self.pos_dim+2*self.n_features+2),:]
        b3 = W[:,(self.pos_dim+2*self.n_features+2):(self.pos_dim+2*self.n_features+3),:].repeat(1, n_query_pts, 1)
        
        W4 = W[:,(self.pos_dim+2*self.n_features+3):(self.pos_dim+3*self.n_features+3),:]
        b4 = W[:,(self.pos_dim+3*self.n_features+3):(self.pos_dim+3*self.n_features+4),:].repeat(1, n_query_pts, 1)
        
        #W1 = torch.transpose(W1,1,2)
        #W2 = torch.transpose(W2,1,2)
        #W3 = torch.transpose(W3,1,2)
        #W4 = torch.transpose(W4,1,2)
        
        #W5 = W[:,(self.pos_dim+3*self.n_features+4):(self.pos_dim+4*self.n_features+4),:]
        #b5 = W[:,(self.pos_dim+4*self.n_features+4):(self.pos_dim+4*self.n_features+5),:].repeat(1, n_query_pts, 1)
        #W5 = W[:,(self.pos_dim+3*self.n_features+4):(self.pos_dim+3*self.n_features+5),:]
        #W5 = torch.transpose(W5,1,2)
        
        #W1 = W[:,:self.m_size*2*2,:]
        #b1 = W[:,self.m_size*2*2:self.m_size*2*2+1,:].repeat(1, n_query_pts, 1)
        
        
        #W2 = W[:,(self.m_size*2*2+1):(2*self.m_size*4+1),:]
        #b2 = W[:,(2*self.m_size*2*2+1):(2*self.m_size*4+2),:].repeat(1, n_query_pts, 1)
        #print(W1.shape, W2.shape,b1.shape,b2.shape)
        
        #W3 = W[:,(2*self.m_size*4+2):(3*self.m_size*4+2),:]
        #b3 = W[:,(3*self.m_size*4+2):(3*self.m_size*4+3),:].repeat(1, n_query_pts, 1)
        
        #W4 = W[:,(3*self.m_size*4+3):(4*self.m_size*4+3),:]
        #b4 = W[:,(4*self.m_size*4+3):(4*self.m_size*4+4),:].repeat(1, n_query_pts, 1)
        

        #W1 = self.bn1(W1)
        
        out1 = torch.einsum("bij, bjk -> bik", x2, W1) + b1  ##[b 1024 self.n_features]
        out1 = self.act1(out1)
        #out1 = torch.sin(self.omega_0 * out1)

        
        out2 = torch.einsum("bij, bjk -> bik", out1, W2) + b2
        out2 = self.act2(out2)+ out1
        #out2 = self.act2(out2+ out1)
        #out2 = torch.sin(self.omega_0 * out2) + out1

        
        out3 = torch.einsum("bij, bjk -> bik", out2, W3) + b3
        out3 = self.act3(out3) + out2
        #out3 = self.act3(out3 + out2)
        #out3 = torch.sin(self.omega_0 * out3) + out2

        
        out4 = torch.einsum("bij, bjk -> bik", out3, W4) + b4
        out4 = self.act4(out4) + out3
        #out4 = self.act4(out4 + out3)
        #out4 = torch.sin(self.omega_0 * out4) + out3

        #out5 = torch.einsum("bij, bjk -> bik", out4, W5) + b5
        #out5 = self.act5(out5) + out4
        
        
        #out = torch.squeeze(self.act(self.last(out4)))
        #out = torch.squeeze(self.last(out4))
        #out = torch.einsum("bij, bjk -> bik", out4, W5)
        
        
        #out_mu = torch.pow(torch.squeeze(self.last1(out4)), 2)#torch.squeeze(self.last1(out4))
        out_mu = torch.squeeze(self.act(self.last1(out4)))#torch.exp(torch.squeeze(self.last1(out4)))
        #out_mu = torch.squeeze(self.last1(out4))
        out_sigma = torch.exp(torch.squeeze(self.last2(out4)))
        
        #print(out_mu.shape)
        #if mode:
        #    out = out_mu + out_sigma*torch.mean(self.N.sample([out_mu.shape[0], out_mu.shape[1], 500]), axis = 2)
        #else:
        #    out = out_mu
        out = out_mu
        
        self.kl = torch.mean((0.5*out_sigma**2 + 0.5*(out_mu)**2 - torch.log(out_sigma) - 1/2))
        
        #out = (out*0.5+0.5)*10
        #out = torch.pow(out, 2)

        #out1 = torch.einsum("bij, bjk -> bik", x2, W1)  ##[b 1024 self.n_features]
        #out1 = self.act1(out1)
        #out = torch.einsum("bij, bjk -> bik", out1, W2)
        #out = self.act2(out) + out1
        #out = torch.squeeze(self.act(self.last(self.res3(self.res2(self.res1(out))))))
        
        
        
        #out = torch.squeeze(self.act(self.last(out)))
        #out = torch.squeeze(self.act(self.last(self.res1(out))))
        #out = torch.squeeze(self.act(self.last(self.res2(self.res1(out)))))
        #print(x2.shape, W.shape)
        #out = torch.einsum("bij, bjk -> bik", x2, W1)  ##[b 1024 self.n_features]
        #out = self.act1(out)
        #out = torch.einsum("bij, bjk -> bik", out, W2)
        #out = self.act2(out)
        #out = torch.einsum("bij, bjk -> bik", out, W3)
        #out = self.act3(out)
        #out = torch.einsum("bij, bjk -> bik", out, W4)
        #out = self.act4(out)

        #
        #out = out[:,:,0]
        
        ################################
        
        #if mode:
        #    out = out.reshape([b, 1,32,32])
        #else:
        #out = out.reshape([b, 1,32,32])
        #out = out.reshape([b, 1,64,64])
        
        
 
        #return torch.squeeze(self.last(self.res3(self.res2(self.res1(x)))))
        #return torch.squeeze(self.last(self.res2(self.res1(x))))
        return out, out_sigma, self.kl
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                #nn.init.kaiming_uniform_(m.weight)
                #nn.init.constant_(m.weight, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 
                    #nn.init.kaiming_uniform_(m.bias)
    
def build_cc_decoder(feature_size, m_size, pos_out_dim):
    return CC_Decoder(feature_size, m_size, pos_out_dim)
    

def make_layers_2(cfg, in_channels = 3,batch_norm=False,dilation = False):
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
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_layers(cfg, in_channels = 3,batch_norm=True,dilation = False):
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
    
def make_layers_3(cfg, in_channels = 3,batch_norm=False,dilation = False):
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
                layers += [conv2d, nn.BatchNorm2d(v), nn.PReLU()]
            else:
                layers += [conv2d, nn.PReLU()]
            in_channels = v
    return nn.Sequential(*layers)



class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[-1, -1],
                    [1, 1]]
        kernel_h = [[-1, 1],
                    [-1, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)


    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sum(torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6))/16

        return x
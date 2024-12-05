import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from lpcvc.loss import OhemCELoss2D,CrossEntropyLoss,CriterionOhemDSN, CriterionPixelWise, \
    CriterionAdv, CriterionAdvForG, CriterionAdditionalGP, CriterionPairWiseforWholeFeatAfterPool

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation, norm_layer = None):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.norm_layer = norm_layer

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out
    
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)

        return wd_params, nowd_params

# class Generator(nn.Module):
#     """Generator."""

#     def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, norm_layer = None):
#         super(Generator, self).__init__()
        
#         self.imsize = image_size
#         self
        
        
#         layer1 = []
#         layer2 = []
#         layer3 = []
#         last = []
        
#         repeat_num = int(np.log2(self.imsize)) - 3
#         mult = 2 ** repeat_num # 8
#         layer1.append(self.norm_layer(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
#         layer1.append(nn.BatchNorm2d(conv_dim * mult))
#         layer1.append(nn.ReLU())

#         curr_dim = conv_dim * mult

#         layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
#         layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
#         layer2.append(nn.ReLU())

#         curr_dim = int(curr_dim / 2)

#         layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
#         layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
#         layer3.append(nn.ReLU())

#         if self.imsize == 64:
#             layer4 = []
#             curr_dim = int(curr_dim / 2)
#             layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
#             layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
#             layer4.append(nn.ReLU())
#             self.l4 = nn.Sequential(*layer4)
#             curr_dim = int(curr_dim / 2)

#         self.l1 = nn.Sequential(*layer1)
#         self.l2 = nn.Sequential(*layer2)
#         self.l3 = nn.Sequential(*layer3)

#         last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
#         last.append(nn.Tanh())
#         self.last = nn.Sequential(*last)

#         self.attn1 = Self_Attn( 128, 'relu')
#         self.attn2 = Self_Attn( 64,  'relu')

#     def forward(self, z):
#         z = z.view(z.size(0), z.size(1), 1, 1)
#         out=self.l1(z)
#         out=self.l2(out)
#         out=self.l3(out)
#         out,p1 = self.attn1(out)
#         out=self.l4(out)
#         out,p2 = self.attn2(out)
#         out=self.last(out)

#         return out, p1, p2


## 자자 loss 코드를 짜보아용!
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=None, activation='leaky_relu',*args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn = norm_layer(out_chan, activation=activation)
        else:
            self.bn =  lambda x:x

        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, preprocess_GAN_mode=1, input_channel=14, 
                 batch_size=16, imsize=65, conv_dim=64,
                 d_loss=None,d_wgp_loss=None,
                 lambda_d = 0.1, lambda_gp = 10.0, norm_layer = None):
        super(discriminator, self).__init__()
        
        self.imsize = imsize

        self.d_loss = d_loss
        self.d_wgp_loss = d_wgp_loss
        self.lambda_d = lambda_d
        self.lambda_gp = lambda_gp

        self.norm_layer = norm_layer


        #layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        # self.layer1_conv = (nn.Conv2d(input_channel, conv_dim, 4, 2, 1))
        # self.layer1_bn = self.norm_layer(128)
        # self.layer1_lu = nn.LeakyReLU(0.1)
        # self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.layer1  = ConvBNReLU(input_channel, conv_dim, ks=4, stride=2, padding= 1,norm_layer=norm_layer)
        curr_dim = conv_dim

        # self.layer2_conv = nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)
        # self.layer2_bn = self.norm_layer(128)
        # self.layer2_lu = nn.LeakyReLU(0.1)
        self.layer2  = ConvBNReLU(curr_dim, curr_dim * 2, ks=4, stride=2, padding= 1,norm_layer=norm_layer)
        curr_dim = curr_dim * 2


        # self.layer3_conv = nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)
        # self.layer3_bn = self.norm_layer(128)
        # self.layer3_lu = nn.LeakyReLU(0.1)
        self.layer3  = ConvBNReLU(curr_dim, curr_dim * 2, ks=4, stride=2, padding= 1,norm_layer=norm_layer)
        curr_dim = curr_dim * 2

        #if self.imsize == 64:
        if self.imsize == 65:
            # self.layer4_conv = nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)
            # self.layer4_bn = self.norm_layer(128)
            # self.layer4_lu = nn.LeakyReLU(0.1)     
            self.layer4  = ConvBNReLU(curr_dim, curr_dim * 2, ks=4, stride=2, padding= 1,norm_layer=norm_layer)
            curr_dim = curr_dim * 2

       
    
        # self.l1 = nn.Sequential(*layer1)
        # self.l2 = nn.Sequential(*layer2)
        # self.l3 = nn.Sequential(*layer3)

        self.last = FPNOutputForSE(curr_dim, 1, 4, norm_layer=norm_layer)
        

        self.attn1 = Self_Attn(256, 'relu',norm_layer=norm_layer)
        self.attn2 = Self_Attn(512, 'relu',norm_layer=norm_layer)

        if preprocess_GAN_mode == 1: #'bn':
            self.preprocess_additional = ConvBNReLUforbatch(in_chan=0, out_chan=input_channel, norm_layer=norm_layer)
        elif preprocess_GAN_mode == 2: #'tanh':
            self.preprocess_additional = nn.Tanh()
        elif preprocess_GAN_mode == 3:
            self.preprocess_additional = lambda x: 2*(x/255 - 0.5)
        else:
            raise ValueError('preprocess_GAN_mode should be 1:bn or 2:tanh or 3:-1 - 1')
        
    def forward(self, x=None, t_pred=None, s_pred=None, loss_check=True):
        if loss_check:
            t_out_d, s_out_d, inter_out, inter = self.forward_common(t_pred, s_pred)
            D_loss = self.calculate_D_loss(s_out_d, t_out_d, inter_out, inter)
            return D_loss
        else:
            x = self.preprocess_additional(x)
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.attn1(out)
            out = self.layer4(out)
            out = self.attn2(out)
            out = self.last(out)
            return out

    def forward_common(self, t_pred, s_pred):
        real_images = t_pred
        # print(real_images)
        # assert False
        fake_images = s_pred
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)

        t = self.preprocess_additional(t_pred)
        s = self.preprocess_additional(s_pred)
        inter = self.preprocess_additional(interpolated)

        t_out = self.layer1(t)
        s_out = self.layer1(s)
        inter_out = self.layer1(inter)

        t_out = self.layer2(t_out)
        s_out = self.layer2(s_out)
        inter_out = self.layer2(inter_out)

        t_out = self.layer3(t_out)
        s_out = self.layer3(s_out)
        inter_out = self.layer3(inter_out)

        t_out = self.attn1(s_out)
        s_out = self.attn1(t_out)
        inter_out = self.attn1(inter_out)

        t_out = self.layer4(t_out)
        s_out = self.layer4(s_out)
        inter_out = self.layer4(inter_out)

        t_out = self.attn2(t_out)
        s_out = self.attn2(s_out)
        inter_out = self.attn2(inter_out)

        t_out = self.last(t_out)
        s_out = self.last(s_out)
        inter_out = self.last(inter_out)

        return t_out, s_out, inter_out, interpolated

    def calculate_D_loss(self, s_out_d, t_out_d, inter_out,interpolated):
        D_loss = 0
        d_loss = self.lambda_d * self.d_loss(s_out_d, t_out_d)
        D_loss += d_loss
        wgp_loss = self.lambda_d * self.lambda_gp * self.d_wgp_loss(inter_out, interpolated)
        D_loss += wgp_loss
        return D_loss


    # def forward(self, x, t_pred=None, s_pred=None, loss_check=False):
    #     #import pdb;pdb.set_trace()
    #     #print(x.size())

    #     if loss_check:
    #         real_images = t_pred
    #         fake_images = s_pred
    #         alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
    #         x = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        
    #     inter = x
    #     x = self.preprocess_additional(x)
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.attn1(out)
    #     out = self.layer4(out)
    #     out = self.attn2(out)
    #     out = self.last(out)

    #     return out , inter
        
    # def forward_common(self, t_pred, s_pred):
    #     real_images = t_pred
    #     fake_images = s_pred
    #     alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
    #     interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)

    #     t_pred = t_pred.detach()
    #     s_pred = s_pred.detach()
    #     t = self.preprocess_additional(t_pred)
    #     s = self.preprocess_additional(s_pred)
    #     inter = self.preprocess_additional(interpolated)

    #     t_out = self.layer1(t)
    #     s_out = self.layer1(s)
    #     inter_out = self.layer1(inter)

    #     t_out = self.layer2(t_out)
    #     s_out = self.layer2(s_out)
    #     inter_out = self.layer2(inter_out)

    #     s_out = self.layer3(t_out)
    #     t_out = self.layer3(s_out)
    #     inter_out = self.layer3(inter_out)

    #     t_out, tp1 = self.attn1(s_out)
    #     s_out, sp1 = self.attn1(t_out)
    #     inter_out, interp1 = self.attn1(inter_out)

    #     t_out = self.layer4(t_out)
    #     s_out = self.layer4(s_out)
    #     inter_out = self.layer4(inter_out)

    #     t_out, tp2 = self.attn2(t_out)
    #     s_out, sp2 = self.attn2(s_out)
    #     inter_out, sp2 = self.attn2(inter_out)

    #     t_out = self.last(t_out)
    #     s_out = self.last(s_out)
    #     inter_out = self.last(inter_out)

    #     return t_out, s_out, inter_out, interpolated

    # def calculate_D_loss(self, s_out, t_out, inter_out, interpolated):
    #     D_loss = 0
    #     d_loss = self.lambda_d * self.d_loss(s_out, t_out)
    #     D_loss += d_loss
    #     wgp_loss = self.lambda_d * self.lambda_gp * self.d_wgp_loss(inter_out, interpolated)
    #     D_loss += wgp_loss
    #     return D_loss

        # if loss_check == True:
        #     real_images = t_pred
        #     fake_images = s_pred
        #     alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        #     interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
            
        #     t_pred = t_pred.detach()
        #     s_pred = s_pred.detach()
        #     t = self.preprocess_additional(t_pred)
        #     s = self.preprocess_additional(s_pred)
        #     inter = self.preprocess_additional(interpolated)
            


        #     t_out = self.layer1(t)
        #     s_out = self.layer1(s)
        #     inter_out = self.layer1(inter)


        #     t_out = self.layer2(t_out)
        #     s_out = self.layer2(s_out)
        #     inter_out = self.layer2(inter_out)

        #     s_out = self.layer3(t_out)
        #     t_out = self.layer3(s_out)
        #     inter_out = self.layer3(inter_out)

        #     t_out, tp1 = self.attn1(s_out)
        #     s_out, sp1 = self.attn1(t_out)
        #     inter_out, interp1 = self.attn1(inter_out)
        #     # 여기까지
        #     t_out=self.layer4(t_out)
        #     s_out=self.layer4(s_out)
        #     inter_out=self.layer4(inter_out)

        #     t_out, tp2  = self.attn2(t_out)
        #     s_out, sp2 = self.attn2(s_out)
        #     inter_out, sp2 = self.attn2(inter_out)

        #     t_out=self.last(t_out)
        #     s_out=self.last(s_out)
        #     inter_out=self.last(inter_out)


        #     # d_loss
        #     # CriterionAdv
        #     D_loss = 0
        #     d_loss = self.lambda_d * self.d_loss(s_out, t_out)
        #     D_loss += d_loss
            
        #     wgp_loss = self.lambda_d * self.lambda_gp * self.d_wgp_loss(inter_out,interpolated)
        #     D_loss += wgp_loss

        #     return D_loss


    #     # #return [out.squeeze(), p1, p2]
    #     # return d_out

    def discriminator_backward(self, t_pred=None, s_pred=None):

        lambda_d = self.lambda_d
        lambda_gp = self.lambda_gp
        d_out_T = self.forward(t_pred.cuda())
        d_out_S = self.forward(s_pred.cuda())


        # d_loss
        d_loss = lambda_d*self.d_loss(d_out_S, d_out_T)

        #print("wgp_loss")
        #print(d_out_S.size())
        #print(d_out_T.size())        
        # d_wgp_loss
        d_loss += lambda_d * lambda_gp * self.d_wgp_loss(s_pred, t_pred)
        #assert False
        d_loss.backward() 

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if isinstance(child,(CriterionOhemDSN, CriterionPixelWise,CriterionAdv,CriterionAdditionalGP,
                                 CriterionAdvForG, CriterionPairWiseforWholeFeatAfterPool)):
                continue
            elif isinstance(child, (Self_Attn, ConvBNReLU, FPNOutputForSE)): # not in FPNout
                child_wd_params, child_nowd_params = child.get_params()
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                child_wd_params, child_nowd_params = child.get_params()
                wd_params += child_wd_params
                nowd_params += child_nowd_params

        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    



# channel?
class FPNOutputForSE(nn.Module):
    def __init__(self, mid_chan, n_classes, kernel_size=1, norm_layer=None, *args, **kwargs):
        super(FPNOutputForSE, self).__init__()
        self.norm_layer = norm_layer
        # 값 튄는걸 방지하고, 이후 14 채널 맞춰주기 위해 256 그대로 빼줌
        #   256 보다 낮게할 경우 정확도가 오히려 낮게 나왔다?
        #   일단 해보자.
        #   대체 불가.
        #self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size, bias=False)
        self.init_weight()

    def forward(self, x):
        # x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, self.norm_layer):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class ConvBNReLUforbatch(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=None, activation='leaky_relu',*args, **kwargs):
        super(ConvBNReLUforbatch, self).__init__()
        # self.conv = nn.Conv2d(in_chan,
        #         out_chan,
        #         kernel_size = ks,
        #         stride = stride,
        #         padding = padding,
        #         bias = False)
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn = norm_layer(out_chan, activation=activation)
        else:
            self.bn =  lambda x:x

        self.init_weight()

    def forward(self, x):
        #x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

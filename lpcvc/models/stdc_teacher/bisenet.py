#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lpcvc.loss import OhemCELoss2D,CrossEntropyLoss, DetailAggregateLoss

#from nets.stdcnet import STDCNet1446, STDCNet813
from .stdcnet import STDC1_pt, STDC2_pt
#from modules.bn import InPlaceABNSync as BatchNorm2d
# BatchNorm2d = nn.BatchNorm2d

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class BatchNorm2d(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, num_features, activation='none'):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))
    
# class ConvBNReLU(nn.Module):
#     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
#         super(ConvBNReLU, self).__init__()
#         self.conv = nn.Conv2d(in_chan,
#                 out_chan,
#                 kernel_size = ks,
#                 stride = stride,
#                 padding = padding,
#                 bias = False)
#         # self.bn = BatchNorm2d(out_chan)
#         self.bn = BatchNorm2d(out_chan, activation='none')
#         self.relu = nn.ReLU()
#         self.init_weight()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)

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

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.norm_layer = norm_layer

        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
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


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm_layer=None, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.norm_layer = norm_layer

        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        # self.bn_atten = BatchNorm2d(out_chan)
        self.bn_atten = BatchNorm2d(out_chan, activation='none')

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, backbone='STDC1', pretrain_model='', use_conv_last=False, norm_layer=None, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.norm_layer = norm_layer
        self.backbone_name = backbone
        if backbone == 'STDC2':
            self.backbone = STDC2_pt(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128, norm_layer=norm_layer)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128, norm_layer=norm_layer)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1, norm_layer=norm_layer)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1, norm_layer=norm_layer)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        elif backbone == 'STDC1':
            self.backbone = STDC1_pt(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128, norm_layer=norm_layer)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128, norm_layer=norm_layer)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1, norm_layer=norm_layer)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1, norm_layer=norm_layer)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0, norm_layer=norm_layer)
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]

        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        
        avg = F.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        
        return feat2, feat4, feat8, feat16, feat16_up, feat32_up # x8, x16

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


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm_layer=None, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.norm_layer = norm_layer

        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

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


class BiSeNet(nn.Module):
    def __init__(self, 
                 n_class=14,
                 backbone='STDC1',
                 norm_layer = BatchNorm2d,
                 loss_fn=None, 
                 detail_loss=None,
                 use_boundary_2=True, 
                 use_boundary_4=True, 
                 use_boundary_8=True, 
                 use_boundary_16=False, 
                 use_conv_last=False, 
                 heat_map=False, *args, **kwargs):
        super(BiSeNet, self).__init__()
        
        self.detail_loss = detail_loss
        self.loss_fn = loss_fn
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = n_class   
        
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16

        # self.heat_map = heat_map
        self.cp = ContextPath(backbone, pretrain_model=False, use_conv_last=use_conv_last, norm_layer=norm_layer)
        
        
        
        if backbone == 'STDC2':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDC1':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.ffm = FeatureFusionModule(inplane, 256, norm_layer=norm_layer)
        self.conv_out = BiSeNetOutput(256, 256, n_class, norm_layer=norm_layer)
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_class, norm_layer=norm_layer)
        self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, n_class, norm_layer=norm_layer)

        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1, norm_layer=norm_layer)
        
        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1, norm_layer=norm_layer)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1, norm_layer=norm_layer)
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1, norm_layer=norm_layer)
        self.init_weight()

    def forward(self, x, lbl=None):
        H, W = x.size()[2:]
        
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)

        feat_out_sp2 = self.conv_out_sp2(feat_res2)

        feat_out_sp4 = self.conv_out_sp4(feat_res4)
  
        feat_out_sp8 = self.conv_out_sp8(feat_res8)

        #feat_out_sp16 = self.conv_out_sp16(feat_res16)

        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)


        if self.training:
            feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
            feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
            
            if self.use_boundary_2: 
                boundery_bce_loss2, boundery_dice_loss2 = self.detail_loss(feat_out_sp2, lbl)
            if self.use_boundary_4:     
                boundery_bce_loss4, boundery_dice_loss4 = self.detail_loss(feat_out_sp4, lbl)
            if self.use_boundary_8:         
                boundery_bce_loss8, boundery_dice_loss8 = self.detail_loss(feat_out_sp8, lbl)

            boundery_bice_loss = boundery_dice_loss2 + boundery_dice_loss4 + boundery_dice_loss8
            boundery_bce_loss = boundery_bce_loss2 + boundery_bce_loss4 + boundery_bce_loss8

            ohem_loss = self.loss_fn(feat_out,lbl) + self.loss_fn(feat_out16,lbl) + self.loss_fn(feat_out32,lbl)
            loss = ohem_loss + boundery_bce_loss + boundery_bice_loss
            
            return loss
        
        else: 
            return feat_out        
        # if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
        #     return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8
        
        # if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
        #     return feat_out, feat_out16, feat_out32, feat_out_sp4, feat_out_sp8

        # if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
        #     return feat_out, feat_out16, feat_out32, feat_out_sp8
        
        # if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
        #     return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if isinstance(child,(OhemCELoss2D,CrossEntropyLoss, DetailAggregateLoss)):
                continue
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


import torch
import torch.nn as nn
import torch.nn.functional as F
from .stdcnet import STDC1_pt
from lpcvc.loss import OhemCELoss2D,CrossEntropyLoss


def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)

class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0

# SEblock + FuseUp
class SEBlockFusionModule(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    approx_sigmoid : bool, default False
        Whether to use approximated sigmoid function.
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_chan,
                 mid_chn,
                 out_chan=128,
                 norm_layer=None,
                 reduction=16,
                 approx_sigmoid=False, *args, **kwargs):
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        super(SEBlockFusionModule, self).__init__()
        # [2, 4, 8, 16, 32]
        mid_cannels = in_chan // reduction
        mid_chn = int(in_chan/2)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 1x1 spatial +
        self.conv1 = conv1x1(
            in_channels=in_chan,
            out_channels=mid_cannels,
            bias=True)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(
            in_channels=mid_cannels,
            out_channels=in_chan,
            bias=True)
        self.sigmoid = HSigmoid() if approx_sigmoid else nn.Sigmoid()

        # fuse
        self.up = ConvBNReLU(in_chan, mid_chn, ks=1, stride=1, padding=1, norm_layer=norm_layer)
        self.smooth = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)


        self.init_weight()

    def forward(self, x, up_fea_in, up_flag, smf_flag):

        # replacement of Fast Attention module
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        p_feat = x * w

        if up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            smooth_feat = self.smooth(p_feat)
            return up_feat, smooth_feat

        if up_flag and not smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            return up_feat

        if not up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            smooth_feat = self.smooth(p_feat)
            return smooth_feat

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, (H,W), **self._up_kwargs) + y

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


class FANet_se_stdc1(nn.Module):
    def __init__(self,
                 nclass=14,
                 backbone='stdc',
                 norm_layer=BatchNorm2d,
                 loss_fn=None):
        super(FANet_se_stdc1, self).__init__()

        self.loss_fn = loss_fn
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass
        # copying modules from pretrained models
        self.backbone = backbone
        if backbone == 'stdc1':
            self.expansion = 1
            self.resnet = STDC1_pt(num_classes=nclass, pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options

        # self.ffm_32 = LAFeatureFusionModule(512*self.expansion,256,128,norm_layer=norm_layer)
        # self.ffm_16 = LAFeatureFusionModule(256*self.expansion,256,128,norm_layer=norm_layer)
        # self.ffm_8 = LAFeatureFusionModule(128*self.expansion,256,128,norm_layer=norm_layer)
        # self.ffm_4 = LAFeatureFusionModule(64*self.expansion,256,128,norm_layer=norm_layer)

        # in, mid, out
        #self.ffm_32 = SEBlockFusionModule(512*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_16 = SEBlockFusionModule(1024*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_8 = SEBlockFusionModule(512*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_4 = SEBlockFusionModule(256*self.expansion,256,128,norm_layer=norm_layer)


        self.clslayer_16 = FPNOutput(128, 64, nclass,norm_layer=norm_layer)
        self.clslayer_4 = FPNOutput(128, 64, nclass,norm_layer=norm_layer)
        self.clslayer_8  = FPNOutputForSE(256, 14, nclass,norm_layer=norm_layer)

    def forward(self, x, lbl=None):

        _, _, h, w = x.size()

        # feat4, feat8, feat16, feat32 = self.resnet(x)
        _, _, feat4, feat8, feat16 = self.resnet(x)

        # upfeat_32, smfeat_32 = self.ffm_32(feat32,None,True,True)
        # upfeat_16, smfeat_16 = self.ffm_16(feat16,upfeat_32,True,True)
        #print(feat4.size())
        #print(feat8.size())
        #print(feat16.size())

        # feat4= torch.Size([16, 64, 48, 48])
        # feat8= torch.Size([16, 128, 24, 24])
        # feat16= torch.Size([16, 256, 12, 12])
        upfeat_16, smfeat_16 = self.ffm_16(feat16,None,True,True)
        # print('upfeat_16=', upfeat_16.shape)
        # print('smfeat_16=', smfeat_16.shape)
        # upfeat_16= torch.Size([16, 128, 14, 14])
        # smfeat_16= torch.Size([16, 128, 12, 12])

        upfeat_8 = self.ffm_8(feat8,upfeat_16,True,False)
        # print('upfeat_8=', upfeat_8.shape)
        # upfeat_8= torch.Size([16, 64, 26, 26])

        smfeat_4 = self.ffm_4(feat4,upfeat_8,False,True)
        # print('smfeat_4=', smfeat_4.shape)
        # smfeat_4= torch.Size([16, 128, 48, 48])

        pair_feat = self._upsample_cat(smfeat_16, smfeat_4)
        # print('_upsample_cat=', x.shape)
        # _upsample_cat= torch.Size([16, 256, 48, 48])


        x = self.clslayer_8(pair_feat)
        # print('clslayer_8=', x.shape)
        # clslayer_8= torch.Size([16, 14, 48, 48])

        outputs = F.interpolate(x, (h,w), **self._up_kwargs)
        # print('outputs=', outputs.shape)
        # outputs= torch.Size([16, 14, 384, 384])

        # Auxiliary layers for training
        if self.training:
            auxout_1 = self.clslayer_16 (smfeat_4)
            auxout_2 = self.clslayer_16 (smfeat_16)
            auxout_1 = F.interpolate(auxout_1, (h,w), **self._up_kwargs)
            auxout_2 = F.interpolate(auxout_2, (h,w), **self._up_kwargs)
            loss = self.loss_fn(outputs,lbl) + self.loss_fn(auxout_1,lbl) + self.loss_fn(auxout_2,lbl)
            return loss
        else:
            return outputs# , pair_feat

    def _upsample_cat(self, x1, x2):
        '''Upsample and concatenate feature maps.
        '''
        _,_,H,W = x2.size()
        x1 = F.interpolate(x1, (H,W), **self._up_kwargs)
        x = torch.cat([x1,x2],dim=1)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if isinstance(child,(OhemCELoss2D,CrossEntropyLoss)):
                continue
            elif isinstance(child, (SEBlockFusionModule, FPNOutput)):
                child_wd_params, child_nowd_params = child.get_params()
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:

                child_wd_params, child_nowd_params = child.get_params()
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


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

# channel?
class FPNOutputForSE(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None, *args, **kwargs):
        super(FPNOutputForSE, self).__init__()
        self.norm_layer = norm_layer
        # 값 튄는걸 방지하고, 이후 14 채널 맞춰주기 위해 256 그대로 빼줌
        #   256 보다 낮게할 경우 정확도가 오히려 낮게 나왔다?
        #   일단 해보자.
        #   대체 불가.
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        # x = self.conv_out(x)
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


# channel?
class FPNOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None, *args, **kwargs):
        super(FPNOutput, self).__init__()
        self.norm_layer = norm_layer
        # 값 튄는걸 방지하고, 이후 14 채널 맞춰주기 위해 256 그대로 빼줌
        #   256 보다 낮게할 경우 정확도가 오히려 낮게 나왔다?
        #   일단 해보자.
        #   대체 불가.
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

class LAFeatureFusionModule(nn.Module):
    def __init__(self, in_chan, mid_chn=256, out_chan=128, norm_layer=None, *args, **kwargs):
        super(LAFeatureFusionModule, self).__init__()
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        mid_chn = int(in_chan/2)
        self.w_qs = ConvBNReLU(in_chan, 32, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_ks = ConvBNReLU(in_chan, 32, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_vs = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.latlayer3 = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.up = ConvBNReLU(in_chan, mid_chn, ks=1, stride=1, padding=1, norm_layer=norm_layer)
        self.smooth = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)

        self.init_weight()

    def forward(self, feat, up_fea_in,up_flag, smf_flag):

        query = self.w_qs(feat)
        key   = self.w_ks(feat)
        value = self.w_vs(feat)

        N,C,H,W = feat.size()

        query_ = query.view(N,32,-1).permute(0, 2, 1)
        query = F.normalize(query_, p=2, dim=2, eps=1e-12)

        key_   = key.view(N,32,-1)
        key   = F.normalize(key_, p=2, dim=1, eps=1e-12)

        value = value.view(N,C,-1).permute(0, 2, 1)

        f = torch.matmul(key, value)
        y = torch.matmul(query, f)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W_y = self.latlayer3(y)
        p_feat = W_y + feat

        if up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            smooth_feat = self.smooth(p_feat)
            return up_feat, smooth_feat

        if up_flag and not smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            return up_feat

        if not up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            smooth_feat = self.smooth(p_feat)
            return smooth_feat


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, (H,W), **self._up_kwargs) + y

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

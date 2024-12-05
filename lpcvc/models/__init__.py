import copy
import torchvision.models as models
import torch.nn as nn
from lpcvc.models.fanet import FANet

from lpcvc.models.KCC_fanet_18 import FANet as kcc_fanet18
from lpcvc.models.KCC_fanet_9 import FANet as kcc_fanet9

from lpcvc.models.fanet_student import FANet as FANet_student

from lpcvc.models.stdc_teacher import BiSeNet

from lpcvc.models.fanet_teacher import FANet as FANet_teacher

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
    
class BatchNorm2dForD(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, num_features, activation='none'):
        super(BatchNorm2dForD, self).__init__(num_features=num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2dForD, self).forward(x))

def get_model(model_dict, nclass, loss_fn=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    param_dict["loss_fn"] = loss_fn
    param_dict['norm_layer'] = BatchNorm2d
    
    model = model(nclass=nclass, **param_dict)
    return model

def get_stdc_model(model_dict, nclass, loss_fn=None, detail_loss=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    param_dict["loss_fn"] = loss_fn
    param_dict["detail_loss"] = detail_loss
    param_dict['norm_layer'] = BatchNorm2d
    
    model = model(nclass=nclass, **param_dict)
    return model

def get_student_model(model_dict, nclass, 
                      st_loss=None,
                      pi_loss=None,
                      pa_loss=None,
                      ho_loss=None,
                      lambda_pa=0.5,
                      lambda_pi=10.0,
                      lambda_d =0.1):
    
    name = model_dict["student"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("student")
    
    param_dict["st_loss"] = st_loss
    param_dict["pi_loss"] = pi_loss
    param_dict["pa_loss"] = pa_loss
    param_dict["ho_loss"] = ho_loss

    param_dict["lambda_pa"] = lambda_pa
    param_dict["lambda_pi"] = lambda_pi
    param_dict["lambda_d"] = lambda_d
    
    param_dict['norm_layer'] = BatchNorm2d
    
    model = model(nclass=nclass, **param_dict)
    return model

def get_teacher_model(model_dict, nclass):
    name = model_dict["teacher"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("teacher")
    #param_dict["loss_fn"] = loss_fn
    param_dict['norm_layer'] = BatchNorm2d
    
    model = model(nclass=nclass, **param_dict)
    return model

# def get_discriminator_model(model_dict, d_loss=None, d_wgp_loss=None,
#                             lambda_gp=10.0, lambda_d=0.1):
    
#     name = model_dict["Discriminator_name"]
#     model = _get_model_instance(name)
#     param_dict = copy.deepcopy(model_dict)

#     param_dict["d_loss"] = d_loss
#     param_dict["d_wgp_loss"] = d_wgp_loss

#     param_dict["lambda_gp"] = lambda_gp
#     param_dict["lambda_d"] = lambda_d

#     param_dict['norm_layer'] = BatchNorm2dForD
#     param_dict.pop("Discriminator_name")
    
#     model = model(**param_dict)
#     return model

def _get_model_instance(name):
    if name == "fanet":
        return {
            "fanet": FANet
        }[name]
    # basic model ----------------------------------------
    if name == "kcc_fanet18":
        return {
            "kcc_fanet18": kcc_fanet18
        }[name]   
    if name == "kcc_fanet9":
        return {
            "kcc_fanet9": kcc_fanet9
        }[name]  

    # ----------------------------------------------------
 
    # student model --------------------------------------
    if name == "fanet_student":
        return {
            "fanet_student": FANet_student
        }[name]
 
    # ----------------------------------------------------    

    # teacher model --------------------------------------
    if name == "bisenet":
        return {
            "bisenet": BiSeNet
        }[name]   
    if name == "fanet_teacher":
        return {
            "fanet_teacher": FANet_teacher
        }[name]   
    # ----------------------------------------------------
    #   
    else:
        raise ("Model {} not available".format(name))

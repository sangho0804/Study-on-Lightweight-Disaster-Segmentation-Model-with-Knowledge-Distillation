import logging
import torch
import functools

from lpcvc.loss.loss import (
    CrossEntropyLoss,
    OhemCELoss2D,
    DetailAggregateLoss
)
from lpcvc.loss.loss import CriterionDSN, CriterionOhemDSN, CriterionPixelWise, \
    CriterionAdv, CriterionAdvForG, CriterionAdditionalGP, CriterionPairWiseforWholeFeatAfterPool

logger = logging.getLogger("lpcvc")

key2loss = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "OhemCELoss2D": OhemCELoss2D,
    "CriterionOhemDSN" : CriterionOhemDSN,
    "CriterionPixelWise" : CriterionPixelWise,
    "CriterionPairWiseforWholeFeatAfterPool" : CriterionPairWiseforWholeFeatAfterPool,
    "CriterionAdvForG" : CriterionAdvForG,
    "CriterionAdv" : CriterionAdv,
    "CriterionAdditionalGP" : CriterionAdditionalGP,
    "DetailAggregateLoss" : DetailAggregateLoss,
}

def get_loss_function(cfg):
    assert(cfg["loss"] is not None)
    loss_dict = cfg["loss"]
    loss_name = loss_dict["name"]
    loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
    if loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(loss_name))

    if loss_name == "OhemCELoss2D":
        n_img_per_gpu = int(cfg["batch_size"]/torch.cuda.device_count())
        cropsize = cfg["train_augmentations"]["rcrop"]
        n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
        loss_params["n_min"] = n_min

    logger.info("Using {} with {} params".format(loss_name, loss_params))
    return key2loss[loss_name](**loss_params)

def get_detail_loss_function(cfg):
    assert(cfg["detail_loss"] is not None)
    loss_dict = cfg["detail_loss"]
    loss_name = loss_dict["name"]
    loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
    if loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(loss_name))

    logger.info("Using {} with {} params".format(loss_name, loss_params))
    return key2loss[loss_name](**loss_params)

def get_dist_loss_function(cfg):
    assert(cfg["st_loss"] is not None)
    assert(cfg["pi_loss"] is not None)
    assert(cfg["pa_loss"] is not None)
    assert(cfg["ho_loss"] is not None)
    assert(cfg["d_loss"] is not None)
    assert(cfg["d_wgp_loss"] is not None)

    st_loss_dict = cfg["st_loss"]
    pi_loss_dict = cfg["pi_loss"]
    pa_loss_dict = cfg["pa_loss"]
    ho_loss_dict = cfg["ho_loss"]
    d_loss_dict = cfg["d_loss"]
    d_wgp_loss_dict = cfg["d_wgp_loss"]

    st_loss_name = st_loss_dict["name"]
    pi_loss_name = pi_loss_dict["name"]
    pa_loss_name = pa_loss_dict["name"]
    ho_loss_name = ho_loss_dict["name"]
    d_loss_name = d_loss_dict["name"]
    d_wgp_loss_name = d_wgp_loss_dict["name"]

    st_loss_params = {k: v for k, v in st_loss_dict.items() if k != "name"}
    pi_loss_params = {k: v for k, v in pi_loss_dict.items() if k != "name"}
    pa_loss_params = {k: v for k, v in pa_loss_dict.items() if k != "name"}
    ho_loss_params = {k: v for k, v in ho_loss_dict.items() if k != "name"}
    d_loss_params = {k: v for k, v in d_loss_dict.items() if k != "name"}
    d_wgp_loss_params = {k: v for k, v in d_wgp_loss_dict.items() if k != "name"}

    if st_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(st_loss_name))
    if pi_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(pi_loss_name))
    if pa_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(pa_loss_name))
    if ho_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(ho_loss_name))    
    if d_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(d_loss_name))
    if d_wgp_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(d_wgp_loss_name))  
    
    logger.info("Using {} with {} params".format(st_loss_name, st_loss_params))
    logger.info("Using {} with {} params".format(pi_loss_name, pi_loss_params))
    logger.info("Using {} with {} params".format(pa_loss_name, pa_loss_params))
    logger.info("Using {} with {} params".format(ho_loss_name, ho_loss_params))
    logger.info("Using {} with {} params".format(d_loss_name, d_loss_params))
    logger.info("Using {} with {} params".format(d_wgp_loss_name, d_wgp_loss_params))

    if st_loss_name == "OhemCELoss2D":
        n_img_per_gpu = int(cfg["batch_size"])
        cropsize = cfg["train_augmentations"]["rcrop"]
        n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
        st_loss_params["n_min"] = n_min

    st_loss = key2loss[st_loss_name](**st_loss_params)
    pi_loss = key2loss[pi_loss_name](**pi_loss_params)
    pa_loss = key2loss[pa_loss_name](**pa_loss_params)
    ho_loss = key2loss[ho_loss_name](**ho_loss_params)
    d_loss = key2loss[d_loss_name](**d_loss_params)
    d_wgp_loss = key2loss[d_wgp_loss_name](**d_wgp_loss_params)

    return st_loss, pi_loss, pa_loss, ho_loss, d_loss, d_wgp_loss

# def get_wgp_loss_function(cfg, d_model):
#     assert(cfg["d_wgp_loss"] is not None)
#     d_wgp_loss_dict = cfg["d_wgp_loss"]
#     d_wgp_loss_name = d_wgp_loss_dict["name"]
#     d_wgp_loss_params = {k: v for k, v in d_wgp_loss_dict.items() if k != "name"}
#     if d_wgp_loss_name not in key2loss:
#         raise NotImplementedError("Loss {} not implemented".format(d_wgp_loss_name)) 
#     logger.info("Using {} with {} params".format(d_wgp_loss_name, d_wgp_loss_params))
#     d_wgp_loss = key2loss[d_wgp_loss_name](d_model, **d_wgp_loss_params)
#     return d_wgp_loss

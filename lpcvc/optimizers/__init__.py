import logging
import copy

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from .adaoptimizer import AdaOptimizer

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
    "adaoptimizer": AdaOptimizer,
}


def get_optimizer(cfg, model):
    if cfg["optimizer"] is None:
        return SGD(model.parameters())

    else:
        opt_name = cfg["optimizer"]["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        optimizer_cls = key2opt[opt_name]
        if opt_name == "adaoptimizer":
            param_dict = copy.deepcopy(cfg["optimizer"])
            param_dict.pop("name")
            optimizer = optimizer_cls(model, **param_dict) # module for multi GPU

        else:
            param_dict = copy.deepcopy(cfg["optimizer"])
            param_dict.pop("name")
            optimizer = optimizer_cls(model.parameters(), **param_dict)

        return optimizer
    
def get_st_optimizer(cfg, model):
    if cfg["s_optimizer"] is None:
        return SGD(model.parameters())

    else:
        opt_name = cfg["s_optimizer"]["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        optimizer_cls = key2opt[opt_name]
        if opt_name == "adaoptimizer":
            param_dict = copy.deepcopy(cfg["s_optimizer"])
            param_dict.pop("name")
            optimizer = optimizer_cls(model, **param_dict) # module for multi GPU

        else:
            param_dict = copy.deepcopy(cfg["s_optimizer"])
            param_dict.pop("name")
            optimizer = optimizer_cls(model.parameters(), **param_dict)

        return optimizer
    
def get_disc_optimizer(cfg, model):
    if cfg["d_optimizer"] is None:
        return SGD(model.parameters())

    else:
        opt_name = cfg["d_optimizer"]["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        optimizer_cls = key2opt[opt_name]
        if opt_name == "adaoptimizer":
            param_dict = copy.deepcopy(cfg["d_optimizer"])
            param_dict.pop("name")
            optimizer = optimizer_cls(model, **param_dict) # module for multi GPU

        else:
            param_dict = copy.deepcopy(cfg["d_optimizer"])
            param_dict.pop("name")
            optimizer = optimizer_cls(model.parameters(), **param_dict)

        return optimizer

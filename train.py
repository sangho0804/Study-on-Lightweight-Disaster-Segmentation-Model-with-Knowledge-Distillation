import os
import oyaml as yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from tqdm import tqdm
import torch.distributed as dist

from lpcvc.models import get_model, get_stdc_model
from lpcvc.loss import get_loss_function, get_detail_loss_function
from lpcvc.loader import get_loader
from lpcvc.utils import get_logger
from lpcvc.metrics import runningScore, averageMeter, AccuracyTracker
from lpcvc.augmentations import get_composed_augmentations
from lpcvc.optimizers import get_optimizer
from lpcvc.utils import convert_state_dict

def get_dice(image, groundTruth):
    accuracyTracker: AccuracyTracker = AccuracyTracker(n_classes=14)
    accuracyTracker.update(groundTruth, image)
    accuracyTracker.get_scores()
    return accuracyTracker.mean_dice


def init_seed(manual_seed, en_cudnn=False):
    torch.cuda.benchmark = en_cudnn
    torch.cuda.cudnn_enabled = en_cudnn
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)

def train(cfg):

    # train id 설정
    #run_id = random.randint(1, 100000)
    run_id = 2023100100
    init_seed(11733, en_cudnn=True)


    #gpu lank
    global local_rank
    local_rank = cfg["local_rank"]

    # gpu 가 single 이 아닐 경우 사용하세요.
    if local_rank == 0:
        logdir = os.path.join("runs", os.path.basename(args.config)[:-4])
        work_dir = os.path.join(logdir, str(run_id))

        if not os.path.exists("runs"):
            os.makedirs("runs")
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        shutil.copy(args.config, work_dir)

        logger = get_logger(work_dir)
        logger.info("Let the games begin RUNDIR: {}".format(work_dir))


    # Setup nodes
    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend='nccl', init_method='env://')

    global gpus_num
    gpus_num = torch.cuda.device_count()
    if local_rank == 0:
        logger.info(f'use {gpus_num} gpus')
        logger.info(f'configure: {cfg}')

    # Setup Augmentations
    train_augmentations = cfg["training"].get("train_augmentations", None)
    t_data_aug = get_composed_augmentations(train_augmentations)
    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(data_path,split=cfg["data"]["train_split"],augmentations=t_data_aug)
    v_loader = data_loader(data_path,split=cfg["data"]["val_split"],augmentations=v_data_aug)
    
    # multi GPU 를 위한 t_samper
    #t_sampler = torch.utils.data.distributed.DistributedSampler(t_loader, shuffle=True)

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg["training"]["batch_size"]//gpus_num,
                                  num_workers=cfg["training"]["n_workers"]//gpus_num,
                                  shuffle=False,
                                  #sampler = t_sampler,
                                  pin_memory = True,
                                  drop_last=True  )
    valloader = data.DataLoader(v_loader,
                                batch_size=cfg["validating"]["batch_size"],
                                num_workers=cfg["validating"]["n_workers"] )

    if local_rank == 0:
        logger.info("Using training seting {}".format(cfg["training"]))
    

    # Setup Loss

    if cfg["model"]["arch"] == "bisenet":
        loss_fn = get_loss_function(cfg["training"])
        detail_loss = get_detail_loss_function(cfg["training"])
        # print(detail_loss)
        # print(loss_fn)
        # assert False
    else:
        loss_fn = get_loss_function(cfg["training"])
    
    
    if local_rank == 0:
        logger.info("Using loss {}".format(loss_fn))

    # Setup Model
    if cfg["model"]["arch"] == "bisenet":
        model = get_stdc_model(cfg["model"],t_loader.n_classes,loss_fn=loss_fn, detail_loss=detail_loss)
    
    else:
        model = get_model(cfg["model"],t_loader.n_classes,loss_fn=loss_fn)

    # Setup optimizer
    optimizer = get_optimizer(cfg["training"], model)

   #Initialize training param
    start_iter = 0
    best_iou = -100.0
    best_dice = -100.0
    best_total_dice = -100.0

    # Resume from checkpoint
    if cfg["training"]["resume"] is not None and  local_rank == 0:
        if os.path.isfile(cfg["training"]["resume"]):
            ckpt = torch.load(cfg["training"]["resume"])
            model.load_state_dict(ckpt, strict=False)
            #optimizer.load_state_dict(ckpt['optimizer'])
            #best_iou = ckpt['best_iou']
            #start_iter = ckpt['iter']
            if local_rank == 0:
                logger.info( "Resuming training from checkpoint '{}'".format(cfg["training"]["resume"]))
        else:
            if local_rank == 0:
                logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))




    # Setup multi GPU
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #         device_ids = [cfg["local_rank"]],
    #         output_device = cfg["local_rank"],
    #         find_unused_parameters=True
    #         )
    if local_rank == 0:
        logger.info("Model initialized on GPUs.")


 

    # Setup Metrics
    if local_rank == 0:
        running_metrics_val = runningScore(t_loader.n_classes)


    time_meter = averageMeter()
    i = start_iter

    while i <= cfg["training"]["train_iters"]:
        for (images, labels) in trainloader:
            i += 1
            model.train()
            optimizer.zero_grad()

            start_ts = time.time()

            loss = model(images.cuda(), labels.cuda())
            loss =torch.mean(loss)
            loss.backward()
            time_meter.update(time.time() - start_ts)

            optimizer.step()

            if local_rank == 0 and (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                                            i + 1,
                                            cfg["training"]["train_iters"],
                                            loss.item(),
                                            time_meter.avg / cfg["training"]["batch_size"], )
                logger.info(print_str)
                time_meter.reset()

            if local_rank == 0 and (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:
                total_dice = 0
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.cuda()
                        labels_val = labels_val.cuda()
                        outputs = model(images_val)
                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()
                        running_metrics_val.update(gt, pred)
                        score_dice = get_dice(pred, gt)
                        total_dice += score_dice
                
                everage_dice = total_dice / 100

                score, class_iou = running_metrics_val.get_scores()
                logger.info("Mean Dice       : \t:{}".format(everage_dice))

                for k, v in score.items():
                    logger.info("{}: {}".format(k, v))

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))

                running_metrics_val.reset()


                state = {
                    "iter": i + 1,
                    "model_state": model.state_dict(),
                    "best_iou": score["Mean IoU        : \t"],
                    "optimizer" : optimizer.state_dict(),
                }
                save_path = os.path.join(
                    work_dir,
                    "{}_{}_last_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                )
                torch.save(state, save_path)


                if score["Mean IoU        : \t"] >= best_iou:
                    best_iou = score["Mean IoU        : \t"]
                    # state = {
                    #     "iter": i + 1,
                    #     "model_state": model.module.state_dict(),
                    #     "best_iou": best_iou,
                    #     "optimizer" : optimizer.state_dict(),
                    # }
                    save_path_1 = os.path.join(
                        work_dir,
                        "{}_{}_best_model.pth".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(model.state_dict(), save_path_1)


                if everage_dice >= best_dice:
                    best_dice = everage_dice
                    # state = {
                    #     "iter": i + 1,
                    #     "model_state": model.module.state_dict(),
                    #     "best_iou": best_iou,
                    #     "optimizer" : optimizer.state_dict(),
                    # }
                    save_path_2 = os.path.join(
                        work_dir,
                        "{}_{}_best_Dice_model.pth".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(model.state_dict(), save_path_2)

                if score["total Mean Dice : \t"] >= best_total_dice:
                    best_total_dice = score["total Mean Dice : \t"]
                    # state = {
                    #     "iter": i + 1,
                    #     "model_state": model.module.state_dict(),
                    #     "best_iou": best_iou,
                    #     "optimizer" : optimizer.state_dict(),
                    # }
                    save_path_3 = os.path.join(
                        work_dir,
                        "{}_{}_best_total_Dice_model.pth".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(model.state_dict(), save_path_3)
                    
                logger.info("best Mean Dice       : \t :{}".format(best_dice))
                logger.info("best Mean iou        : \t :{}".format(best_iou))
                logger.info("best total Mean Dice : \t :{}".format(best_total_dice))
                    
#os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="",
        help="Configuration file to use",
    )
    parser.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = 0,
            )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    cfg["local_rank"] = args.local_rank

    if cfg["training"]["optimizer"]["max_iter"] is not None:
        assert(cfg["training"]["train_iters"]==cfg["training"]["optimizer"]["max_iter"])
    
    train(cfg)


#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=2 train.py --config ./configs/BFA_HRNet48-trainval.yml
import os
import pprint
import sys
import time
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from utils.dataloader.dataloader import ValPre
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
import importlib
from utils.visualize import print_iou, show_img
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.metric import hist_info, compute_score
from tensorboardX import SummaryWriter
import random
import numpy as np
from val_mm import evaluate, evaluate_msf

# from eval import evaluate_mid

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
# parser.add_argument('--save_path', '-p', default=None)

# os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    exec("from " + args.config + " import C as config")
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)

    cudnn.benchmark = True

    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

    val_loader, val_sampler = get_val_loader(
        engine, RGBXDataset, config, int(args.gpus)
    )
    logger.info(f"val dataset len:{len(val_loader)*int(args.gpus)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(
            time.strftime("%b%d_%d-%H-%M", time.localtime())
        )
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n" + pp.pformat(config))

    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        single_GPU=(not engine.distributed),
    )
    # weight=torch.load('checkpoints/NYUv2_DFormer_Large.pth')['model']
    # w_list=list(weight.keys())
    # # for k in w_list:
    # #     weight[k[7:]] = weight[k]
    # print('load model')
    # model.load_state_dict(weight)

    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch * config.warm_up_epoch,
    )

    if engine.distributed:
        logger.info(".............distributed training.............")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=True,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()

    logger.info("begin trainning:")
    best_miou = 0.0
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "class_names": config.class_names,
    }
    # val_pre = ValPre()
    # val_dataset = RGBXDataset(data_setting, 'val', val_pre)
    # test_loader, test_sampler = get_test_loader(engine, RGBXDataset,config)
    all_dev = [0]
    # segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, None,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, config,args.verbose, args.save_path,args.show_image)

    miou, best_miou = 0.0, 0.0
    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
        pbar = tqdm(
            range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format
        )
        dataloader = iter(train_loader)

        sum_loss = 0
        i = 0
        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch["data"]
            gts = minibatch["label"]
            modal_xs = minibatch["modal_x"]

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            loss = model(imgs, modal_xs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = (
                    "Epoch {}/{}".format(epoch, config.nepochs)
                    + " Iter {}/{}:".format(idx + 1, config.niters_per_epoch)
                    + " lr=%.4e" % lr
                    + " loss=%.4f total_loss=%.4f"
                    % (reduce_loss.item(), (sum_loss / (idx + 1)))
                )

            else:
                sum_loss += loss
                print_str = (
                    "Epoch {}/{}".format(epoch, config.nepochs)
                    + " Iter {}/{}:".format(idx + 1, config.niters_per_epoch)
                    + " lr=%.4e" % lr
                    + " loss=%.4f total_loss=%.4f" % (loss, (sum_loss / (idx + 1)))
                )

            del loss
            pbar.set_description(print_str, refresh=False)

        if (engine.distributed and (engine.local_rank == 0)) or (
            not engine.distributed
        ):
            tb.add_scalar("train_loss", sum_loss / len(pbar), epoch)
        logger.info(print_str)

        if (
            epoch % 1 == 0 and epoch > int(config.checkpoint_start_epoch)
        ) or epoch == 1:
            torch.cuda.empty_cache()
            if engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    all_metrics = evaluate_msf(
                        model,
                        val_loader,
                        config,
                        device,
                        [0.5, 0.75, 1.0, 1.25, 1.5],
                        True,
                        engine,
                    )
                    if engine.local_rank == 0:
                        metric = all_metrics[0]
                        for other_metric in all_metrics[1:]:
                            metric.update_hist(other_metric.hist)
                        ious, miou = metric.compute_iou()
                        acc, macc = metric.compute_pixel_acc()
                        f1, mf1 = metric.compute_f1()
                        if miou > best_miou:
                            best_miou = miou
                            engine.save_and_link_checkpoint(
                                config.log_dir,
                                config.log_dir,
                                config.log_dir_link,
                                infor="_miou_" + str(miou),
                                metric=miou,
                            )
                        print("miou", miou, "best", best_miou)
            elif not engine.distributed:
                with torch.no_grad():
                    model.eval()
                    device = torch.device("cuda")
                    metric = evaluate_msf(
                        model,
                        val_loader,
                        config,
                        device,
                        [0.5, 0.75, 1.0, 1.25, 1.5],
                        True,
                        engine,
                    )
                    ious, miou = metric.compute_iou()
                    acc, macc = metric.compute_pixel_acc()
                    f1, mf1 = metric.compute_f1()
                    # print('miou',miou)
                # print('acc, macc, f1, mf1, ious, miou',acc, macc, f1, mf1, ious, miou)
                # print('miou',miou)
                if miou > best_miou:
                    best_miou = miou
                    engine.save_and_link_checkpoint(
                        config.log_dir,
                        config.log_dir,
                        config.log_dir_link,
                        infor="_miou_" + str(miou),
                        metric=miou,
                    )
                print("miou", miou, "best", best_miou)
            logger.info(
                f"Epoch {epoch} validation result: mIoU {miou}, best mIoU {best_miou}"
            )

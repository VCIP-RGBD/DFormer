import os
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
from utils.dataloader.dataloader import get_train_loader,get_val_loader
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
from val_mm import evaluate,evaluate_msf
# from eval import evaluate_mid

# SEED=1
# # np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic=False
# torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='train config file path')
parser.add_argument('--gpus', help='used gpu number')
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument('-v', '--verbose', default=False, action='store_true')
parser.add_argument('--epochs', default=0)
parser.add_argument('--show_image', '-s', default=False,
                    action='store_true')
parser.add_argument('--save_path', default=None)
parser.add_argument('--checkpoint_dir')
parser.add_argument('--continue_fpath')
# parser.add_argument('--save_path', '-p', default=None)
logger = get_logger()

# os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    exec('from ' + args.config+' import config')
 

    cudnn.benchmark = True
   

    val_loader, val_sampler = get_val_loader(engine, RGBXDataset,config,int(args.gpus))
    print(len(val_loader))
    

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, norm_layer=BatchNorm2d)
    weight=torch.load(args.continue_fpath)['model']

    print('load model')
    model.load_state_dict(weight)
    
    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=val_loader, model=model)
    
    logger.info('begin testing:')
    best_miou=0.0
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    # val_pre = ValPre()
    # val_dataset = RGBXDataset(data_setting, 'val', val_pre)
    # test_loader, test_sampler = get_test_loader(engine, RGBXDataset,config)
    all_dev = [0]
    
    # segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, None,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, config,args.verbose, args.save_path,args.show_image)
    
    
    
    if engine.distributed:
        print('multi GPU test')
        with torch.no_grad():
            model.eval()
            device = torch.device('cuda')
            # all_metrics=evaluate(model, val_loader,config, device,engine)
            all_metrics=evaluate_msf(model, val_loader,config, device, [0.5,0.75,1.0,1.25,1.5], True, engine)
            if engine.local_rank == 0:
                metric = all_metrics[0]
                for other_metric in all_metrics[1:]:
                    metric.update_hist(other_metric.hist)
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                print(miou,'---------')
    else:
        with torch.no_grad():
            model.eval()
            device = torch.device('cuda')
            # metric=evaluate(model, val_loader,config, device, engine)
            # print('acc, macc, f1, mf1, ious, miou',acc, macc, f1, mf1, ious, miou)
            metric = evaluate_msf(model, val_loader,config, device,[0.5,0.75,1.0,1.25,1.5],True,engine)
            ious, miou = metric.compute_iou()
            acc, macc = metric.compute_pixel_acc()
            f1, mf1 = metric.compute_f1()
            print('miou',miou)
        
       
import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

remoteip = os.popen("pwd").read()
C.root_dir = "datasets"
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = "dataset_name"
C.dataset_path = osp.join(C.root_dir, "dataset_name")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = True
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = True
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = None
C.num_eval_imgs = None
C.num_classes = N
C.class_names = []

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = "DFormer-Large"  # Remember change the path below.
C.pretrained_model = "checkpoints/pretrained/DFormer_Large.pth.tar"
C.decoder = "ham"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10
C.channels = [96, 192, 288, 576]


C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.15
C.aux_rate = 0.0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  # [0.75, 1, 1.25] #
C.eval_flip = True  # False #
C.eval_crop_size = [480, 640]  # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 250
C.checkpoint_step = 25

"""Path Config"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath("checkpoints/" + C.dataset_name + "_" + C.backbone)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(
    osp.join(C.log_dir, "checkpoint")
)  #'/mnt/sda/repos/2023_RGBX/pretrained/'#osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"

if __name__ == "__main__":
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument("-tb", "--tensorboard", default=False, action="store_true")
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()

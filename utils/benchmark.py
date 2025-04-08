import argparse
from models.builder import EncoderDecoder as segmodel
import numpy as np
import torch
import torch.nn as nn
from importlib import import_module
from thop import profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    args = parser.parse_args()
    # config network and criterion
    config = getattr(import_module(args.config), "C")
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d
    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)
    dump_input = torch.ones(1, 3, 480, 640).to(device)
    input_shape = (3, 480, 640)
    flops, params = profile(model, inputs=(dump_input, dump_input))
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))

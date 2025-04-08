# Modified from: https://blog.csdn.net/Caesar6666/article/details/117926306

import argparse
from importlib import import_module
import numpy as np
import torch
from torch.backends import cudnn
import tqdm
from models.builder import EncoderDecoder as segmodel
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
args = parser.parse_args()
# config network and criterion
cfg = getattr(import_module(args.config), "C")

criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=cfg.background)
BatchNorm2d = nn.SyncBatchNorm
model = segmodel(
    cfg=cfg,
    criterion=criterion,
    norm_layer=BatchNorm2d,
    syncbn=True,
).cuda()
cudnn.benchmark = True

device = "cuda:0"
repetitions = 300

dummy_input = (torch.rand(1, 3, 480, 640).cuda(), torch.rand(1, 3, 480, 640).cuda())

# 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
print("warm up ...\n")
with torch.no_grad():
    for _ in range(100):
        _ = model(*dummy_input)

# synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
torch.cuda.synchronize()


# 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# 初始化一个时间容器
timings = np.zeros((repetitions, 1))

print("testing ...\n")
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = model(*dummy_input)
        ender.record()
        torch.cuda.synchronize()  # 等待GPU任务完成
        curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
        timings[rep] = curr_time

avg = timings.sum() / repetitions
print(f"\nAvg Latency={avg}ms\n")

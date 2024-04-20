import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C

C.seed = 12345

# remoteip = os.popen('pwd').read()
C.root_dir = "datasets"  # os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

from eecs598 import *
from a4_helper import *
from eecs598.grad import *
from eecs598.utils import *
from one_stage_detector import *
import multiprocessing
import torch
from common import *
import time
import torch
import torchvision
from two_stage_detector import *

from torch import nn

from a4_helper import train_detector
from common import DetectorBackboneWithFPN
from two_stage_detector import RPN

reset_seed(0)
dummy_images = torch.randn(2, 3, 224, 224).to(device='cuda')
gt_boxes = torch.randn(2, 40, 5).to(device='cuda')

DEVICE = 'cuda'
# Create a wrapper module to contain backbone + RPN:
FPN_CHANNELS = 64
backbone = DetectorBackboneWithFPN(out_channels=FPN_CHANNELS)
rpn = RPN(
    fpn_channels=FPN_CHANNELS,
    stem_channels=[FPN_CHANNELS, FPN_CHANNELS],
    batch_size_per_image=16,
    anchor_stride_scale=8,
    anchor_aspect_ratios=[0.5, 1.0, 2.0],
    anchor_iou_thresholds=(0.3, 0.6),
    pre_nms_topk=400,
    post_nms_topk=80,
)
# fmt: off
faster_rcnn = FasterRCNN(
    backbone, rpn, num_classes=20, roi_size=(7, 7),
    stem_channels=[FPN_CHANNELS, FPN_CHANNELS],
    batch_size_per_image=32,
)

GOOGLE_DRIVE_PATH = "D:\AppData\AI\A4\Data"

faster_rcnn.forward(dummy_images, gt_boxes)
faster_rcnn.train()

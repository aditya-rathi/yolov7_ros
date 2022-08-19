#!/usr/bin/env python

from models.yolo import Model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_param_file', type=str)
parser.add_argument('--hyp_file', type=str)
opt, _ = parser.parse_known_args()
m = Model(cfg = '/home/cerlab-ugv/submodule_ws/src/yolov7_pytorch_ros/config/yolov7-mask.yaml')
print(opt.hyp_file)


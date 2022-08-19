#!/usr/bin/env python

import torch
from models.yolo import Model
import os

weights = torch.load('/home/cerlab-ugv/submodule_ws/src/yolov7_ros/src/yolov7-mask-statedict.pt')
yolo = Model(cfg='/home/cerlab-ugv/submodule_ws/src/yolov7_ros/config/yolov7-mask-test.yaml', ch=3, nc=80)
yolo.float().eval()
yolo.load_state_dict(weights)



#yolo = Model(cfg='/home/cerlab-ugv/submodule_ws/src/yolov7_ros/config/yolov7-mask.yaml')
#yolo.half()
#yolo.load_state_dict(weights)
# weights['model']
print(1)


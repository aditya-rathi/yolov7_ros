#!/usr/bin/env python

import argparse
import torch
import cv2
import yaml
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image

from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

class YoloObjectDetector:
    def __init__(self, opt):
        rospy.init_node('yolo_detector',anonymous=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(opt.root_dir+'/data/'+opt.hyp_file) as f:
            self.hyp = yaml.load(f, Loader=yaml.FullLoader)
        self.weights = torch.load(opt.root_dir+'/src/yolov7-mask.pt')
        self.model = self.weights['model']
        self.model = self.model.half().to(self.device)
        self.model.eval()
        self.bridge = CvBridge()
        rospy.Subscriber(opt.sub_topic,Image,self.cam_callback)
        self.pub = rospy.Publisher('detected_img',Image,queue_size=1)
        self.rate = rospy.Rate(20)

    def cam_callback(self,msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image = letterbox(image, 640, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        self.image = image.half()
        self.output = self.model(self.image)
        self.publisher()
    
    def publisher(self):
        inf_out, train_out, attn, mask_iou, bases, sem_output = self.output['test'], self.output['bbox_and_cls'], self.output['attn'], self.output['mask_iou'], self.output['bases'], self.output['sem']
        bases = torch.cat([bases, sem_output], dim=1)
        nb, _, height, width = self.image.shape
        print(height)
        names = self.model.names
        pooler_scale = self.model.pooler_scale
        pooler = ROIPooler(output_size=self.hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
        self.output, output_mask, output_mask_score, output_ac, output_ab = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, self.hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)

        # print(self.output)
        pred, pred_masks = self.output[0], output_mask[0]
        base = bases[0]
        bboxes = Boxes(pred[:, :4])
        original_pred_masks = pred_masks.view(-1, self.hyp['mask_resolution'], self.hyp['mask_resolution'])
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
        pred_masks_np = pred_masks.detach().cpu().numpy()
        pred_cls = pred[:, 5].detach().cpu().numpy()
        pred_conf = pred[:, 4].detach().cpu().numpy()
        nimg = self.image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int)
        pnimg = nimg.copy()

        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
            if conf < 0.25:
                continue
            color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]                               
                                
            pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
            pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = '%s %.3f' % (names[int(cls)], conf)
            t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
            c2 = bbox[0] + t_size[0], bbox[1] - t_size[1] - 3
            pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), c2, color, -1, cv2.LINE_AA)  # filled
            pnimg = cv2.putText(pnimg, label, (bbox[0], bbox[1] - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        data = self.bridge.cv2_to_imgmsg(pnimg)
        if not rospy.is_shutdown():
            self.pub.publish(data)
            self.rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--model_param_file', type=str)
    parser.add_argument('--hyp_file', type=str)
    parser.add_argument('--sub_topic', type=str)
    opt, _  = parser.parse_known_args()

    detector = YoloObjectDetector(opt)
    rospy.spin()









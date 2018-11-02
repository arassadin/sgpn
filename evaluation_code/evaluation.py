import matplotlib
import numpy as np
from timer import timer

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform as sktf
from BinaryReader import gt_reader
from mAP import DetectionMAP
from visualization import visualize

import argparse
import math
import pickle
import os

DEBUG = False

# multiple
def unmold_mask(mask, bbox, scene_info):
    """

    :param mask:
    :param bbox:
    :param scene_info:
    :return:
    """
    num_mask = len(mask)
    full_mask = np.zeros(shape=(num_mask, *scene_info[:3]), dtype=np.uint8)
    for i in range(num_mask):
        x1, y1, z1, x2, y2, z2 = bbox[i]
        mask_x = mask[i].shape[0]
        mask_y = mask[i].shape[1]
        mask_z = mask[i].shape[2]
        # Put the mask in the right location.
        full_mask[i, int(x1):int(x1+mask_x), int(y1):int(y1+mask_y), int(z1): int(z1+mask_z)] = mask[i]
    return full_mask

def box_overlap(box1, box2):
    inter = max(min(box1[3], box2[3]) - max(box1[0], box2[0]), 0) * max(min(box1[4], box2[4]) - max(box1[1], box2[1]), 0) * max(min(box1[5], box2[5]) - max(box1[2], box2[2]), 0)
    if inter == 0:
        return 0
    area1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    area2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    return inter / (area1 + area2 - inter)

def mask_overlap(mask1, mask2):
    inter = sum(sum(sum((mask1 * mask2 > 0))))
    iou = inter/sum(sum(sum(mask1 + mask2 > 0)))
    return iou

def parse_args():
    parser = argparse.ArgumentParser(description='evaluation 3d --> 2d')
    parser.add_argument('--pred', default="SGPN_merged", type=str, help='pred coords')
    parser.add_argument('--labelset', default="evaluation_code/nyu40labels.csv", type=str, help='csv of labelset')
    parser.add_argument('--gt', default='/mnt/local_datasets/Voxelization/ji/ScanNet/vox-5cm_sdf/test/scene', type=str)
    parser.add_argument('--gpu', type=str, default='3')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    timer.tic()

    args = parse_args()

    mAP_mask25 = DetectionMAP(19, ignore_class=[0], overlap_threshold=0.25)
    mAP_mask5 = DetectionMAP(19, ignore_class=[0], overlap_threshold=0.5)

    gt_data = gt_reader(args.gt, args.labelset)
    predictions = os.listdir(args.pred)
    
    for idx, scene_id in enumerate(predictions):
        print(scene_id)
        predict = pickle.load(open(os.path.join(args.pred, scene_id.strip()), 'rb'))
        blob = gt_data.get_gt(scene_id.strip() + '__0__', args.labelset)

        if False:
            visualize(output_dir=os.path.join('./output', str(idx) + 'pred'),
                      data=np.where(blob['data'][0] <=1.0, 1, 0),
                      bbox=None,
                      mask=None)

            #visualize(output_dir=os.path.join('./output', str(idx) + 'gt'),
            #          data=None,
            #          bbox=gt_box,
            #          mask=None)

        gt_box = blob['gt_box'][:, :6]
        gt_mask = blob['gt_mask']
        gt_class = blob['gt_box'][:, 6]

        pred_box = np.zeros((len(predict['rois']), 6))
        pred_mask = []
        pred_class = np.zeros(len(predict['rois']))
        pred_conf = np.zeros(len(predict['rois']))
        for roi_ind, roi_key in enumerate(predict['rois'].keys()):
            pred_box[roi_ind] = predict['rois'][roi_key]
            pred_conf[roi_ind] = predict['scores'][roi_key]
            pred_class[roi_ind] = predict['class'][roi_key]
            pred_mask.append(predict['masks'][roi_key])

        mAP_mask25.evaluate_mask(pred_mask, pred_class, pred_box, pred_conf, gt_mask, gt_class, gt_box)
        mAP_mask5.evaluate_mask(pred_mask, pred_class, pred_box, pred_conf, gt_mask, gt_class, gt_box)

    mAP_mask25.finalize()
    print('mAP of mask@0.25: {}'.format(mAP_mask25.mAP()))
    for class_ind in range(19):
        if class_ind not in mAP_mask25.ignore_class:
            print('class {}: {}'.format(class_ind, mAP_mask25.AP(class_ind)))

    mAP_mask5.finalize()
    print('mAP of mask@0.5: {}'.format(mAP_mask5.mAP()))
    for class_ind in range(19):
        if class_ind not in mAP_mask5.ignore_class:
            print('class {}: {}'.format(class_ind, mAP_mask5.AP(class_ind)))

    timer.toc()
    print(timer.total_time())

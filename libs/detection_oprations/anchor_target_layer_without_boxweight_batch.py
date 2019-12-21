# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from libs.configs import cfgs
import numpy as np
from libs.box_utils.cython_utils.cython_bbox import bbox_overlaps
from libs.box_utils import bbox_transform


def anchor_target_layer(gt_boxes_batch, anchors):
    """
    :param gt_boxes: np.array of shape (batch, M, 5) for (x1, y1, x2, y2, label).
    :param img_shape:
    :param anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
    :return:
    """
    all_labels, all_target_delta, all_anchor_states = [], [], []
    for i in range(cfgs.BATCH_SIZE):
        gt_boxes = gt_boxes_batch[i, :, :]
        anchor_states = np.zeros((anchors.shape[0],))
        labels = np.zeros((anchors.shape[0], cfgs.CLASS_NUM))
        if gt_boxes.shape[0]:
            # [N, M]
            overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                                     np.ascontiguousarray(gt_boxes, dtype=np.float))

            argmax_overlaps_inds = np.argmax(overlaps, axis=1)
            max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

            positive_indices = max_overlaps >= cfgs.IOU_POSITIVE_THRESHOLD
            ignore_indices = (max_overlaps > cfgs.IOU_NEGATIVE_THRESHOLD) & ~positive_indices
            anchor_states[ignore_indices] = -1
            anchor_states[positive_indices] = 1

            # compute box regression targets
            target_boxes = gt_boxes[argmax_overlaps_inds]

            # compute target class labels
            labels[positive_indices, target_boxes[positive_indices, 4].astype(int) - 1] = 1
        else:
            # no annotations? then everything is background
            target_boxes = np.zeros((anchors.shape[0], gt_boxes.shape[1]))

        target_delta = bbox_transform.bbox_transform(ex_rois=anchors, gt_rois=target_boxes)
        all_labels.append(labels)
        all_target_delta.append(target_delta)
        all_anchor_states.append(anchor_states)

    return np.array(all_labels, np.float32), np.array(all_target_delta, np.float32), \
           np.array(all_anchor_states, np.float32)


if __name__ == '__main__':
    anchors = np.array([[0, 0, 4, 4],
                        [1, 1, 4, 4],
                        [4, 4, 6, 6]])

    gt_boxes = np.array([[0, 0, 5, 5, 1]])

    labels, gt_boxes, anchor_states = anchor_target_layer(gt_boxes, anchors)



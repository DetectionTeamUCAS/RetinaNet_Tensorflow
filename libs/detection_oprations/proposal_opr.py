# encoding: utf-8
from libs.configs import cfgs
from libs.box_utils import bbox_transform
from libs.box_utils import boxes_utils
import tensorflow as tf


def filter_detections(boxes, scores, is_training):
    """
    :param boxes: [-1, 4]
    :param scores: [-1, ]
    :param labels: [-1, ]
    :return:
    """
    if is_training:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.VIS_SCORE)), [-1, ])
    else:
        indices = tf.reshape(tf.where(tf.greater(scores, cfgs.FILTERED_SCORE)), [-1, ])

    if cfgs.NMS:
        filtered_boxes = tf.gather(boxes, indices)
        filtered_scores = tf.gather(scores, indices)

        # perform NMS
        xmin = filtered_boxes[:, 0]
        ymin = filtered_boxes[:, 1]
        xmax = filtered_boxes[:, 2]
        ymax = filtered_boxes[:, 3]
        filtered_boxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))
        nms_indices = tf.image.non_max_suppression(boxes=filtered_boxes,
                                                   scores=filtered_scores,
                                                   max_output_size=cfgs.MAXIMUM_DETECTIONS,
                                                   iou_threshold=cfgs.NMS_IOU_THRESHOLD)

        # filter indices based on NMS
        indices = tf.gather(indices, nms_indices)

    # add indices to list of all indices
    return indices


def postprocess_detctions(rpn_bbox_pred, rpn_cls_prob, img_shape, anchors, is_training):

    # 1. decode boxes
    boxes_pred = bbox_transform.bbox_transform_inv(boxes=anchors, deltas=rpn_bbox_pred)

    # 2. clip to img boundaries
    boxes_pred = boxes_utils.clip_boxes_to_img_boundaries(boxes=boxes_pred,
                                                          img_shape=img_shape)

    return_boxes_pred = []
    return_scores = []
    return_labels = []
    for j in range(0, cfgs.CLASS_NUM):
        indices = filter_detections(boxes_pred, rpn_cls_prob[:, j], is_training)
        tmp_boxes_pred = tf.reshape(tf.gather(boxes_pred, indices), [-1, 4])
        tmp_scores = tf.reshape(tf.gather(rpn_cls_prob[:, j], indices), [-1, ])

        return_boxes_pred.append(tmp_boxes_pred)
        return_scores.append(tmp_scores)
        return_labels.append(tf.ones_like(tmp_scores)*(j+1))

    return_boxes_pred = tf.concat(return_boxes_pred, axis=0)
    return_scores = tf.concat(return_scores, axis=0)
    return_labels = tf.concat(return_labels, axis=0)

    return return_boxes_pred, return_scores, return_labels

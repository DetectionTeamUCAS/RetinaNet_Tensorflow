# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
cls : motorbike|| Recall: 0.9969230769230769 || Precison: 0.012111244019138757|| AP: 0.8608716836726598
cls : bus|| Recall: 0.9812206572769953 || Precison: 0.010832944591302546|| AP: 0.8665177973232026
cls : cow|| Recall: 0.9918032786885246 || Precison: 0.014500569237222122|| AP: 0.8700933653540773
cls : cat|| Recall: 0.994413407821229 || Precison: 0.019201725997842502|| AP: 0.8901733848581077
cls : pottedplant|| Recall: 0.89375 || Precison: 0.009184918748795684|| AP: 0.5459186868755065
cls : sheep|| Recall: 0.987603305785124 || Precison: 0.013457964975505377|| AP: 0.8678004330444818
cls : aeroplane|| Recall: 0.9649122807017544 || Precison: 0.02806981729100745|| AP: 0.8901971050794585
cls : boat|| Recall: 0.9467680608365019 || Precison: 0.008034849951597289|| AP: 0.7574725882243678
cls : bicycle|| Recall: 0.9792284866468842 || Precison: 0.01154047910473859|| AP: 0.8696655881282686
cls : car|| Recall: 0.9816819317235637 || Precison: 0.021344775146643492|| AP: 0.8910282687046879
cls : tvmonitor|| Recall: 0.9642857142857143 || Precison: 0.007807981492192018|| AP: 0.8129339370651543
cls : person|| Recall: 0.9783568904593639 || Precison: 0.04236477698722362|| AP: 0.8492903923966436
cls : bottle|| Recall: 0.9253731343283582 || Precison: 0.007948572370469406|| AP: 0.7258658516344598
cls : chair|| Recall: 0.9616402116402116 || Precison: 0.008080920357916967|| AP: 0.643958822348765
cls : sofa|| Recall: 0.9874476987447699 || Precison: 0.005221354454744574|| AP: 0.7747622585263062
cls : horse|| Recall: 0.9942528735632183 || Precison: 0.019444756659548163|| AP: 0.8831973097154177
cls : train|| Recall: 0.9432624113475178 || Precison: 0.016586643387167175|| AP: 0.8595258237569346
cls : bird|| Recall: 0.9673202614379085 || Precison: 0.02792628467199195|| AP: 0.8461742818633661
cls : dog|| Recall: 0.9979550102249489 || Precison: 0.02050764834425954|| AP: 0.8791514392825484
cls : diningtable|| Recall: 0.9805825242718447 || Precison: 0.0026451908596870294|| AP: 0.7533332150753614
mAP is : 0.8168966116464886 ()

"""

# ------------------------------------------------
VERSION = 'RetinaNet_VOC0712_20190526'
NET_NAME = 'resnet101_v1d'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2,3,4,5,6,7"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = (11540 + 5000) * 2

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
FIXED_BLOCKS = 1  # allow 0~3
FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
USE_07_METRIC = True

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 5e-4 * NUM_GPU * BATCH_SIZE
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*20
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pascal'  # 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 600
IMG_MAX_LENGTH = 1000
CLASS_NUM = 20

# --------------------------------------------- Network_config
BATCH_SIZE = 1
SUBNETS_WEIGHTS_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
SUBNETS_BIAS_INITIALIZER = tf.constant_initializer(value=0.0)
PROBABILITY = 0.01
FINAL_CONV_BIAS_INITIALIZER = tf.constant_initializer(value=-math.log((1.0 - PROBABILITY) / PROBABILITY))
WEIGHT_DECAY = 1e-4

# ---------------------------------------------Anchor config
LEVEL = ['P3', 'P4', 'P5', 'P6', 'P7']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
ANCHOR_STRIDE = [8, 16, 32, 64, 128]
ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True

# --------------------------------------------RPN config
SHARE_NET = True
USE_P5 = True
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4

NMS = True
NMS_IOU_THRESHOLD = 0.5
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.01
VIS_SCORE = 0.5



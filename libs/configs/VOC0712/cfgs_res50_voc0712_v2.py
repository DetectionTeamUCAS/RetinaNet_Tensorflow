# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
cls : train|| Recall: 0.9574468085106383 || Precison: 0.011264550043806583|| AP: 0.8429614751356598
cls : boat|| Recall: 0.9619771863117871 || Precison: 0.005608636857389878|| AP: 0.7247602142711771
cls : motorbike|| Recall: 0.9938461538461538 || Precison: 0.00780136705069681|| AP: 0.8620336717055882
cls : aeroplane|| Recall: 0.968421052631579 || Precison: 0.015942698706099816|| AP: 0.877200533188366
cls : person|| Recall: 0.9882950530035336 || Precison: 0.031197060853440043|| AP: 0.8549563913368059
cls : pottedplant|| Recall: 0.9145833333333333 || Precison: 0.006810847710065781|| AP: 0.5597989508143375
cls : sofa|| Recall: 0.99581589958159 || Precison: 0.004477303083319224|| AP: 0.7809330043853933
cls : car|| Recall: 0.9900083263946711 || Precison: 0.015191781872077275|| AP: 0.8929651092188676
cls : cat|| Recall: 0.994413407821229 || Precison: 0.0132925098947054|| AP: 0.8900856486739736
cls : horse|| Recall: 0.9913793103448276 || Precison: 0.009228793836770725|| AP: 0.8822391183389151
cls : tvmonitor|| Recall: 0.961038961038961 || Precison: 0.006890771952695782|| AP: 0.7956874756219561
cls : sheep|| Recall: 0.9834710743801653 || Precison: 0.008470956719817768|| AP: 0.847823073416635
cls : dog|| Recall: 0.9959100204498977 || Precison: 0.0186640095044648|| AP: 0.8711714077751838
cls : bus|| Recall: 0.9859154929577465 || Precison: 0.006062180652983459|| AP: 0.8646568186454162
cls : chair|| Recall: 0.9708994708994709 || Precison: 0.007137300661221315|| AP: 0.6544968051248526
cls : bicycle|| Recall: 0.9881305637982196 || Precison: 0.00703704486380254|| AP: 0.8543458412723929
cls : bird|| Recall: 0.9760348583877996 || Precison: 0.016314639475600873|| AP: 0.8255116265985485
cls : diningtable|| Recall: 0.9854368932038835 || Precison: 0.0024620086594787332|| AP: 0.7423151271987897
cls : bottle|| Recall: 0.9381663113006397 || Precison: 0.0067838421214924454|| AP: 0.7201777267756303
cls : cow|| Recall: 0.9959016393442623 || Precison: 0.00847162181006833|| AP: 0.8656722996831789
mAP is : 0.8104896159590833  (pascal_430040model.ckpt)

"""

# ------------------------------------------------
VERSION = 'RetinaNet_VOC0712_20190524'
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

MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   # 10.0  if None, will not clip

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
ANCHOR_SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]
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



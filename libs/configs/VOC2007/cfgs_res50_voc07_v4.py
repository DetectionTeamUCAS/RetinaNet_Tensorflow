# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
cls : sheep|| Recall: 0.9834710743801653 || Precison: 0.01343721770551039|| AP: 0.7366031642566769
cls : chair|| Recall: 0.9470899470899471 || Precison: 0.005148153207889042|| AP: 0.5386288781309365
cls : bus|| Recall: 0.9812206572769953 || Precison: 0.005191127890514393|| AP: 0.8133672291455416
cls : bicycle|| Recall: 0.9910979228486647 || Precison: 0.009744427587816549|| AP: 0.800899636526437
cls : motorbike|| Recall: 0.9938461538461538 || Precison: 0.006038963467075496|| AP: 0.7903998534004216
cls : dog|| Recall: 0.9897750511247444 || Precison: 0.01628204265626051|| AP: 0.8218698566712613
cls : aeroplane|| Recall: 0.9157894736842105 || Precison: 0.0318603515625|| AP: 0.763830044188939
cls : bottle|| Recall: 0.8784648187633263 || Precison: 0.004865950159442542|| AP: 0.6133434933374315
cls : train|| Recall: 0.925531914893617 || Precison: 0.0163125|| AP: 0.7771032196731267
cls : pottedplant|| Recall: 0.8916666666666667 || Precison: 0.008113590263691683|| AP: 0.4671915218987904
cls : horse|| Recall: 0.9971264367816092 || Precison: 0.010073738605353306|| AP: 0.8489630879498543
cls : boat|| Recall: 0.9467680608365019 || Precison: 0.005862271924661566|| AP: 0.5963519636065165
cls : person|| Recall: 0.9763692579505301 || Precison: 0.02421948186414958|| AP: 0.8160043258059765
cls : diningtable|| Recall: 0.9757281553398058 || Precison: 0.00147582510371159|| AP: 0.6417165363609527
cls : car|| Recall: 0.984179850124896 || Precison: 0.01689681790891157|| AP: 0.8698967787733508
cls : tvmonitor|| Recall: 0.9415584415584416 || Precison: 0.007871878393051032|| AP: 0.7485201769391812
cls : bird|| Recall: 0.9607843137254902 || Precison: 0.02073635209479475|| AP: 0.7663305161757082
cls : cat|| Recall: 0.9664804469273743 || Precison: 0.018309784621897654|| AP: 0.8533618661718122
cls : cow|| Recall: 0.9836065573770492 || Precison: 0.009199279389781134|| AP: 0.7622093974966482
cls : sofa|| Recall: 0.9916317991631799 || Precison: 0.003074129320967637|| AP: 0.6257163312044963
mAP is : 0.732615393885703
"""

# ------------------------------------------------
VERSION = 'RetinaNet_20190524'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2,3,4,5,6,7"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 5000 * 2

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
WARM_SETP = int(1.0 / 8.0 * SAVE_WEIGHTS_INTE)

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



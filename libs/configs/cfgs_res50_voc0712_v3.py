# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
cls : boat|| Recall: 0.9695817490494296 || Precison: 0.004761726919629519|| AP: 0.7281771379381616
cls : bicycle|| Recall: 0.9881305637982196 || Precison: 0.006502382254159182|| AP: 0.8489062261958693
cls : pottedplant|| Recall: 0.9395833333333333 || Precison: 0.005247664149494432|| AP: 0.5544772267130251
cls : tvmonitor|| Recall: 0.961038961038961 || Precison: 0.005156255443681845|| AP: 0.8054443156582788
cls : motorbike|| Recall: 0.9938461538461538 || Precison: 0.006123338831067887|| AP: 0.8633424176495925
cls : horse|| Recall: 0.9942528735632183 || Precison: 0.009731949483869153|| AP: 0.8800299351210068
cls : car|| Recall: 0.9933388842631141 || Precison: 0.013872415637572967|| AP: 0.8903760611050385
cls : person|| Recall: 0.9860865724381626 || Precison: 0.02783284088217327|| AP: 0.8557435827115095
cls : aeroplane|| Recall: 0.9859649122807017 || Precison: 0.011732286752118909|| AP: 0.8786021097248355
cls : train|| Recall: 0.9680851063829787 || Precison: 0.010527939531834484|| AP: 0.830439540076824
cls : sofa|| Recall: 1.0 || Precison: 0.0033251248660906827|| AP: 0.7736534869486386
cls : sheep|| Recall: 0.9917355371900827 || Precison: 0.006887447626700339|| AP: 0.8304997616000941
cls : dog|| Recall: 0.9938650306748467 || Precison: 0.012805311833056676|| AP: 0.8685913031298029
cls : bottle|| Recall: 0.9381663113006397 || Precison: 0.006485657852068041|| AP: 0.7242882524763133
cls : bus|| Recall: 0.9906103286384976 || Precison: 0.005803718780943998|| AP: 0.8669273157836549
cls : cow|| Recall: 0.9877049180327869 || Precison: 0.0075783780384264645|| AP: 0.8606713765311622
cls : bird|| Recall: 0.9738562091503268 || Precison: 0.011930817274328724|| AP: 0.8311803084919606
cls : diningtable|| Recall: 0.970873786407767 || Precison: 0.0017596959245442388|| AP: 0.730199218623714
cls : cat|| Recall: 0.9888268156424581 || Precison: 0.010336370007007708|| AP: 0.8831441748779935
cls : chair|| Recall: 0.9708994708994709 || Precison: 0.005643332180063815|| AP: 0.6334387740339388
mAP is : 0.8069066262695707 (pascal_430040model.ckpt)

"""

# ------------------------------------------------
VERSION = 'RetinaNet_VOC0712_20190525'
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
IMG_SHORT_SIDE_LEN = 800
IMG_MAX_LENGTH = 1333
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



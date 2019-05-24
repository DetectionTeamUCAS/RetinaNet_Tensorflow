# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
cls : cat|| Recall: 0.9860335195530726 || Precison: 0.014905206266098045|| AP: 0.8527470573973246
cls : dog|| Recall: 0.9897750511247444 || Precison: 0.021586905133580126|| AP: 0.8274904377033
cls : aeroplane|| Recall: 0.9403508771929825 || Precison: 0.02553110412498809|| AP: 0.7767873078576267
cls : diningtable|| Recall: 0.9805825242718447 || Precison: 0.0015802855466458049|| AP: 0.6266256425436841
cls : sofa|| Recall: 0.9874476987447699 || Precison: 0.0036989436067834864|| AP: 0.6896703646112717
cls : chair|| Recall: 0.9523809523809523 || Precison: 0.00560245885694277|| AP: 0.5297096851641577
cls : boat|| Recall: 0.9467680608365019 || Precison: 0.008553173948887056|| AP: 0.6212390013387435
cls : horse|| Recall: 0.9683908045977011 || Precison: 0.010542781166901298|| AP: 0.8207037469800831
cls : motorbike|| Recall: 0.9876923076923076 || Precison: 0.008685064935064934|| AP: 0.7891094623537858
cls : bus|| Recall: 0.9812206572769953 || Precison: 0.005848280493606067|| AP: 0.7896720643953054
cls : bottle|| Recall: 0.8848614072494669 || Precison: 0.004798076144890338|| AP: 0.5944787858656694
cls : tvmonitor|| Recall: 0.961038961038961 || Precison: 0.007717980809345015|| AP: 0.744114515720316
cls : bicycle|| Recall: 0.9910979228486647 || Precison: 0.006523182688176243|| AP: 0.8000690709871533
cls : train|| Recall: 0.925531914893617 || Precison: 0.018238993710691823|| AP: 0.7820253371438621
cls : sheep|| Recall: 0.9710743801652892 || Precison: 0.014397745374341379|| AP: 0.7390184562935749
cls : car|| Recall: 0.9766860949208993 || Precison: 0.01623844066670358|| AP: 0.8465859465895317
cls : cow|| Recall: 0.9877049180327869 || Precison: 0.013388145103049831|| AP: 0.7609817867782512
cls : person|| Recall: 0.974160777385159 || Precison: 0.02294217386329356|| AP: 0.8104809262654363
cls : bird|| Recall: 0.9477124183006536 || Precison: 0.026685479418440586|| AP: 0.7601748193240988
cls : pottedplant|| Recall: 0.8729166666666667 || Precison: 0.006048969220996708|| AP: 0.4708156926277445
mAP is : 0.731625005397046
"""

# ------------------------------------------------
VERSION = 'RetinaNet_20190523'
NET_NAME = 'resnet_v1_50'  # 'MobilenetV2'
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



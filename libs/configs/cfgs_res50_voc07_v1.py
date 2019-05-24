# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
cls : aeroplane|| Recall: 0.9263157894736842 || Precison: 0.03303303303303303|| AP: 0.784830711465123
cls : cat|| Recall: 0.9832402234636871 || Precison: 0.024157573261958686|| AP: 0.850236615704726
cls : person|| Recall: 0.9750441696113075 || Precison: 0.02212888383213124|| AP: 0.810026599570606
cls : boat|| Recall: 0.9429657794676806 || Precison: 0.008830022075055188|| AP: 0.5809657874405858
cls : tvmonitor|| Recall: 0.9383116883116883 || Precison: 0.008098187014879368|| AP: 0.7375754411267105
cls : cow|| Recall: 0.9836065573770492 || Precison: 0.011059398184415465|| AP: 0.7811246236306398
cls : diningtable|| Recall: 0.970873786407767 || Precison: 0.0016768957306234698|| AP: 0.6460893690943793
cls : bottle|| Recall: 0.8763326226012793 || Precison: 0.005528577770005784|| AP: 0.5819266031025552
cls : sofa|| Recall: 0.9874476987447699 || Precison: 0.003152888366376316|| AP: 0.6819754644795543
cls : motorbike|| Recall: 0.9907692307692307 || Precison: 0.007024432809773124|| AP: 0.7992360964706795
cls : sheep|| Recall: 0.9834710743801653 || Precison: 0.015635264748390488|| AP: 0.7077056201470088
cls : bird|| Recall: 0.9520697167755992 || Precison: 0.023698481561822127|| AP: 0.7697731877317816
cls : bus|| Recall: 0.9577464788732394 || Precison: 0.005901411710252257|| AP: 0.7751698924057402
cls : pottedplant|| Recall: 0.86875 || Precison: 0.006732756393696719|| AP: 0.4587773532876927
cls : dog|| Recall: 0.9959100204498977 || Precison: 0.020203277328355113|| AP: 0.8272196688960622
cls : train|| Recall: 0.9184397163120568 || Precison: 0.013024238157497738|| AP: 0.7859268074059661
cls : horse|| Recall: 0.9798850574712644 || Precison: 0.010243624019946529|| AP: 0.8180135223615187
cls : car|| Recall: 0.9783513738551207 || Precison: 0.01963175833723184|| AP: 0.843137188974707
cls : chair|| Recall: 0.9484126984126984 || Precison: 0.005720120944258738|| AP: 0.5283113983671771
cls : bicycle|| Recall: 0.9762611275964391 || Precison: 0.008112439896436938|| AP: 0.7976561577200354
mAP is : 0.7282839054691624
"""

# ------------------------------------------------
VERSION = 'RetinaNet_20190521'
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
ANCHOR_SCALE_FACTORS = None
USE_CENTER_OFFSET = True

# --------------------------------------------RPN config
SHARE_NET = True
USE_P5 = False
IOU_POSITIVE_THRESHOLD = 0.5
IOU_NEGATIVE_THRESHOLD = 0.4

NMS = True
NMS_IOU_THRESHOLD = 0.5
MAXIMUM_DETECTIONS = 100
FILTERED_SCORE = 0.01
VIS_SCORE = 0.5



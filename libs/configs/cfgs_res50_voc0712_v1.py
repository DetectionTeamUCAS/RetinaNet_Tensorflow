# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
cls : boat|| Recall: 0.9505703422053232 || Precison: 0.005342907824154218|| AP: 0.7096663579871666
cls : horse|| Recall: 0.9971264367816092 || Precison: 0.010271438297368499|| AP: 0.8762579915407834
cls : cat|| Recall: 0.9832402234636871 || Precison: 0.012731941982855283|| AP: 0.8657648932693558
cls : bottle|| Recall: 0.9147121535181236 || Precison: 0.00529309430097842|| AP: 0.6864996193318956
cls : pottedplant|| Recall: 0.925 || Precison: 0.005485341536636892|| AP: 0.5534334829931036
cls : bus|| Recall: 0.9953051643192489 || Precison: 0.006379009448155504|| AP: 0.8598362716395636
cls : car|| Recall: 0.9891756869275604 || Precison: 0.01537724736917043|| AP: 0.8843564429054561
cls : aeroplane|| Recall: 0.9543859649122807 || Precison: 0.014160029153001198|| AP: 0.8542149220907074
cls : bicycle|| Recall: 0.9881305637982196 || Precison: 0.00675963705011875|| AP: 0.8498077362487978
cls : cow|| Recall: 0.9918032786885246 || Precison: 0.009082720312265426|| AP: 0.8451994158806354
cls : dog|| Recall: 0.9979550102249489 || Precison: 0.01499231950844854|| AP: 0.8590428896283084
cls : motorbike|| Recall: 0.9938461538461538 || Precison: 0.006394773312215403|| AP: 0.8564424864150487
cls : bird|| Recall: 0.9455337690631809 || Precison: 0.017192885156280948|| AP: 0.7948506002769691
cls : tvmonitor|| Recall: 0.9512987012987013 || Precison: 0.00638038412961108|| AP: 0.7842655696224038
cls : chair|| Recall: 0.9629629629629629 || Precison: 0.006125780447989768|| AP: 0.6172742830134138
cls : sheep|| Recall: 0.987603305785124 || Precison: 0.007774886141834743|| AP: 0.8365473938440467
cls : diningtable|| Recall: 0.9805825242718447 || Precison: 0.001985121416708433|| AP: 0.7371673843361081
cls : train|| Recall: 0.9468085106382979 || Precison: 0.011325076348829318|| AP: 0.8543968318356752
cls : person|| Recall: 0.9867491166077739 || Precison: 0.027760691407730496|| AP: 0.8422619773565775
cls : sofa|| Recall: 0.9874476987447699 || Precison: 0.0038979271616153273|| AP: 0.7647898552993955
mAP is : 0.7966038202757706   (coco_463120model.ckpt)
"""

# ------------------------------------------------
VERSION = 'RetinaNet_VOC0712_20190523'
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



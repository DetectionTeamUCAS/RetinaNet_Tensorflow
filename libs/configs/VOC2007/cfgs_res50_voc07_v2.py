# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
cls : aeroplane|| Recall: 0.9438596491228071 || Precison: 0.03152836380684482|| AP: 0.7773597235133686
cls : person|| Recall: 0.9754858657243817 || Precison: 0.021128815456515397|| AP: 0.8110363404804426
cls : sofa|| Recall: 0.9916317991631799 || Precison: 0.003753563509661071|| AP: 0.6926241276381035
cls : car|| Recall: 0.980849292256453 || Precison: 0.016935994019207545|| AP: 0.8612891545940344
cls : motorbike|| Recall: 0.9876923076923076 || Precison: 0.008316708552478172|| AP: 0.7979344387167545
cls : sheep|| Recall: 0.9752066115702479 || Precison: 0.01267318225754484|| AP: 0.6992188982186315
cls : horse|| Recall: 0.9798850574712644 || Precison: 0.010116893134753457|| AP: 0.8387371790700674
cls : train|| Recall: 0.9326241134751773 || Precison: 0.01885980638221585|| AP: 0.7910990164518434
cls : pottedplant|| Recall: 0.9020833333333333 || Precison: 0.0065086355915643275|| AP: 0.4667916780665033
cls : bus|| Recall: 0.9906103286384976 || Precison: 0.005389252145484267|| AP: 0.8013951733234702
cls : diningtable|| Recall: 0.9757281553398058 || Precison: 0.001669157947184853|| AP: 0.6493900213188186
cls : tvmonitor|| Recall: 0.9642857142857143 || Precison: 0.00617386604581549|| AP: 0.7361445157222517
cls : cat|| Recall: 0.9776536312849162 || Precison: 0.016213461805716402|| AP: 0.8708244700458685
cls : bottle|| Recall: 0.8976545842217484 || Precison: 0.004023394942563887|| AP: 0.5570627230945586
cls : cow|| Recall: 0.9877049180327869 || Precison: 0.009388025398309376|| AP: 0.7709867256180059
cls : bird|| Recall: 0.954248366013072 || Precison: 0.01937024588713957|| AP: 0.7632915804610957
cls : boat|| Recall: 0.9163498098859315 || Precison: 0.005341193679218102|| AP: 0.5818730701325913
cls : dog|| Recall: 0.9938650306748467 || Precison: 0.020776333789329686|| AP: 0.8264152853325744
cls : chair|| Recall: 0.9563492063492064 || Precison: 0.005576079160271786|| AP: 0.5278822308428679
cls : bicycle|| Recall: 0.9762611275964391 || Precison: 0.007213646728644098|| AP: 0.811791972130752
mAP is : 0.7316574162386302
"""

# ------------------------------------------------
VERSION = 'RetinaNet_20190522'
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



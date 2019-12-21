# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import math

"""
cls : aeroplane|| Recall: 0.9298245614035088 || Precison: 0.014690393037308055|| AP: 0.7660086217842756
cls : horse|| Recall: 0.985632183908046 || Precison: 0.007518137781382198|| AP: 0.8268787046290528
cls : cat|| Recall: 0.9916201117318436 || Precison: 0.01215254005203341|| AP: 0.816459566212998
cls : bottle|| Recall: 0.8869936034115139 || Precison: 0.004867888319408364|| AP: 0.6310054432564516
cls : tvmonitor|| Recall: 0.9415584415584416 || Precison: 0.00633658174190447|| AP: 0.7378410310650945
cls : bird|| Recall: 0.9607843137254902 || Precison: 0.014496564872949607|| AP: 0.7759534809691434
cls : cow|| Recall: 0.9918032786885246 || Precison: 0.008217317487266554|| AP: 0.7719480426459225
cls : chair|| Recall: 0.9523809523809523 || Precison: 0.005081515985602371|| AP: 0.5370566290465252
cls : diningtable|| Recall: 0.9757281553398058 || Precison: 0.0017804627431527477|| AP: 0.6515301548750645
cls : pottedplant|| Recall: 0.8729166666666667 || Precison: 0.007105669272644021|| AP: 0.45575480742084057
cls : dog|| Recall: 0.9938650306748467 || Precison: 0.018066914498141264|| AP: 0.8087163643031827
cls : bicycle|| Recall: 0.9940652818991098 || Precison: 0.006976986358429658|| AP: 0.8328331802386089
cls : boat|| Recall: 0.9467680608365019 || Precison: 0.004885993485342019|| AP: 0.6198683933081276
cls : person|| Recall: 0.9743816254416962 || Precison: 0.02173024355406703|| AP: 0.8165996449954711
cls : train|| Recall: 0.9326241134751773 || Precison: 0.010920112938050158|| AP: 0.7916449500646859
cls : bus|| Recall: 0.971830985915493 || Precison: 0.004070476265387187|| AP: 0.8298737220933807
cls : motorbike|| Recall: 0.9907692307692307 || Precison: 0.00751669078855222|| AP: 0.810734399188341
cls : sofa|| Recall: 0.9790794979079498 || Precison: 0.0039634146341463415|| AP: 0.7046285231064454
cls : car|| Recall: 0.9833472106577852 || Precison: 0.012978735095334908|| AP: 0.8655978769543121
cls : sheep|| Recall: 0.9834710743801653 || Precison: 0.011968218847430353|| AP: 0.7497673422604396
mAP is : 0.7400350439209181

"""

# ------------------------------------------------
VERSION = 'RetinaNet_20191221'
NET_NAME = 'resnet50_v1d'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "0,1,2,3"
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

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

BATCH_SIZE = 2
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



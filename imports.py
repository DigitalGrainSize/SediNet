
## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.3 (July 2020)

###===================================================
# import libraries
import gc, os, sys, shutil

###===================================================
# import and set global variables from defaults.py
from defaults import *

global IM_HEIGHT, IM_WIDTH

global NUM_EPOCHS, SHALLOW

global VALID_BATCH_SIZE, BATCH_SIZE

VALID_BATCH_SIZE = BATCH_SIZE

global MAX_LR, OPT, USE_GPU, DO_AUG

# global MIN_DELTA, FACTOR, STOP_PATIENCE
##====================================================

# import tensorflow.compat.v1 as tf1
# config = tf1.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf1.Session(config=config)
# tf1.keras.backend.set_session(sess)

##TF/keras
from tensorflow.keras.layers import Input, Dense, MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.layers import Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate

try:
    from tensorflow.keras.utils import plot_model
except:
    pass

import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa

##SKLEARN
from sklearn.preprocessing import RobustScaler #MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

##OTHER
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import joblib
import random

import tensorflow_addons as tfa
import tqdm

from skimage.transform import AffineTransform, warp #rotate,

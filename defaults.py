
## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.3 (June 2020)

## Contains values for defaults that you may change.
## They are listed in order of likelihood that you might change them:

# set to False if you wish to use cpu (not recommended)
USE_GPU = True  ##False

# size of image in pixels. keep this consistent in training and application
IM_HEIGHT = 768 # suggestd: 512 -- 1024
IM_WIDTH = IM_HEIGHT

# max. number of training epics
NUM_EPOCHS = 50 #100

# number of images to feed the network per step in epoch
BATCH_SIZE =  [4,6,8] #suggested: 4 --16

# if True, use a smaller (shallower) network architecture
SHALLOW = True ##False=larger network

SCALE = False # if True, scale all variables before and after training. not stable on small datasets

# optimizer (gradient descent solver) good alternative == 'adam'
OPT = 'rmsprop'

## loss function for continuous models (2 choices)
CONT_LOSS = 'pinball'
## CONT_LOSS = 'mse'

## loss function for categorical (disrete) models (2 choices)
CAT_LOSS = 'focal'
#CAT_LOSS = 'categorical_crossentropy'

# a tolerance for the training. Do not change until you've researched its effects
MIN_DELTA = 0.0001

# minimum learning rate (lambda in the manuscript)
MIN_LR = 1e-5 #1e-5 -- 1e-2

# the factor applied to the learning rate when the appropriate triggers are made
FACTOR = 0.8

# training stops early if the number of successive epochs with no validation loss exceeds this number
STOP_PATIENCE = 15

# base number of conv2d filters in categorical models
BASE_CAT = 30

# base number of conv2d filters in continuous models
BASE_CONT = 30

# number of Dense units for categorical prediction
CAT_DENSE_UNITS = 128

# number of Dense units for continuous prediction
CONT_DENSE_UNITS = 1024

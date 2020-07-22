
## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.3 (July 2020)

## Contains values for defaults that you may change.
## They are listed in order of likelihood that you might change them:

# size of image in pixels. keep this consistent in training and application
# suggestd: 512 -- 1024 (larger = larger GPU required)
# integer
IM_HEIGHT = 768 #768
IM_WIDTH = IM_HEIGHT #1024 #IM_HEIGHT

# number of images to feed the network per step in epoch #suggested: 4 --16
# integer
#BATCH_SIZE = 12

#use an ensemble of batch sizes like this
BATCH_SIZE = [12,13,14]

# if True, use a smaller (shallower) network architecture
##True or False ##False=larger network
SHALLOW = False #True

## if True, carry out data augmentation. 2 x number of images used in training
##True or False
DO_AUG = False # True

# maximum learning rate ##1e-1 -- 1e-4
MAX_LR = 1e-3

# max. number of training epics (20 -100)
# integer
NUM_EPOCHS = 100

## loss function for continuous models (2 choices)
CONT_LOSS = 'pinball'
#CONT_LOSS = 'mse'

## loss function for categorical (disrete) models (2 choices)
CAT_LOSS = 'focal'
#CAT_LOSS = 'categorical_crossentropy'

# optimizer (gradient descent solver) good alternative == 'rmsprop'
OPT = 'rmsprop' #'adam'

# base number of conv2d filters in categorical models
# integer
BASE_CAT = 30

# base number of conv2d filters in continuous models
# integer
BASE_CONT = 30

# number of Dense units for continuous prediction
# integer
CONT_DENSE_UNITS = 1024 #512

# number of Dense units for categorical prediction
# integer
CAT_DENSE_UNITS = 128

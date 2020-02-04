
## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.2 (Feb 4 2020)

## Contains values for defaults that you may change.
## They are listed in order of likelihood that you might change them:

# set to False if you wish to use cpu (not recommended)
use_gpu = True  ##False

# size of image in pixels. keep this consistent in training and application
IM_HEIGHT = 1024 #768 mighter be better for color imagery
IM_WIDTH = IM_HEIGHT

# max. number of training epics
num_epochs = 200

# number of images to feed the network per step in epoch
batch_size = 8

# number of images
valid_batch_size = batch_size

# if True, use a smaller (shallower) network architecture
shallow = False ##True

## proportion of neurons to drop in the dropout layer
#dropout = 0.5
## this is commented because it is recommended to set this per data set using the config file 

# optimizer (gradient descent solver) good alternative == 'adam'
opt = 'rmsprop'

# a tolerance for the training. Do not change until you've researched its effects
min_delta = 0.0001

# minimum learning rate (lambda in the manuscript)
min_lr = 0.0001

# the factor applied to the learning rate when the appropriate triggers are made
factor = 0.8

# training stops early if the number of successive epochs with no validation loss exceeds this number
stop_patience = 25

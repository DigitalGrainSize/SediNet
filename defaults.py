
# ______     ______     _____     __     __   __     ______     ______  
#/\  ___\   /\  ___\   /\  __-.  /\ \   /\ "-.\ \   /\  ___\   /\__  _\ 
#\ \___  \  \ \  __\   \ \ \/\ \ \ \ \  \ \ \-.  \  \ \  __\   \/_/\ \/ 
# \/\_____\  \ \_____\  \ \____-  \ \_\  \ \_\\"\_\  \ \_____\    \ \_\ 
#  \/_____/   \/_____/   \/____/   \/_/   \/_/ \/_/   \/_____/     \/_/ 
#
# By Dr Daniel Buscombe,
# daniel.buscombe@nau.edu

# train/test split (0.5 = 50% train, 0.3 = 30% train, etc)
TRAIN_TEST_SPLIT = 0.5

# size of image in pixels. keep this consistent in training and application
IM_HEIGHT = 512
IM_WIDTH = IM_HEIGHT

# max. number of training epics
num_epochs = 100

# number of images to feed the network per step in epoch
batch_size = 8

# number of images 
valid_batch_size = batch_size

# proportion of neurons to drop in the dropout layer
dropout = 0.5

# a tolerance for the training. Do not change until you've researched its effects
epsilon = 0.0001

# minimum learning rate (lambda in the manuscript)
min_lr = 0.0001

# the factor applied to the learning rate when the appropriate triggers are made
factor = 0.8


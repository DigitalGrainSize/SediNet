
# ______     ______     _____     __     __   __     ______     ______  
#/\  ___\   /\  ___\   /\  __-.  /\ \   /\ "-.\ \   /\  ___\   /\__  _\ 
#\ \___  \  \ \  __\   \ \ \/\ \ \ \ \  \ \ \-.  \  \ \  __\   \/_/\ \/ 
# \/\_____\  \ \_____\  \ \____-  \ \_\  \ \_\\"\_\  \ \_____\    \ \_\ 
#  \/_____/   \/_____/   \/____/   \/_/   \/_/ \/_/   \/_____/     \/_/ 
#
# By Dr Daniel Buscombe,
# daniel.buscombe@nau.edu

from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'-1' == use CPU

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from PIL import Image
#from keras.layers import GlobalMaxPool2D, Dropout, Dense
#Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
#from keras.optimizers import SGD
#from keras.models import Model
#from keras.layers.core import Activation
#from keras.layers.convolutional import MaxPooling2D
import tensorflow as tf
import keras.backend as K
import gc  
from sklearn.metrics import classification_report

from keras.utils import plot_model

from sedinet_utils import *


###===================================================
def get_data_generator_1image(df, indices, for_training, batch_size=16):

    ID_POP_MAP = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    POP_ID_MAP = dict((g, i) for i, g in ID_POP_MAP.items())
 
    images, pops = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, pop = r['files'], r['pop'] 
            im = Image.open(file)#.convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)                     
            if np.ndim(im)==2:
               tmp = np.zeros((IM_HEIGHT,IM_HEIGHT,3))
               tmp[:,:,0] = im
               tmp[:,:,1] = im
               tmp[:,:,2] = im
               im = tmp                                            
            images.append(np.expand_dims(im[:,:,:3],axis=2))   
            pops.append(to_categorical(pop, len(POP_ID_MAP)))                                                                         
            if len(images) >= batch_size:
                yield np.squeeze(np.array(images)), [np.array(pops)] 
                images, pops = [], []
        if not for_training:
            break

###===================================================
def make_pop_sedinet(base, IM_HEIGHT, IM_WIDTH):
    input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
    _ = conv_block(input_layer, filters=base, bn=False, pool=False, drop=False)
    _ = conv_block(_, filters=base*2, bn=False, pool=True,drop=False)    
    _ = conv_block(_, filters=base*3, bn=False, pool=True,drop=False) 
    _ = conv_block(_, filters=base*4, bn=False, pool=True,drop=False)

    bottleneck = GlobalMaxPool2D()(_)
    bottleneck = Dropout(0.5)(bottleneck) 

     # for pop prediction
    _ = Dense(units=128, activation='relu')(bottleneck)
    pop_output = Dense(units=len(POP_ID_MAP), activation='softmax', name='pop_output')(_) 
                     
    model = Model(inputs=input_layer, outputs=[pop_output]) 
    model.compile(optimizer='adam',
                  loss={'pop_output': 'categorical_crossentropy'}, 
                  loss_weights={'pop_output': 1.}, 
                  metrics={'pop_output': 'accuracy'})                   
    print('[INFORMATION] Model summary:')                                  
    model.summary()
    return model

        
###===================================================
## user defined variables: categorical variables (populations), proportion of data to use for training (a.k.a. the "train/test split")
ID_POP_MAP = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
POP_ID_MAP = dict((g, i) for i, g in ID_POP_MAP.items())

TRAIN_TEST_SPLIT = 0.5
IM_HEIGHT = 512
IM_WIDTH = IM_HEIGHT

num_epochs = 100
batch_size = 8 #16 and larger causes step size errors
valid_batch_size = batch_size
df = pd.read_csv('grain_population/dataset_population.csv')

df['files'] = [k.strip() for k in df['files']]

df['files'] = [os.getcwd()+os.sep+f.replace('\\',os.sep) for f in df['files']]

##======================================
np.random.seed(2019)
counter = 0
while counter==0:
   p = np.random.permutation(len(df))
   train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
   train_idx = p[:train_up_to]   
   test_idx = p[train_up_to:]
   if np.min(np.bincount(df['pop'].values[train_idx]))>=3:
      if np.min(np.bincount(df['pop'].values[test_idx]))>=3:                
         counter+=1
   else:
      del p, test_idx, train_idx

##==============================================
print('[INFORMATION] Training and testing data set')
print('number of files per class in training set:')
print(np.bincount(df['pop'].values[train_idx]))
print('number of files per class in testing set:')
print(np.bincount(df['pop'].values[test_idx]))

print(str(len(train_idx))+' train files')
print(str(len(test_idx))+' test files')

##==============================================
epsilon = 0.0001
min_lr = 0.0001
factor = 0.8
base=20

##==============================================
model = make_pop_sedinet(base, IM_HEIGHT, IM_WIDTH)

##==============================================
train_gen = get_data_generator_1image(df, train_idx, True, batch_size)
valid_gen = get_data_generator_1image(df, test_idx, True, valid_batch_size)

weights_path = "pop_base"+str(base)+"_model_checkpoint.hdf5"

plot_model(model, weights_path.replace('.hdf5', '_model.png'), show_shapes=True, show_layer_names=True)

callbacks_list = [
    ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
]

print("[INFORMATION] schematic of the model has been written out to: "+weights_path.replace('.hdf5', '_model.png'))
print("[INFORMATION] weights will be written out to: "+weights_path)

##==============================================
## set checkpoint file and parameters that control early stopping, and reduction of learning rate if and when validation scores plateau upon successive epochs
reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=5, verbose=1, mode='auto', epsilon=epsilon, cooldown=5, min_lr=min_lr)

earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=15) 

model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)

callbacks_list = [model_checkpoint, reduceloss_plat, earlystop]

##==============================================
## train the model
history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=num_epochs,
                    callbacks=callbacks_list,
                    validation_data=valid_gen,
                    validation_steps=len(test_idx)//valid_batch_size)

###===================================================
## Plot the loss and accuracy as a function of epoch
plot_train_history_1var(history)
plt.savefig('pop_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_hist-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
plt.close('all')

# serialize model to JSON							  
model_json = model.to_json()
with open(weights_path.replace('.hdf5','.json'), "w") as json_file:
    json_file.write(model_json)
    
gc.collect()

##==============================================
train_gen = get_data_generator_1image(df, train_idx, False, len(train_idx))

x_train, (pops_trueT)= next(train_gen) 
pops_predT = model.predict(x_train, batch_size=1) 


pops_trueT =np.asarray(pops_trueT).argmax(axis=-1) 
pops_predT = np.asarray(pops_predT).argmax(axis=-1)     
 

test_gen = get_data_generator_1image(df, test_idx, False, len(test_idx))

x_test, (pops_true)= next(test_gen) 
pops_pred = model.predict(x_test, batch_size=1) 


pops_true =np.asarray(pops_true).argmax(axis=-1) 
pops_pred = np.asarray(pops_pred).argmax(axis=-1) 

pops_true = np.squeeze(pops_true)
pops_pred = np.squeeze(pops_pred)    
   
##==============================================
print("Classification report for pop")
print(classification_report(pops_true, pops_pred))

##==============================================
plot_confmat_pop(pops_pred, pops_true, 'pop')
plt.savefig(weights_path.replace('.hdf5','_cm.png'), dpi=300, bbox_inches='tight') 
plt.close('all')   
   
plot_confmat_pop(pops_predT, pops_trueT, 'popT')
plt.savefig(weights_path.replace('.hdf5','_cmT.png'), dpi=300, bbox_inches='tight') 
plt.close('all')   
 
K.clear_session()   
      
##==============================================      
[shutil.move(k, 'grain_population') for k in glob('*.png')]




      


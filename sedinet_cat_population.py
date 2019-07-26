
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'-1' == use CPU

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from PIL import Image
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
import tensorflow as tf
import keras.backend as K
import gc  
from sklearn.metrics import classification_report
import itertools
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model

##-------------------------------------------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Purples,
                          dolabels=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmax=1, vmin=0)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if dolabels==True:
       tick_marks = np.arange(len(classes))
       plt.xticks(tick_marks, classes, fontsize=3) #, rotation=60 
       plt.yticks(tick_marks, classes, fontsize=3)

       plt.ylabel('True label',fontsize=4)
       plt.xlabel('Estimated label',fontsize=4)

    else:
       plt.axis('off')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j]>0:
           plt.text(j, i, str(cm[i, j])[:4],fontsize=5,horizontalalignment="center",color="white" if cm[i, j] > 0.6 else "black")
    plt.tight_layout()
    return cm

##-------------------------------------------------------------
def plot_confmat(y_pred, y_true, prefix):
   """
   This function generates and plots a confusion matrix
   """

   base = prefix+'_'

   y = y_pred.copy()
   del y_pred
   l = y_true.copy()
   del y_true

   l = l.astype('float')
   ytrue = l.flatten()
   ypred = y.flatten()

   ytrue = ytrue[~np.isnan(ytrue)]
   ypred = ypred[~np.isnan(ypred)]

   cm = confusion_matrix(ytrue, ypred)
   cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
   cm[np.isnan(cm)] = 0
   
   fig=plt.figure()
   plt.subplot(221)
   plot_confusion_matrix(cm, classes=['0', '1', '2', '3', '4', '5'])


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
            #colors.append(to_categorical(color, len(COLOR_ID_MAP)))   
            pops.append(to_categorical(pop, len(POP_ID_MAP)))                                                                         
            if len(images) >= batch_size:
                yield np.squeeze(np.array(images)), [np.array(pops)] 
                images, pops = [], []
        if not for_training:
            break
            


###===================================================
def  plot_train_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10)) 

#    axes[0].plot(history.history['color_output_acc'], label='color train accuracy')
#    axes[0].plot(history.history['val_color_output_acc'], label='color test accuracy')
#    axes[0].set_xlabel('Epochs')
#    axes[0].legend()

    axes[1].plot(history.history['acc'], label='pop train accuracy')
    axes[1].plot(history.history['val_acc'], label='pop test accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
        
###===================================================

ID_POP_MAP = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
POP_ID_MAP = dict((g, i) for i, g in ID_POP_MAP.items())

##=====================================
TRAIN_TEST_SPLIT = 0.5
IM_HEIGHT = 512
IM_WIDTH = IM_HEIGHT

num_epochs = 100
batch_size = 8 #16 and larger causes step size errors
valid_batch_size = batch_size
df = pd.read_csv('data_set9_pop.csv')

df['files'] = [k.strip() for k in df['files']]

df['files'] = [os.getcwd()+os.sep+f.replace('\\',os.sep) for f in df['files']]

##======================================
np.random.seed(2019)

#test_idx = np.arange(len(df))[255:]
##train_idx = np.hstack((np.arange(len(df))[:39], np.arange(len(df))[60:]))
#train_idx = np.arange(len(df))[:255]

counter = 0
while counter==0:
   p = np.random.permutation(len(df))
   train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
   train_idx = p[:train_up_to]   
   test_idx = p[train_up_to:]
   if np.min(np.bincount(df['pop'].values[train_idx]))>=3:
      if np.min(np.bincount(df['pop'].values[test_idx]))>=3:   
         #if (np.min(np.bincount(df.type.values[train_idx]))>=4 and len(np.bincount(df.type.values[train_idx]))==3):         
            #if (np.min(np.bincount(df.type.values[test_idx]))>=4 and len(np.bincount(df.type.values[test_idx]))==3): 
               #if np.min(np.bincount(df.color.values[test_idx]))>=6:               
                  counter+=1
   else:
      del p, test_idx, train_idx


print(np.bincount(df['pop'].values[train_idx]))
print(np.bincount(df['pop'].values[test_idx]))

##======================

print(str(len(train_idx))+' train files')
print(str(len(test_idx))+' test files')

epsilon = 0.0001
min_lr = 0.0001
factor = 0.8

for base in [18]: #[4,8,16,24,32]: #16]:

    input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
    #_ = BatchNormalization()(input_layer)

    _ = conv_block(input_layer, filters=base, bn=False, pool=False, drop=False) #original was bn=False, pool=False) 
    _ = conv_block(_, filters=base*2, bn=False, pool=True,drop=False) # original was bn=True, pool=True    
    _ = conv_block(_, filters=base*3, bn=False, pool=True,drop=False) # original was bn=True, pool=True
    _ = conv_block(_, filters=base*4, bn=False, pool=True,drop=False)

    #_ = BatchNormalization(axis=-1)(_) #added DB
    bottleneck = GlobalMaxPool2D()(_)
    bottleneck = Dropout(0.5)(bottleneck) ##added DB ##0.5
#      
#    # for color prediction
#    _ = Dense(units=128, activation='relu')(bottleneck) ##128
#    color_output = Dense(units=len(COLOR_ID_MAP), activation='softmax', name='color_output')(_) #sigmoid
 
     # for pop prediction
    _ = Dense(units=128, activation='relu')(bottleneck) ##128
    pop_output = Dense(units=len(POP_ID_MAP), activation='softmax', name='pop_output')(_) #sigmoid
         
                  
    model = Model(inputs=input_layer, outputs=[pop_output]) 
    model.compile(optimizer='adam', #'rmsprop', # adam
                  loss={'pop_output': 'categorical_crossentropy'}, 
                  loss_weights={'pop_output': 1.}, 
                  metrics={'pop_output': 'accuracy'})                   
                                   
    model.summary()

    train_gen = get_data_generator_1image(df, train_idx, True, batch_size)
    valid_gen = get_data_generator_1image(df, test_idx, True, valid_batch_size)
    
    weights_path = "pop_base"+str(base)+"_model_checkpoint.hdf5"


    plot_model(model, weights_path.replace('.hdf5', '_model.png'), show_shapes=True, show_layer_names=True)
    
    callbacks_list = [
        ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
    ]

    reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=5, verbose=1, mode='auto', epsilon=epsilon, cooldown=5, min_lr=min_lr)

    earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=15) 

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)

    callbacks_list = [model_checkpoint, reduceloss_plat, earlystop] #earlystop	

    ##==============================================

    history = model.fit_generator(train_gen,
                        steps_per_epoch=len(train_idx)//batch_size,
                        epochs=num_epochs,
                        callbacks=callbacks_list,
                        validation_data=valid_gen,
                        validation_steps=len(test_idx)//valid_batch_size)

    ###===================================================
    plot_train_history(history)
    #plt.show()
    plt.savefig('pop_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_hist-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
    plt.close('all')

#    # load the new model weights							  
#    model.load_weights(weights_path)

    # serialize model to JSON							  
    model_json = model.to_json()
    with open(weights_path.replace('.hdf5','.json'), "w") as json_file:
        json_file.write(model_json)
        
    gc.collect()
    
    #====
    train_gen = get_data_generator_1image(df, train_idx, False, len(train_idx))

    x_train, (pops_trueT)= next(train_gen) 
    pops_predT = model.predict(x_train, batch_size=1) 

    #colors_trueT =np.asarray(colors_trueT).argmax(axis=-1) 
    #colors_predT = np.asarray(colors_predT).argmax(axis=-1) 
    
    pops_trueT =np.asarray(pops_trueT).argmax(axis=-1) 
    pops_predT = np.asarray(pops_predT).argmax(axis=-1)     
     
    
    test_gen = get_data_generator_1image(df, test_idx, False, len(test_idx))

    x_test, (pops_true)= next(test_gen) 
    pops_pred = model.predict(x_test, batch_size=1) 

#    colors_true =np.asarray(colors_true).argmax(axis=-1) 
#    colors_pred = np.asarray(colors_pred).argmax(axis=-1) 
#    
#    colors_true = np.squeeze(colors_true)
#    colors_pred = np.squeeze(colors_pred)
    
    pops_true =np.asarray(pops_true).argmax(axis=-1) 
    pops_pred = np.asarray(pops_pred).argmax(axis=-1) 
    
    pops_true = np.squeeze(pops_true)
    pops_pred = np.squeeze(pops_pred)    
       
           
    print("Classification report for pop")
    print(classification_report(pops_true, pops_pred))
    plot_confmat(pops_pred, pops_true, 'pop')
    plt.savefig(weights_path.replace('.hdf5','_cm.png'), dpi=300, bbox_inches='tight') 
    plt.close('all')      
    plot_confmat(pops_predT, pops_trueT, 'popT')
    plt.savefig(weights_path.replace('.hdf5','_cmT.png'), dpi=300, bbox_inches='tight') 
    plt.close('all')    
    K.clear_session()   
      
    d = {'files': df.files.values[test_idx], 'pop':pops_pred}
    df2 = pd.DataFrame(d)
    df2.to_csv('pop.csv') 
    
    d = {'files': df.files.values[train_idx], 'pop':pops_predT}
    df2 = pd.DataFrame(d)
    df2.to_csv('pop2.csv')     
#    
#    
#import shutil
##for j in [0,1,2,3]:

##   for k in np.where(colors_pred==j)[0]:
##      try:
##         shutil.move(df.files[test_idx[k]], 'images/'+str(j))
##      except:
##         pass
##    
##   for k in np.where(colors_predT==j)[0]:
##      try:
##         shutil.move(df.files[train_idx[k]], 'images/'+str(j))
##      except:
##         pass
#               
#                 
##for j in [0,1,2]:

##   for k in np.where(df['shape'][test_idx]==j)[0]:
##      try:
##         shutil.move(df.files[test_idx[k]], 'images/'+str(j))
##      except:
##         pass
##    
##   for k in np.where(df['shape'][train_idx]==j)[0]:
##      try:
##         shutil.move(df.files[train_idx[k]], 'images/'+str(j))
##      except:
##         pass
##         
#         
#         
#             

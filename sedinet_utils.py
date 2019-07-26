
# ______     ______     _____     __     __   __     ______     ______  
#/\  ___\   /\  ___\   /\  __-.  /\ \   /\ "-.\ \   /\  ___\   /\__  _\ 
#\ \___  \  \ \  __\   \ \ \/\ \ \ \ \  \ \ \-.  \  \ \  __\   \/_/\ \/ 
# \/\_____\  \ \_____\  \ \____-  \ \_\  \ \_\\"\_\  \ \_____\    \ \_\ 
#  \/_____/   \/_____/   \/____/   \/_/   \/_/ \/_/   \/_____/     \/_/ 
#
# By Dr Daniel Buscombe,
# daniel.buscombe@nau.edu

import itertools
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from keras.models import Model
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
def plot_confmat_pop(y_pred, y_true, prefix):
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
def  plot_train_history_1var(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10)) 

    axes[0].plot(history.history['loss'], label='Training loss')
    axes[0].plot(history.history['val_loss'], label='Validation loss')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()
    
    axes[1].plot(history.history['acc'], label='pop train accuracy')
    axes[1].plot(history.history['val_acc'], label='pop test accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()

###===================================================
def file_generator(df, indices, batch_size=16):
    images = []
    while True:
        for i in indices:
            r = df.iloc[i]
            file = r['files']
            images.append(file)                                           
            if len(images) >= batch_size:
                yield np.array(images)
                images = []

            
###===================================================
def conv_block(inp, filters=32, bn=True, pool=True, drop=True):
    _ = Conv2D(filters=filters, kernel_size=3, activation='relu', kernel_initializer='he_uniform')(inp)
    if bn:
        _ = BatchNormalization()(_)
    if pool:
        _ = MaxPool2D()(_)
    if drop:        
        _ = Dropout(0.2)(_)        
    return _
    
    
    
    

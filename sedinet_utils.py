
# ______     ______     _____     __     __   __     ______     ______  
#/\  ___\   /\  ___\   /\  __-.  /\ \   /\ "-.\ \   /\  ___\   /\__  _\ 
#\ \___  \  \ \  __\   \ \ \/\ \ \ \ \  \ \ \-.  \  \ \  __\   \/_/\ \/ 
# \/\_____\  \ \_____\  \ \____/  \ \_\  \ \_\\"\_\  \ \_____\    \ \_\ 
#  \/_____/   \/_____/   \/____/   \/_/   \/_/ \/_/   \/_____/     \/_/ 
#
# By Dr Daniel Buscombe,
# daniel@mardascience.com

###===================================================
# import libraries

import gc, os, shutil
## use the first available GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

## to use the CPU (not recommended):
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import itertools
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from keras.models import Model
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model
from keras.utils import to_categorical
from PIL import Image
from glob import glob
import keras.backend as K

from scipy.stats import mode

###===================================================
# import and set global variables from defaults.py
from defaults import *

global TRAIN_TEST_SPLIT, IM_HEIGHT, IM_WIDTH

global num_epochs, batch_size, valid_batch_size

global epsilon, min_lr, factor

###===================================================
def plot_train_history_9var(history, varuse):
    """
    This function makes a plot of error train/validation history for 9 variables, plus overall loss functions
    """
    fig, axes = plt.subplots(1, 10, figsize=(30, 5))
    for k in range(9):
       axes[k].plot(history.history[varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Train MAE')
       axes[k].plot(history.history['val_'+varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Val MAE')
       axes[k].set_xlabel('Epochs')
       axes[k].legend() 

    axes[9].plot(history.history['loss'], label='Training loss')
    axes[9].plot(history.history['val_loss'], label='Validation loss')
    axes[9].set_xlabel('Epochs')
    axes[9].legend()

###===================================================
def plot_train_history_8var(history, varuse):
    """
    This function makes a plot of error train/validation history for 8 variables, plus overall loss functions
    """
    fig, axes = plt.subplots(1, 9, figsize=(30, 5))
    for k in range(8):
       axes[k].plot(history.history[varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Train MAE')
       axes[k].plot(history.history['val_'+varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Val MAE')
       axes[k].set_xlabel('Epochs')
       axes[k].legend() 

    axes[8].plot(history.history['loss'], label='Training loss')
    axes[8].plot(history.history['val_loss'], label='Validation loss')
    axes[8].set_xlabel('Epochs')
    axes[8].legend()


###===================================================
def plot_train_history_7var(history, varuse):
    """
    This function makes a plot of error train/validation history for 7 variables, plus overall loss functions
    """
    fig, axes = plt.subplots(1, 8, figsize=(25, 5))
    for k in range(7):
       axes[k].plot(history.history[varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Train MAE')
       axes[k].plot(history.history['val_'+varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Val MAE')
       axes[k].set_xlabel('Epochs')
       axes[k].legend() 

    axes[7].plot(history.history['loss'], label='Training loss')
    axes[7].plot(history.history['val_loss'], label='Validation loss')
    axes[7].set_xlabel('Epochs')
    axes[7].legend()

###===================================================
def plot_train_history_6var(history, varuse):
    """
    This function makes a plot of error train/validation history for 6 variables, plus overall loss functions
    """
    fig, axes = plt.subplots(1, 7, figsize=(25, 5))
    for k in range(6):
       axes[k].plot(history.history[varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Train MAE')
       axes[k].plot(history.history['val_'+varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Val MAE')
       axes[k].set_xlabel('Epochs')
       axes[k].legend() 

    axes[6].plot(history.history['loss'], label='Training loss')
    axes[6].plot(history.history['val_loss'], label='Validation loss')
    axes[6].set_xlabel('Epochs')
    axes[6].legend()
            
###===================================================
def plot_train_history_5var(history, varuse):
    """
    This function makes a plot of error train/validation history for 5 variables, plus overall loss functions
    """
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    for k in range(5):
       axes[k].plot(history.history[varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Train MAE')
       axes[k].plot(history.history['val_'+varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Val MAE')
       axes[k].set_xlabel('Epochs')
       axes[k].legend() 

    axes[5].plot(history.history['loss'], label='Training loss')
    axes[5].plot(history.history['val_loss'], label='Validation loss')
    axes[5].set_xlabel('Epochs')
    axes[5].legend()
    
###===================================================
def plot_train_history_4var(history, varuse):
    """
    This function makes a plot of error train/validation history for 4 variables, plus overall loss functions
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for k in range(4):
       axes[k].plot(history.history[varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Train MAE')
       axes[k].plot(history.history['val_'+varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Val MAE')
       axes[k].set_xlabel('Epochs')
       axes[k].legend() 

    axes[4].plot(history.history['loss'], label='Training loss')
    axes[4].plot(history.history['val_loss'], label='Validation loss')
    axes[4].set_xlabel('Epochs')
    axes[4].legend()
        
###===================================================
def plot_train_history_3var(history, varuse):
    """
    This function makes a plot of error train/validation history for 3 variables, plus overall loss functions
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for k in range(3):
       axes[k].plot(history.history[varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Train MAE')
       axes[k].plot(history.history['val_'+varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Val MAE')
       axes[k].set_xlabel('Epochs')
       axes[k].legend() 

    axes[3].plot(history.history['loss'], label='Training loss')
    axes[3].plot(history.history['val_loss'], label='Validation loss')
    axes[3].set_xlabel('Epochs')
    axes[3].legend()        
        
###===================================================
def plot_train_history_2var(history, varuse):
    """
    This function makes a plot of error train/validation history for 2 variables, plus overall loss functions
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for k in range(2):
       axes[k].plot(history.history[varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Train MAE')
       axes[k].plot(history.history['val_'+varuse[k]+'_output_mean_absolute_error'], label=varuse[k]+' Val MAE')
       axes[k].set_xlabel('Epochs')
       axes[k].legend() 

    axes[2].plot(history.history['loss'], label='Training loss')
    axes[2].plot(history.history['val_loss'], label='Validation loss')
    axes[2].set_xlabel('Epochs')
    axes[2].legend() 

###===================================================
def predict_test_train_cont(df, train_idx, test_idx, vars, models, weights_path, name, base):
    """
    This function creates makes predcitions on test and train data
    """
    ##==============================================
    ## make predictions on training data
    
    if len(vars)==1:
       train_gen = get_data_generator_1vars(df, train_idx, False, vars, len(train_idx))
    elif len(vars)==2:
       train_gen = get_data_generator_2vars(df, train_idx, False, vars, len(train_idx))       
    elif len(vars)==3:
       train_gen = get_data_generator_3vars(df, train_idx, False, vars, len(train_idx))
    elif len(vars)==4:
       train_gen = get_data_generator_4vars(df, train_idx, False, vars, len(train_idx))       
    elif len(vars)==5:
       train_gen = get_data_generator_5vars(df, train_idx, False, vars, len(train_idx))       
    elif len(vars)==6:
       train_gen = get_data_generator_6vars(df, train_idx, False, vars, len(train_idx))
    elif len(vars)==7:
       train_gen = get_data_generator_7vars(df, train_idx, False, vars, len(train_idx))       
    elif len(vars)==8:
       train_gen = get_data_generator_8vars(df, train_idx, False, vars, len(train_idx))       
    elif len(vars)==9:
       train_gen = get_data_generator_9vars(df, train_idx, False, vars, len(train_idx))
    else:
       print('Currently supports up to 9 variables ... exiting')
       import sys
       sys.exit(2)   
       
                                   
    x_train, tmp = next(train_gen)   
    if len(vars)>1:    
       counter = 0
       for v in vars:
          exec(v+'_trueT = np.squeeze(tmp[counter])')
          counter +=1
    else:
       exec(vars[0]+'_trueT = np.squeeze(tmp)')

    for v in vars:
       exec(v+'_PT = []')
    #PT = []      
    for model in models:   
       tmp = model.predict(x_train, batch_size=1)
       if len(vars)>1:    
          counter = 0
          for v in vars:
             exec(v+'_PT.append(np.squeeze(tmp[counter]))')
             counter +=1
       else:
          exec(vars[0]+'_PT.append(np.asarray(np.squeeze(tmp)))') #.argmax(axis=-1))')
          
    if len(vars)>1:
       for k in range(len(vars)):  
          exec(vars[k]+'_predT = np.squeeze(np.mean(np.asarray('+vars[k]+'_PT), axis=0))')
    else:   
       exec(vars[0]+'_predT = np.squeeze(np.mean(np.asarray('+vars[0]+'_PT), axis=0))')
       
    ## make predictions on testing data
    if len(vars)==1:    
       test_gen = get_data_generator_1vars(df, test_idx, False, vars, len(test_idx))
    elif len(vars)==2:    
       test_gen = get_data_generator_2vars(df, test_idx, False, vars, len(test_idx))       
    elif len(vars)==3:    
       test_gen = get_data_generator_3vars(df, test_idx, False, vars, len(test_idx))       
    elif len(vars)==4:    
       test_gen = get_data_generator_4vars(df, test_idx, False, vars, len(test_idx))       
    elif len(vars)==5:    
       test_gen = get_data_generator_5vars(df, test_idx, False, vars, len(test_idx))
    elif len(vars)==6:    
       test_gen = get_data_generator_6vars(df, test_idx, False, vars, len(test_idx))       
    elif len(vars)==7:    
       test_gen = get_data_generator_7vars(df, test_idx, False, vars, len(test_idx))       
    elif len(vars)==8:    
       test_gen = get_data_generator_8vars(df, test_idx, False, vars, len(test_idx))       
    elif len(vars)==9:    
       test_gen = get_data_generator_9vars(df, test_idx, False, vars, len(test_idx))       
    else:
       print('Currently supports up to 9 variables ... exiting')
       import sys
       sys.exit(2)   


    x_test, tmp = next(test_gen)   
    if len(vars)>1:    
       counter = 0
       for v in vars:
          exec(v+'_true = np.squeeze(tmp[counter])')
          counter +=1
    else:
       exec(vars[0]+'_true = np.squeeze(tmp)')
 
 
    for v in vars:
       exec(v+'_P = []')
    #P = [];      
    for model in models:   
       
       tmp = model.predict(x_test, batch_size=1)
       if len(vars)>1:    
          counter = 0
          for v in vars:
             #exec(v+'_pred = np.squeeze(tmp[counter])')
             exec(v+'_P.append(np.squeeze(tmp[counter]))')             
             counter +=1
       else:
          #exec(vars[0]+'_pred = np.squeeze(tmp)')
          #P.append(pred)
          exec(vars[0]+'_P.append(np.asarray(np.squeeze(tmp)))') #.argmax(axis=-1))')

    #pred = np.squeeze(np.mean(np.asarray(P), axis=0))
    if len(vars)>1:
       for k in range(len(vars)):  
          exec(vars[k]+'_pred = np.squeeze(np.mean(np.asarray('+vars[k]+'_P), axis=0))')
    else:
      exec(vars[0]+'_pred = np.squeeze(np.mean(np.asarray('+vars[0]+'_P), axis=0))')    


    if len(vars)==9:    
       nrows = 3; ncols = 3
    elif len(vars)==8:    
       nrows = 4; ncols = 2
    elif len(vars)==7:    
       nrows = 4; ncols = 2           
    elif len(vars)==6:    
       nrows = 3; ncols = 2
    elif len(vars)==5:    
       nrows = 3; ncols = 2       
    elif len(vars)==4:    
       nrows = 2; ncols = 2       
    elif len(vars)==3:    
       nrows = 3; ncols = 1      
    elif len(vars)==2:    
       nrows = 2; ncols = 1      
    elif len(vars)==1:    
       nrows = 1; ncols = 1
       
    ## make a plot                  
    fig = plt.figure(figsize=(4*nrows,4*ncols))
    labs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for k in range(1,1+(nrows*ncols)):
       plt.subplot(nrows,ncols,k)
       plt.plot(eval(vars[k-1]+'_trueT'), eval(vars[k-1]+'_predT'), 'ko', markersize=3)
       plt.plot(eval(vars[k-1]+'_true'), eval(vars[k-1]+'_pred'), 'bx', markersize=5)
       plt.plot([5, 1000], [5, 1000], 'k', lw=2)
       plt.xscale('log'); plt.yscale('log')
       plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(eval(vars[k-1]+'_pred') - eval(vars[k-1]+'_true')) / eval(vars[k-1]+'_true'))))[:5]+' %',  fontsize=8, color='b')
       plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(eval(vars[k-1]+'_predT') - eval(vars[k-1]+'_trueT')) / eval(vars[k-1]+'_trueT'))))[:5]+' %', fontsize=8)
       plt.xlim(10,1300); plt.ylim(10,1300)
       plt.title(r''+labs[k-1]+') '+vars[k-1], fontsize=8, loc='left')
                    
    #plt.show()
    plt.savefig(name+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_xy-base'+str(base)+'_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    del fig   

###===================================================
def get_data_generator_1image(df, indices, for_training, ID_MAP, var, batch_size=16):
    """
    This function creates a dataset generator consisting of batches of images and corresponding one-hot-encoded labels describing the sediment in each image
    """
    try:
        ID_MAP2 = dict((g, i) for i, g in ID_MAP.items())
    except:
        ID_MAP = dict(zip(np.arange(ID_MAP), [str(k) for k in range(ID_MAP)]))
        ID_MAP2 = dict((g, i) for i, g in ID_MAP.items())
        
    images, pops = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, pop = r['files'], r[var]
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
            pops.append(to_categorical(pop, len(ID_MAP2)))                                                                         
            if len(images) >= batch_size:
                yield np.squeeze(np.array(images)), [np.array(pops)] 
                images, pops = [], []
        if not for_training:
            break

###===================================================
def predict_test_train_cat(df, train_idx, test_idx, var, models, classes, weights_path):
   """
   This function creates makes predcitions on test and train data, prints a classification report, and prints confusion matrices
   """
   
   ##==============================================
   ## make predictions on training data
   train_gen = get_data_generator_1image(df, train_idx, False, len(classes), var, len(train_idx))
   x_train, (trueT)= next(train_gen) 

   ## make predictions on testing data
   test_gen = get_data_generator_1image(df, test_idx, False, len(classes), var, len(test_idx))
   x_test, (true)= next(test_gen) 
   
   true = np.squeeze(np.asarray(true).argmax(axis=-1) )
   trueT = np.squeeze(np.asarray(trueT).argmax(axis=-1) )
         
   P = []; PT = []      
   for model in models:   
      predT = model.predict(x_train, batch_size=1) 
      pred = model.predict(x_test, batch_size=1)
      predT = np.asarray(predT).argmax(axis=-1)      
      pred = np.asarray(pred).argmax(axis=-1) 
      P.append(pred)
      PT.append(predT)
      
   pred = np.squeeze(mode(np.asarray(P), axis=0)[0])
   predT = np.squeeze(mode(np.asarray(PT), axis=0)[0])
      
   ##==============================================
   ## print a classification report to screen, showing f1, precision, recall and accuracy
   print("==========================================")
   print("Classification report for "+var)
   print(classification_report(true, pred))

   ##==============================================
   ## create figures showing confusion matrices for train and test data sets
   plot_confmat(pred, true, var, classes)
   plt.savefig(weights_path.replace('.hdf5','_cm.png'), dpi=300, bbox_inches='tight') 
   plt.close('all')   
   
   plot_confmat(predT, trueT, var+'T',classes)  
   plt.savefig(weights_path.replace('.hdf5','_cmT.png'), dpi=300, bbox_inches='tight') 
   plt.close('all')   

###===================================================
def tidy(res_folder, name, base):
    """
    This function moves training outputs to a specific folder
    """
    
    pngfiles = glob('*.png')
    jsonfiles = glob('*.json')
    hfiles = glob('*.hdf5')

    if base is not '':
       ##because you could have other files from other model runs in there ...
       pngfiles = [p for p in pngfiles if p.startswith(name)]
       jsonfiles = [p for p in jsonfiles if p.startswith(name)]    
       hfiles = [p for p in hfiles if p.startswith(name)]

       pngfiles = [p for p in pngfiles if 'base'+str(base) in p]
       jsonfiles = [p for p in jsonfiles if 'base'+str(base) in p]
       hfiles = [p for p in hfiles if 'base'+str(base) in p]    
                    
    try:
       for p in pngfiles:
          os.remove(os.getcwd()+os.sep+res_folder+os.sep+p)                  
    except:
       pass
    try:
       for j in jsonfiles:
          os.remove(os.getcwd()+os.sep+res_folder+os.sep+j)                  
    except:
       pass      
    try:
       for h in hfiles:
          os.remove(os.getcwd()+os.sep+res_folder+os.sep+h)                  
    except:
       pass      
       
    try:
       [shutil.move(k, res_folder) for k in pngfiles]
       [shutil.move(k, res_folder) for k in hfiles]
       [shutil.move(k, res_folder) for k in jsonfiles] 
    except:
       pass

###===================================================
def random_sample(csvfile):
    """
    This function reads a csvfile with image names and labels and samples randomly
    """
    ###===================================================
    ## read the data set in, clean and modify the pathnames so they are absolute
    df = pd.read_csv(csvfile)

    df['files'] = [k.strip() for k in df['files']]

    df['files'] = [os.getcwd()+os.sep+f.replace('\\',os.sep) for f in df['files']]
    
    np.random.seed(2019)

    p = np.random.permutation(len(df))
    train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
    train_idx = p[:train_up_to]
    test_idx = p[train_up_to:]

    print(str(len(train_idx))+' train files')
    print(str(len(test_idx))+' test files')
    print("==========================================")
   
    return train_idx, test_idx, df

###===================================================
def stratify_random_sample(k, var, csvfile):
    """
    This function reads a csvfile with image names and labels, performs
    stratified random sampling by ensuring at least k examples per class in both train and test splits 
    """
    ###===================================================
    ## read the data set in, clean and modify the pathnames so they are absolute
    df = pd.read_csv(csvfile)

    df['files'] = [k.strip() for k in df['files']]

    df['files'] = [os.getcwd()+os.sep+f.replace('\\',os.sep) for f in df['files']]
    
    np.random.seed(2019)
    counter = 0
    while counter==0:
       p = np.random.permutation(len(df))
       train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
       train_idx = p[:train_up_to]   
       test_idx = p[train_up_to:]
       if np.min(np.bincount(df[var].values[train_idx]))>=k:
          if np.min(np.bincount(df[var].values[test_idx]))>=k:                
             counter+=1
       else:
          del p, test_idx, train_idx

    ##==============================================
    print("==========================================")
    print('[INFORMATION] Training and testing data set')
    print('number of files per class in training set:')
    print(np.bincount(df[var].values[train_idx]))
    print('number of files per class in testing set:')
    print(np.bincount(df[var].values[test_idx]))

    print(str(len(train_idx))+' train files')
    print(str(len(test_idx))+' test files')
    print("==========================================")
   
    return train_idx, test_idx, df

###===================================================
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

###===================================================
def plot_confmat(y_pred, y_true, prefix, classes):
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
   plot_confusion_matrix(cm, classes=classes)
   
###===================================================
def  plot_train_history_1var_mae(history):
   """
   This function plots loss and accuracy curves from the model training
   """
   #print(history.history.keys())
   
   fig, axes = plt.subplots(1, 2, figsize=(10, 10)) 

   axes[0].plot(history.history['loss'], label='Training loss')
   axes[0].plot(history.history['val_loss'], label='Validation loss')
   axes[0].set_xlabel('Epochs')
   axes[0].legend()
    
   axes[1].plot(history.history['mean_absolute_error'], label='pop train MAE')
   axes[1].plot(history.history['val_mean_absolute_error'], label='pop test MAE')
   axes[1].set_xlabel('Epochs')
   axes[1].legend()

###===================================================
def  plot_train_history_1var(history):
   """
   This function plots loss and accuracy curves from the model training
   """
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
   """
   This function creates a file generator according to a certain batch size
   """
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
def get_data_generator_9vars(df, indices, for_training, vars, batch_size=16):
    """
    This function generates data for a batch of images and 9 associated metrics
    """
    images, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s, p9s = [], [], [], [], [], [], [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1, p2, p3, p4, p5, p6, p7, p8, p9 = r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[vars[6]], r[vars[7]], r[vars[8]] 
            #r['P5'], r['P10'], r['P16'], r['P25'], r['P50'], r['P75'], r['P84'], r['P90'], r['P95']
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p1s.append(p1)
            p2s.append(p2)            
            p3s.append(p3)
            p4s.append(p4)                        
            p5s.append(p5)
            p6s.append(p6)            
            p7s.append(p7)  
            p8s.append(p8)  
            p9s.append(p9)                          
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s), np.array(p2s), np.array(p3s), np.array(p4s), np.array(p5s), np.array(p6s), np.array(p7s), np.array(p8s), np.array(p9s)]
                images, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s, p9s = [], [], [], [], [], [], [], [], [], []
        if not for_training:
            break           

###===================================================
def get_data_generator_8vars(df, indices, for_training, vars, batch_size=16):
    """
    This function generates data for a batch of images and 8 associated metrics
    """
    images, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s = [], [], [], [], [], [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1, p2, p3, p4, p5, p6, p7, p8= r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[vars[6]], r[vars[7]]
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p1s.append(p1)
            p2s.append(p2)            
            p3s.append(p3)
            p4s.append(p4)                        
            p5s.append(p5)
            p6s.append(p6)            
            p7s.append(p7)  
            p8s.append(p8)  
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s), np.array(p2s), np.array(p3s), np.array(p4s), np.array(p5s), np.array(p6s), np.array(p7s), np.array(p8s)]
                images, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s = [], [], [], [], [], [], [], [], []
        if not for_training:
            break  

###===================================================
def get_data_generator_7vars(df, indices, for_training, vars, batch_size=16):
    """
    This function generates data for a batch of images and 7 associated metrics
    """
    images, p1s, p2s, p3s, p4s, p5s, p6s, p7s = [], [], [], [], [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1, p2, p3, p4, p5, p6, p7= r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[vars[6]]
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p1s.append(p1)
            p2s.append(p2)            
            p3s.append(p3)
            p4s.append(p4)                        
            p5s.append(p5)
            p6s.append(p6)            
            p7s.append(p7)  
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s), np.array(p2s), np.array(p3s), np.array(p4s), np.array(p5s), np.array(p6s), np.array(p7s)]
                images, p1s, p2s, p3s, p4s, p5s, p6s, p7s = [], [], [], [], [], [], [], []
        if not for_training:
            break  

###===================================================
def get_data_generator_6vars(df, indices, for_training, vars, batch_size=16):
    """
    This function generates data for a batch of images and 6 associated metrics
    """
    images, p1s, p2s, p3s, p4s, p5s, p6s = [], [], [], [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1, p2, p3, p4, p5, p6= r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]]
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p1s.append(p1)
            p2s.append(p2)            
            p3s.append(p3)
            p4s.append(p4)                        
            p5s.append(p5)
            p6s.append(p6)            
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s), np.array(p2s), np.array(p3s), np.array(p4s), np.array(p5s), np.array(p6s)]
                images, p1s, p2s, p3s, p4s, p5s, p6s = [], [], [], [], [], [], []
        if not for_training:
            break  

###===================================================
def get_data_generator_5vars(df, indices, for_training, vars, batch_size=16):
    """
    This function generates data for a batch of images and 5 associated metrics
    """
    images, p1s, p2s, p3s, p4s, p5s = [], [], [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1, p2, p3, p4, p5= r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]]
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p1s.append(p1)
            p2s.append(p2)            
            p3s.append(p3)
            p4s.append(p4)                        
            p5s.append(p5)
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s), np.array(p2s), np.array(p3s), np.array(p4s), np.array(p5s)]
                images, p1s, p2s, p3s, p4s, p5s = [], [], [], [], [], []
        if not for_training:
            break  

###===================================================
def get_data_generator_4vars(df, indices, for_training, vars, batch_size=16):
    """
    This function generates data for a batch of images and 4 associated metrics
    """
    images, p1s, p2s, p3s, p4s = [], [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1, p2, p3, p4 = r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]]
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p1s.append(p1)
            p2s.append(p2)            
            p3s.append(p3)
            p4s.append(p4)                        
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s), np.array(p2s), np.array(p3s), np.array(p4s)]
                images, p1s, p2s, p3s, p4s = [], [], [], [], []
        if not for_training:
            break  
            
###===================================================
def get_data_generator_3vars(df, indices, for_training, vars, batch_size=16):
    """
    This function generates data for a batch of images and 3 associated metrics
    """
    images, p1s, p2s, p3s = [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1, p2, p3 = r['files'], r[vars[0]], r[vars[1]], r[vars[2]]
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p1s.append(p1)
            p2s.append(p2)            
            p3s.append(p3)
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s), np.array(p2s), np.array(p3s)]
                images, p1s, p2s, p3s = [], [], [], []
        if not for_training:
            break 

###===================================================
def get_data_generator_2vars(df, indices, for_training, vars, batch_size=16):
    """
    This function generates data for a batch of images and 2 associated metrics
    """
    images, p1s, p2s = [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1, p2 = r['files'], r[vars[0]], r[vars[1]]
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p1s.append(p1)
            p2s.append(p2)            
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s), np.array(p2s)]
                images, p1s, p2s = [], [], []
        if not for_training:
            break             

###===================================================
def get_data_generator_1vars(df, indices, for_training, vars, batch_size=16):
    """
    This function generates data for a batch of images and 1 associated metric
    """
    images, p1s = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1 = r['files'], r[vars[0]]
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p1s.append(p1)
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s)]
                images, p1s = [], []
        if not for_training:
            break    
                        
#            
####===================================================
#def get_data_generator_cont(df, indices, for_training, vars, batch_size=16):
#    """
#    This function generates a batch of images amd associated lists of variables according to 'vars' which are column headings in 'df'
#    """
#    exec(','.join(vars)+' = '+(len(vars)-1)*'[],'+'[]')
#    images = []
#    while True:
#        for i in indices:
#            r = df.iloc[i]
#            file = r['files']
#            
#            for var in vars:
#               exec(var+'.append('+str(r[var])+')')            
#            
#            im = Image.open(file).convert('LA')
#            im = im.resize((IM_HEIGHT, IM_HEIGHT))
#            im = np.array(im) / 255.0
#            im2 = np.rot90(im)         
#            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
#                               
#            if len(images) >= batch_size:
#                yield np.array(images), [eval('np.array('+v+')') for v in vars] 
#                exec(','.join(vars)+' = '+(len(vars)-1)*'[],'+'[]')
#                images = []
#        if not for_training:
#            break




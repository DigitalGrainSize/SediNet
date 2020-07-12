
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

global MIN_DELTA, MIN_LR, FACTOR, OPT, USE_GPU, STOP_PATIENCE

##====================================================

##OS
if USE_GPU == True:
   ##use the first available GPU
   os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
else:
   ## to use the CPU (not recommended):
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D
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
#from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report

##OTHER
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import joblib

import tensorflow_addons as tfa
import tqdm

#normalizer = rescaled independently of other samples so that its norm (l2) equals one. The Normalizer rescales the vector for each sample to have unit norm, independently of the distribution of the samples.
#minmaxsacler = MinMaxScaler rescales the data set such that all feature values are in the range [0, 1] as shown in the right panel below. However, this scaling compress all inliers in the narrow range

###===================================================
def get_data_generator_1image(df, indices, for_training, ID_MAP,
                              var, batch_size, greyscale): ##BATCH_SIZE
    """
    This function creates a dataset generator consisting of batches of images
    and corresponding one-hot-encoded labels describing the sediment in each image
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
#            im = Image.open(file)#.convert('LA')
#            im = im.resize((IM_HEIGHT, IM_HEIGHT))
#            im = np.array(im) / 255.0
#            im2 = np.rot90(im)
#            if np.ndim(im)==2:
#               tmp = np.zeros((IM_HEIGHT,IM_HEIGHT,3))
#               tmp[:,:,0] = im
#               tmp[:,:,1] = im
#               tmp[:,:,2] = im
#               im = tmp
#            images.append(np.expand_dims(im[:,:,:3],axis=2))

            if greyscale==True:
               im = Image.open(file).convert('LA')
            else:
               im = Image.open(file)
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0

            if np.ndim(im)==2:
               im = np.dstack((im, im , im)) ##np.expand_dims(im[:,:,0], axis=2)
            #print(file)
            #print(im.shape)
            im = im[:,:,:3]

            if greyscale==True:
               images.append(np.expand_dims(im[:,:,0], axis=2))
            else:
               images.append(im)

            pops.append(to_categorical(pop, len(ID_MAP2)))
            try:
               if len(images) >= batch_size:
                  yield np.squeeze(np.array(images)),np.array(pops) #[np.array(pops)]
                  images, pops = [], []
            except GeneratorExit:
               print(" ") #pass

        if not for_training:
            break


###===================================================
def  plot_train_history_1var(history):
   """
   This function plots loss and accuracy curves from the model training
   """
   fig, axes = plt.subplots(1, 2, figsize=(10, 10))

   print(history.history.keys())

   axes[0].plot(history.history['loss'], label='Training loss')
   axes[0].plot(history.history['val_loss'], label='Validation loss')
   axes[0].set_xlabel('Epochs')
   axes[0].legend()
   try:
      axes[1].plot(history.history['acc'], label='pop train accuracy')
      axes[1].plot(history.history['val_acc'], label='pop test accuracy')
   except:
      axes[1].plot(history.history['accuracy'], label='pop train accuracy')
      axes[1].plot(history.history['val_accuracy'], label='pop test accuracy')
   axes[1].set_xlabel('Epochs')
   axes[1].legend()


###===================================================
def plot_train_history_Nvar(history, varuse, N):
    """
    This function makes a plot of error train/validation history for 9 variables,
    plus overall loss functions
    """
    fig, axes = plt.subplots(1, N+1, figsize=(20, 5))
    for k in range(N):
       try:
          axes[k].plot(history.history[varuse[k]+'_output_mean_absolute_error'],
                       label=varuse[k]+' Train MAE')
          axes[k].plot(history.history['val_'+varuse[k]+'_output_mean_absolute_error'],
                       label=varuse[k]+' Val MAE')
       except:
          axes[k].plot(history.history[varuse[k]+'_output_mae'],
                       label=varuse[k]+' Train MAE')
          axes[k].plot(history.history['val_'+varuse[k]+'_output_mae'],
                       label=varuse[k]+' Val MAE')
       axes[k].set_xlabel('Epochs')
       axes[k].legend()

    axes[N].plot(np.log(history.history['loss']), label='Log Training loss')
    axes[N].plot(np.log(history.history['val_loss']), label='Log Validation loss')
    axes[N].set_xlabel('Epochs')
    axes[N].legend()


###===================================================
def predict_test_train_cat(train_df, test_df, train_idx, test_idx, var, SM,
                           classes, weights_path, greyscale, name):
   """
   This function creates makes predcitions on test and train data,
   prints a classification report, and prints confusion matrices
   """
   if type(SM) == list:
      counter = 0
      for s,wp in zip(SM, weights_path):
         exec('SM[counter].load_weights(wp)')
      counter += 1
   else:
     SM.load_weights(weights_path)

   ##==============================================
   ## make predictions on training data
   train_gen = get_data_generator_1image(train_df, train_idx, False,
               len(classes), var, len(train_idx), greyscale)
   x_train, (trueT)= next(train_gen)

   PT = []

   if type(SM) == list:
       counter = 0
       for s in SM:
           exec('tmp'+str(counter)+'=s.predict(x_train, batch_size=32)')
           exec(
              'PT.append(np.asarray(np.squeeze(tmp'+str(counter)+')))'
           )
           exec('del tmp'+str(counter))

       PT = np.median(PT, axis=0)
       #K.clear_session()
       #gc.collect()

       predT = np.squeeze(np.asarray(PT))
       del PT

   else:
     predT = SM.predict(x_train, batch_size=32)
     predT = np.asarray(predT).argmax(axis=-1)

   del train_gen, x_train

   ## make predictions on testing data
   test_gen = get_data_generator_1image(test_df, test_idx, False,
              len(classes), var, len(test_idx), greyscale)
   x_test, (true)= next(test_gen)

   PT = []

   if type(SM) == list:
       counter = 0
       for s in SM:
           exec('tmp'+str(counter)+'=s.predict(x_test, batch_size=32)')
           exec(
              'PT.append(np.asarray(np.squeeze(tmp'+str(counter)+')))'
           )
           exec('del tmp'+str(counter))

       PT = np.median(PT, axis=0)
       # K.clear_session()
       # gc.collect()

       pred = np.squeeze(np.asarray(PT))

   else:

       pred = SM.predict(x_test, batch_size=32) #1)
       pred = np.asarray(pred).argmax(axis=-1)

   del test_gen, x_test

   true = np.squeeze(np.asarray(true).argmax(axis=-1) )
   trueT = np.squeeze(np.asarray(trueT).argmax(axis=-1) )

   pred = np.squeeze(np.asarray(pred).argmax(axis=-1))#[0])
   predT = np.squeeze(np.asarray(predT).argmax(axis=-1))#[0])

   ##==============================================
   ## print a classification report to screen, showing f1, precision, recall and accuracy
   print("==========================================")
   print("Classification report for "+var)
   print(classification_report(true, pred))

   fig = plt.figure()
   ##==============================================
   ## create figures showing confusion matrices for train and test data sets
   if type(SM) == list:

      plot_confmat(pred, true, var, classes)
      plt.savefig(weights_path[0].replace('.hdf5','_cm.png').\
               replace('batch','_'.join(np.asarray(BATCH_SIZE, dtype='str'))),
               dpi=300, bbox_inches='tight')
      plt.close('all')

      plot_confmat(predT, trueT, var+'T',classes)
      plt.savefig(weights_path[0].replace('.hdf5','_cmT.png').\
               replace('batch','_'.join(np.asarray(BATCH_SIZE, dtype='str'))),
               dpi=300, bbox_inches='tight')
      plt.close('all')

   else:

      plot_confmat(pred, true, var, classes)
      plt.savefig(weights_path[0].replace('.hdf5','_cm.png'),
               dpi=300, bbox_inches='tight')
      plt.close('all')

      plot_confmat(predT, trueT, var+'T',classes)
      plt.savefig(weights_path.replace('.hdf5','_cmT.png'),
               dpi=300, bbox_inches='tight')
      plt.close('all')

   plt.close()
   del fig

###===================================================
def predict_test_train_siso_simo(train_df, test_df, train_idx, test_idx, vars,
                                 SM, weights_path, name, mode, greyscale,
                                 CS, dropout):
    """
    This function creates makes predcitions on test and train data
    """
    ##==============================================
    ## make predictions on training data
    if type(SM) == list:
        counter = 0
        for s,wp in zip(SM, weights_path):
           exec('SM[counter].load_weights(wp)')
        counter += 1
    else:
        SM.load_weights(weights_path)

    train_gen = get_data_generator_Nvars_siso_simo(train_df, train_idx, False,
                vars, len(train_idx), greyscale, CS)
    x_train, tmp = next(train_gen)

    if len(vars)>1:
       counter = 0
       for v in vars:
          exec(
          v+\
          '_trueT = np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1)))'
          )
          counter +=1
    else:
       exec(
       vars[0]+\
       '_trueT = np.squeeze(CS[0].inverse_transform(tmp[0].reshape(-1,1)))'
       )

    del tmp

    for v in vars:
       exec(v+'_PT = []')

    if type(SM) == list:
        counter = 0
        for s in SM:
            tmp=s.predict(x_train, batch_size=32)

            if len(vars)>1:
               counter2 = 0
               for v in vars:
                  exec(
                  v+\
                  '_PT.append(np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1))))'
                  )
                  counter2 +=1
            else:
               exec(
               vars[0]+\
               '_PT.append(np.asarray(np.squeeze(CS[0].inverse_transform(tmp.reshape(-1,1)))))'
               )

            del tmp

        if len(vars)>1:
           counter = 0
           for v in vars:
              exec(
              v+\
              '_PT = np.median('+v+'_PT, axis=0)'
              )
              counter +=1
        else:
           exec(
           vars[0]+\
           '_PT = np.median('+v+'_PT, axis=0)'
           )

    else:
        tmp = SM.predict(x_train, batch_size=32) #128)

        if len(vars)>1:
           counter = 0
           for v in vars:
              exec(
              v+\
              '_PT.append(np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1))))'
              )
              counter +=1
        else:
           exec(
           vars[0]+\
           '_PT.append(np.asarray(np.squeeze(CS[0].inverse_transform(tmp.reshape(-1,1)))))'
           )

        del tmp

    if len(vars)>1:
       for k in range(len(vars)):
          exec(vars[k]+'_predT = np.squeeze(np.asarray('+vars[k]+'_PT))')
    else:
       exec(vars[0]+'_predT = np.squeeze(np.asarray('+vars[0]+'_PT))')


    for v in vars:
       exec('del '+v+'_PT')


    del train_gen, x_train

    ## make predictions on testing data
    test_gen = get_data_generator_Nvars_siso_simo(test_df, test_idx, False,
               vars, len(test_idx), greyscale, CS)

    x_test, tmp = next(test_gen)
    if len(vars)>1:
       counter = 0
       for v in vars:
          exec(
          v+\
          '_true = np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1)))'
          )
          counter +=1
    else:
       exec(
       vars[0]+\
       '_true = np.squeeze(CS[0].inverse_transform(tmp[0].reshape(-1,1)))'
       )

    for v in vars:
       exec(v+'_P = []')

    del tmp


    if type(SM) == list:
        counter = 0
        for s in SM:
            tmp=s.predict(x_test, batch_size=32)

            if len(vars)>1:
               counter = 0
               for v in vars:
                  exec(
                  v+\
                  '_P.append(np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1))))'
                  )
                  counter +=1
            else:
               exec(
               vars[0]+\
               '_P.append(np.asarray(np.squeeze(CS[0].inverse_transform(tmp.reshape(-1,1)))))'
               )

            del tmp

        if len(vars)>1:
           counter = 0
           for v in vars:
              exec(
              v+\
              '_P = np.median('+v+'_P, axis=0)'
              )
              counter +=1
        else:
           exec(
           vars[0]+\
           '_P = np.median('+v+'_P, axis=0)'
           )

    else:

        tmp = SM.predict(x_test, batch_size=32) #128)
        if len(vars)>1:
           counter = 0
           for v in vars:
              exec(
              v+\
              '_P.append(np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1))))'
              )
              counter +=1
        else:
           exec(
           vars[0]+\
           '_P.append(np.asarray(np.squeeze(CS[0].inverse_transform(tmp.reshape(-1,1)))))'
           )

        del tmp

    del test_gen, x_test

    if len(vars)>1:
       for k in range(len(vars)):
          exec(vars[k]+'_pred = np.squeeze(np.asarray('+vars[k]+'_P))')
    else:
       exec(vars[0]+'_pred = np.squeeze(np.asarray('+vars[0]+'_P))')

    for v in vars:
       exec('del '+v+'_P')

    if len(vars)==9:
       nrows = 3; ncols = 3
    elif len(vars)==8:
       nrows = 4; ncols = 2
    elif len(vars)==7:
       nrows = 3; ncols = 3
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

    Z = []

    ## make a plot
    fig = plt.figure(figsize=(4*nrows,4*ncols))
    labs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for k in range(1,1+(nrows*ncols)):
      try:
         plt.subplot(nrows,ncols,k)
         x = eval(vars[k-1]+'_trueT')
         y = eval(vars[k-1]+'_predT')

         z = np.polyfit(y,x, 1)
         Z.append(z)

         y = np.polyval(z,y)
         y = np.abs(y)

         #plt.plot(x, y, 'rs', markersize=5)
         plt.plot(x, y, 'ko', markersize=5)
         plt.plot([ np.min(np.hstack((x,y))),  np.max(np.hstack((x,y)))],
                  [ np.min(np.hstack((x,y))),  np.max(np.hstack((x,y)))],
                  'k', lw=2)

         x = eval(vars[k-1]+'_true')
         y = eval(vars[k-1]+'_pred')
         y = np.abs(np.polyval(z,y))

         plt.plot(x, y, 'bx', markersize=5)

         plt.text(np.nanmin(x), 0.96*np.max(np.hstack((x,y))),'Test : '+\
                  str(np.mean(100*(np.abs(eval(vars[k-1]+'_pred') -\
                  eval(vars[k-1]+'_true')) / eval(vars[k-1]+'_true'))))[:5]+\
                  ' %',  fontsize=8, color='b')
         plt.text(np.nanmin(x), np.max(np.hstack((x,y))),'Train : '+\
                  str(np.mean(100*(np.abs(eval(vars[k-1]+'_predT') -\
                  eval(vars[k-1]+'_trueT')) / eval(vars[k-1]+'_trueT'))))[:5]+\
                  ' %', fontsize=8)
         plt.title(r''+labs[k-1]+') '+vars[k-1], fontsize=8, loc='left')

         varstring = ''.join([str(k)+'_' for k in vars])
      except:
          pass
    if type(SM) == list:
       plt.savefig(weights_path[0].replace('.hdf5', '_skill_ensemble.png').\
                replace('batch','_'.join(np.asarray(BATCH_SIZE, dtype='str'))),
                dpi=300, bbox_inches='tight')
       joblib.dump(Z, weights_path[0].replace('.hdf5','_bias.pkl').\
                replace('batch','_'.join(np.asarray(BATCH_SIZE, dtype='str'))))

    else:
       plt.savefig(weights_path.replace('.hdf5', '_skill.png'),
                dpi=300, bbox_inches='tight')
       joblib.dump(Z, weights_path.replace('.hdf5','_bias.pkl'))

    plt.close()
    del fig


###===================================================
def tidy(res_folder):
    """
    This function moves training outputs to a specific folder
    """

    pngfiles = glob('*.png')
    jsonfiles = glob('*.json')
    hfiles = glob('*.hdf5')
    tfiles = glob('*.txt')
    pfiles = glob('*.pkl')

    try:
       [shutil.move(k, res_folder) for k in pngfiles]
       [shutil.move(k, res_folder) for k in hfiles]
       [shutil.move(k, res_folder) for k in jsonfiles]
       [shutil.move(k, res_folder) for k in tfiles]
       [shutil.move(k, res_folder) for k in pfiles]
    except:
       pass


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
           plt.text(j, i, str(cm[i, j])[:4],fontsize=5,
                    horizontalalignment="center",
                    color="white" if cm[i, j] > 0.6 else "black")
    #plt.tight_layout()

    plt.xlim(-0.5, len(classes))
    plt.ylim(-0.5, len(classes))
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
def get_df(csvfile):
    """
    This function reads a csvfile with image names and labels
    and returns random indices
    """
    ###===================================================
    ## read the data set in, clean and modify the pathnames so they are absolute
    df = pd.read_csv(csvfile)

    df['files'] = [k.strip() for k in df['files']]

    df['files'] = [os.getcwd()+os.sep+f.replace('\\',os.sep) for f in df['files']]

    np.random.seed(2019)
    return np.random.permutation(len(df)), df

###===================================================
def  plot_train_history_1var_mae(history):
   """
   This function plots loss and accuracy curves from the model training
   """
   print(history.history.keys())

   fig, axes = plt.subplots(1, 2, figsize=(10, 10))

   axes[0].plot(np.log(history.history['loss']), label='Log Training loss')
   axes[0].plot(np.log(history.history['val_loss']),
                label='Log Validation loss')
   axes[0].set_xlabel('Epochs')
   axes[0].legend()

   try:
      axes[1].plot(history.history['mean_absolute_error'],
                   label='pop train MAE')
      axes[1].plot(history.history['val_mean_absolute_error'],
                   label='pop test MAE')
   except:
      axes[1].plot(history.history['mae'], label='pop train MAE')
      axes[1].plot(history.history['val_mae'], label='pop test MAE')

   axes[1].set_xlabel('Epochs')
   axes[1].legend()


###===================================================
def get_data_generator_Nvars_siso_simo(df, indices, for_training, vars,
                                       batch_size, greyscale, CS): ##BATCH_SIZE
    """
    This function generates data for a batch of images and N associated metrics
    """

    if len(vars)==1:
       images, p1s = [], []
    elif len(vars)==2:
       images, p1s, p2s = [], [], []
    elif len(vars)==3:
       images, p1s, p2s, p3s = [], [], [], []
    elif len(vars)==4:
       images, p1s, p2s, p3s, p4s = [], [], [], [], []
    elif len(vars)==5:
       images, p1s, p2s, p3s, p4s, p5s = [], [], [], [], [], []
    elif len(vars)==6:
       images, p1s, p2s, p3s, p4s, p5s, p6s =\
        [], [], [], [], [], [], []
    elif len(vars)==7:
       images, p1s, p2s, p3s, p4s, p5s, p6s, p7s =\
        [], [], [], [], [], [], [], []
    elif len(vars)==8:
       images, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s =\
        [], [], [], [], [], [], [], [], []
    elif len(vars)==9:
       images, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s, p9s =\
        [], [], [], [], [], [], [], [], [], []

    while True:
        for i in indices:
            r = df.iloc[i]
            if len(vars)==1:
               file, p1 = r['files'], r[vars[0]]
            if len(vars)==2:
               file, p1, p2 = r['files'], r[vars[0]], r[vars[1]]
            if len(vars)==3:
               file, p1, p2, p3 = \
               r['files'], r[vars[0]], r[vars[1]], r[vars[2]]
            if len(vars)==4:
               file, p1, p2, p3, p4 = \
               r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]]
            if len(vars)==5:
               file, p1, p2, p3, p4, p5 = \
               r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]]
            if len(vars)==6:
               file, p1, p2, p3, p4, p5, p6 = \
               r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]]
            if len(vars)==7:
               file, p1, p2, p3, p4, p5, p6, p7 = \
               r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[vars[6]]
            if len(vars)==8:
               file, p1, p2, p3, p4, p5, p6, p7, p8 = \
               r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[vars[6]], r[vars[7]]
            elif len(vars)==9:
               file, p1, p2, p3, p4, p5, p6, p7, p8, p9 = \
               r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[vars[6]], r[vars[7]], r[vars[8]]

            if greyscale==True:
               im = Image.open(file).convert('LA')
            else:
               im = Image.open(file)
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0

            if np.ndim(im)==2:
               im = np.dstack((im, im , im)) ##np.expand_dims(im[:,:,0], axis=2)

            im = im[:,:,:3]

            if greyscale==True:
               images.append(np.expand_dims(im[:,:,0], axis=2))
            else:
               images.append(im)

            if len(vars)==1:
               p1s.append(p1)
            elif len(vars)==2:
               p1s.append(p1); p2s.append(p2)
            elif len(vars)==3:
               p1s.append(p1); p2s.append(p2)
               p3s.append(p3);
            elif len(vars)==4:
               p1s.append(p1); p2s.append(p2)
               p3s.append(p3); p4s.append(p4)
            elif len(vars)==5:
               p1s.append(p1); p2s.append(p2)
               p3s.append(p3); p4s.append(p4)
               p5s.append(p5);
            elif len(vars)==6:
               p1s.append(p1); p2s.append(p2)
               p3s.append(p3); p4s.append(p4)
               p5s.append(p5); p6s.append(p6)
            elif len(vars)==7:
               p1s.append(p1); p2s.append(p2)
               p3s.append(p3); p4s.append(p4)
               p5s.append(p5); p6s.append(p6)
               p7s.append(p7);
            elif len(vars)==8:
               p1s.append(p1); p2s.append(p2)
               p3s.append(p3); p4s.append(p4)
               p5s.append(p5); p6s.append(p6)
               p7s.append(p7); p8s.append(p8)
            elif len(vars)==9:
               p1s.append(p1); p2s.append(p2)
               p3s.append(p3); p4s.append(p4)
               p5s.append(p5); p6s.append(p6)
               p7s.append(p7); p8s.append(p8)
               p9s.append(p9)

            if len(images) >= batch_size:
               if len(vars)==1:
                  p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
                  yield np.array(images), [np.array(p1s)]
                  images, p1s = [], []
               elif len(vars)==2:
                  p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
                  p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
                  yield np.array(images), [np.array(p1s), np.array(p2s)]
                  images, p1s, p2s = [], [], []
               elif len(vars)==3:
                  p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
                  p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
                  p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
                  yield np.array(images),[np.array(p1s), np.array(p2s), np.array(p3s)]
                  images, p1s, p2s, p3s = [], [], [], []
               elif len(vars)==4:
                  p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
                  p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
                  p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
                  p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
                  yield np.array(images),[np.array(p1s), np.array(p2s),
                        np.array(p3s), np.array(p4s)]
                  images, p1s, p2s, p3s, p4s = [], [], [], [], []
               elif len(vars)==5:
                  p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
                  p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
                  p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
                  p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
                  p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
                  yield np.array(images),[np.array(p1s), np.array(p2s), np.array(p3s),
                        np.array(p4s), np.array(p5s)]
                  images, p1s, p2s, p3s, p4s, p5s = [], [], [], [], [], []
               elif len(vars)==6:
                  p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
                  p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
                  p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
                  p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
                  p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
                  p6s = np.squeeze(CS[5].transform(np.array(p6s).reshape(-1, 1)))
                  yield np.array(images),[np.array(p1s), np.array(p2s), np.array(p3s),
                        np.array(p4s), np.array(p5s), np.array(p6s)]
                  images, p1s, p2s, p3s, p4s, p5s, p6s = \
                  [], [], [], [], [], [], []
               elif len(vars)==7:
                  p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
                  p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
                  p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
                  p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
                  p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
                  p6s = np.squeeze(CS[5].transform(np.array(p6s).reshape(-1, 1)))
                  p7s = np.squeeze(CS[6].transform(np.array(p7s).reshape(-1, 1)))
                  yield np.array(images),[np.array(p1s), np.array(p2s), np.array(p3s),
                        np.array(p4s), np.array(p5s), np.array(p6s), np.array(p7s)]
                  images, p1s, p2s, p3s, p4s, p5s, p6s, p7s = \
                  [], [], [], [], [], [], [], []
               elif len(vars)==8:
                  p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
                  p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
                  p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
                  p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
                  p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
                  p6s = np.squeeze(CS[5].transform(np.array(p6s).reshape(-1, 1)))
                  p7s = np.squeeze(CS[6].transform(np.array(p7s).reshape(-1, 1)))
                  p8s = np.squeeze(CS[7].transform(np.array(p8s).reshape(-1, 1)))
                  yield np.array(images),[np.array(p1s), np.array(p2s), np.array(p3s),
                        np.array(p4s), np.array(p5s), np.array(p6s),
                        np.array(p7s), np.array(p8s)]
                  images, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s = \
                  [], [], [], [], [], [], [], [], []
               elif len(vars)==9:
                  p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
                  p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
                  p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
                  p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
                  p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
                  p6s = np.squeeze(CS[5].transform(np.array(p6s).reshape(-1, 1)))
                  p7s = np.squeeze(CS[6].transform(np.array(p7s).reshape(-1, 1)))
                  p8s = np.squeeze(CS[7].transform(np.array(p8s).reshape(-1, 1)))
                  p9s = np.squeeze(CS[8].transform(np.array(p9s).reshape(-1, 1)))
                  try:
                     yield np.array(images),[np.array(p1s), np.array(p2s), np.array(p3s),
                           np.array(p4s), np.array(p5s), np.array(p6s),
                           np.array(p7s), np.array(p8s), np.array(p9s)]
                  except GeneratorExit:
                     print(" ") #pass
                  images, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s, p9s = \
                  [], [], [], [], [], [], [], [], [], []
        if not for_training:
            break

#
# ###===================================================
# def predict_test_train_miso_mimo(train_df, test_df, train_idx, test_idx,
#                                  vars, auxin, SM, weights_path, name, mode,
#                                  greyscale, CS, CSaux):
#     """
#     This function creates makes predcitions on test and train data
#     """
#     ##==============================================
#     ## make predictions on training data
#
#     SM.load_weights(weights_path)
#
#     train_gen = get_data_generator_Nvars_miso_mimo(train_df, train_idx, False,
#                  vars, auxin,aux_mean, aux_std, len(train_idx), greyscale)
#
#     x_train, tmp = next(train_gen)
#
#     if len(vars)>1:
#        counter = 0
#        for v in vars:
#           exec(
#           v+\
#           '_trueT = np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1)))'
#           )
#           counter +=1
#     else:
#        exec(
#        vars[0]+\
#        '_trueT = np.squeeze(CS[0].inverse_transform(tmp[0].reshape(-1,1)))'
#        )
#
#     for v in vars:
#        exec(v+'_PT = []')
#
#     del tmp
#     tmp = SM.predict(x_train, batch_size=32) #128)
#     if len(vars)>1:
#        counter = 0
#        for v in vars:
#           exec(
#               v+\
#               '_PT.append(np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1))))'
#               )
#           counter +=1
#     else:
#        exec(
#        vars[0]+\
#        '_PT.append(np.asarray(np.squeeze(CS[0].inverse_transform(tmp.reshape(-1,1)))))'
#        )
#
#
#     if len(vars)>1:
#        for k in range(len(vars)):
#           exec(
#           vars[k]+\
#           '_predT = np.squeeze(np.mean(np.asarray('+vars[k]+'_PT), axis=0))'
#           )
#     else:
#        exec(
#        vars[0]+\
#        '_predT = np.squeeze(np.mean(np.asarray('+vars[0]+'_PT), axis=0))'
#        )
#
#     ## make predictions on testing data
#     test_gen = get_data_generator_Nvars_miso_mimo(test_df, test_idx, False,
#                  vars, auxin, aux_mean, aux_std, len(test_idx), greyscale)
#
#     del tmp
#     x_test, tmp = next(test_gen)
#     if len(vars)>1:
#        counter = 0
#        for v in vars:
#           exec(v+\
#                '_true = np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1)))'
#                )
#           counter +=1
#     else:
#        exec(vars[0]+\
#        '_true = np.squeeze(CS[0].inverse_transform(tmp[0].reshape(-1,1)))'
#        )
#
#     for v in vars:
#        exec(v+'_P = []')
#
#     del tmp
#     tmp = SM.predict(x_test, batch_size=32) #128)
#     if len(vars)>1:
#        counter = 0
#        for v in vars:
#           exec(
#           v+\
#           '_P.append(np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1))))'
#           )
#           counter +=1
#     else:
#        exec(
#        vars[0]+\
#        '_P.append(np.asarray(np.squeeze(CS[0].inverse_transform(tmp.reshape(-1,1)))))'
#        )
#
#     if len(vars)>1:
#        for k in range(len(vars)):
#           exec(
#           vars[k]+\
#           '_pred = np.squeeze(np.mean(np.asarray('+vars[k]+'_P), axis=0))'
#           )
#     else:
#        exec(
#        vars[0]+\
#        '_pred = np.squeeze(np.mean(np.asarray('+vars[0]+'_P), axis=0))'
#        )
#
#
#     if len(vars)==9:
#        nrows = 3; ncols = 3
#     elif len(vars)==8:
#        nrows = 4; ncols = 2
#     elif len(vars)==7:
#        nrows = 4; ncols = 2
#     elif len(vars)==6:
#        nrows = 3; ncols = 2
#     elif len(vars)==5:
#        nrows = 3; ncols = 2
#     elif len(vars)==4:
#        nrows = 2; ncols = 2
#     elif len(vars)==3:
#        nrows = 3; ncols = 1
#     elif len(vars)==2:
#        nrows = 2; ncols = 1
#     elif len(vars)==1:
#        nrows = 1; ncols = 1
#
#     ## make a plot
#     fig = plt.figure(figsize=(4*nrows,4*ncols))
#     labs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     for k in range(1,1+(nrows*ncols)):
#       plt.subplot(nrows,ncols,k)
#       x = eval(vars[k-1]+'_trueT')
#       y = eval(vars[k-1]+'_predT')
#       plt.plot(x, y, 'ko', markersize=5)
#       plt.plot(eval(vars[k-1]+'_true'), eval(vars[k-1]+'_pred'),
#                'bx', markersize=5)
#       plt.plot([ np.min(np.hstack((x,y))),  np.max(np.hstack((x,y)))],
#                [ np.min(np.hstack((x,y))),  np.max(np.hstack((x,y)))], 'k', lw=2)
#
#       plt.text(np.nanmin(x), 0.96*np.max(np.hstack((x,y))),'Test : '+\
#                str(np.mean(100*(np.abs(eval(vars[k-1]+'_pred') -\
#                 eval(vars[k-1]+'_true')) / eval(vars[k-1]+'_true'))))[:5]+\
#                ' %',  fontsize=8, color='b')
#       plt.text(np.nanmin(x), np.max(np.hstack((x,y))),'Train : '+\
#                str(np.mean(100*(np.abs(eval(vars[k-1]+'_predT') -\
#                 eval(vars[k-1]+'_trueT')) / eval(vars[k-1]+'_trueT'))))[:5]+\
#                ' %', fontsize=8)
#       plt.title(r''+labs[k-1]+') '+vars[k-1], fontsize=8, loc='left')
#
#     varstring = ''.join([str(k)+'_' for k in vars])
#
#     plt.savefig(weights_path.replace('.hdf5', '_skill.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()
#     del fig
#

#
# ###===================================================
# def get_data_generator_Nvars_miso_mimo(df, indices, for_training, vars, auxin,
#                                        batch_size, greyscale, CS, CSaux): ##BATCH_SIZE
#     """
#     This function generates data for a batch of images and 1 auxilliary variable,
#     and N associated output metrics
#     """
#     if len(vars)==1:
#        images, a, p1s = [], [], []
#     elif len(vars)==2:
#        images, a, p1s, p2s = [], [], [], []
#     elif len(vars)==3:
#        images, a, p1s, p2s, p3s = [], [], [], [], []
#     elif len(vars)==4:
#        images, a, p1s, p2s, p3s, p4s = [], [], [], [], [], []
#     elif len(vars)==5:
#        images, a, p1s, p2s, p3s, p4s, p5s = [], [], [], [], [], [], []
#     elif len(vars)==6:
#        images, a, p1s, p2s, p3s, p4s, p5s, p6s = \
#        [], [], [], [], [], [], [], []
#     elif len(vars)==7:
#        images, a, p1s, p2s, p3s, p4s, p5s, p6s, p7s = \
#        [], [], [], [], [], [], [], [], []
#     elif len(vars)==8:
#        images, a, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s = \
#        [], [], [], [], [], [], [], [], [], []
#     elif len(vars)==9:
#        images, a, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s, p9s = \
#        [], [], [], [], [], [], [], [], [], [], []
#
#     while True:
#         for i in indices:
#             r = df.iloc[i]
#             if len(vars)==1:
#                file, p1, aa = r['files'], r[vars[0]], r[auxin]
#             if len(vars)==2:
#                file, p1, p2, aa = \
#                r['files'], r[vars[0]], r[vars[1]], r[auxin]
#             if len(vars)==3:
#                file, p1, p2, p3, aa = \
#                r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[auxin]
#             if len(vars)==4:
#                file, p1, p2, p3, p4, aa = \
#                r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[auxin]
#             if len(vars)==5:
#                file, p1, p2, p3, p4, p5, aa = \
#                r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[auxin]
#             if len(vars)==6:
#                file, p1, p2, p3, p4, p5, p6, aa = \
#                r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[auxin]
#             if len(vars)==7:
#                file, p1, p2, p3, p4, p5, p6, p7, aa =\
#                r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[vars[6]], r[auxin]
#             if len(vars)==8:
#                file, p1, p2, p3, p4, p5, p6, p7, p8, aa = \
#                r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[vars[6]], r[vars[7]], r[auxin]
#             elif len(vars)==9:
#                file, p1, p2, p3, p4, p5, p6, p7, p8, p9, aa = \
#                r['files'], r[vars[0]], r[vars[1]], r[vars[2]], r[vars[3]], r[vars[4]], r[vars[5]], r[vars[6]], r[vars[7]], r[vars[8]], r[auxin]
#
#             if greyscale==True:
#                im = Image.open(file).convert('LA')
#             else:
#                im = Image.open(file)
#             im = im.resize((IM_HEIGHT, IM_HEIGHT))
#             im = np.array(im) / 255.0
#
#             if np.ndim(im)==2:
#                im = np.dstack((im, im , im)) ##np.expand_dims(im[:,:,0], axis=2)
#
#             im = im[:,:,:3]
#
#             if greyscale==True:
#                images.append(np.expand_dims(im, axis=2))
#             else:
#                images.append(im)
#
#             if len(vars)==1:
#                p1s.append(p1); a.append(aa)
#             elif len(vars)==2:
#                p1s.append(p1); p2s.append(p2); a.append(aa)
#             elif len(vars)==3:
#                p1s.append(p1); p2s.append(p2); a.append(aa)
#                p3s.append(p3);
#             elif len(vars)==4:
#                p1s.append(p1); p2s.append(p2); a.append(aa)
#                p3s.append(p3); p4s.append(p4)
#             elif len(vars)==5:
#                p1s.append(p1); p2s.append(p2); a.append(aa)
#                p3s.append(p3); p4s.append(p4)
#                p5s.append(p5);
#             elif len(vars)==6:
#                p1s.append(p1); p2s.append(p2); a.append(aa)
#                p3s.append(p3); p4s.append(p4)
#                p5s.append(p5); p6s.append(p6)
#             elif len(vars)==7:
#                p1s.append(p1); p2s.append(p2); a.append(aa)
#                p3s.append(p3); p4s.append(p4)
#                p5s.append(p5); p6s.append(p6)
#                p7s.append(p7);
#             elif len(vars)==8:
#                p1s.append(p1); p2s.append(p2); a.append(aa)
#                p3s.append(p3); p4s.append(p4)
#                p5s.append(p5); p6s.append(p6)
#                p7s.append(p7); p8s.append(p8)
#             elif len(vars)==9:
#                p1s.append(p1); p2s.append(p2); a.append(aa)
#                p3s.append(p3); p4s.append(p4)
#                p5s.append(p5); p6s.append(p6)
#                p7s.append(p7); p8s.append(p8)
#                p9s.append(p9)
#
#
#             if len(images) >= batch_size:
#                if len(vars)==1:
#                   p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
#                   a = np.squeeze(CSaux[0].transform(np.array(a).reshape(-1, 1)))
#                   yield [np.array(a), np.array(images)], [np.array(p1s)]
#                   images, a, p1s = [], [], []
#                elif len(vars)==2:
#                   p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
#                   p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
#                   a = np.squeeze(CSaux[0].transform(np.array(a).reshape(-1, 1)))
#                   yield [np.array(a), np.array(images)],[np.array(p1s), np.array(p2s)]
#                   images, a, p1s, p2s = [], [], [], []
#                elif len(vars)==3:
#                   p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
#                   p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
#                   p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
#                   a = np.squeeze(CSaux[0].transform(np.array(a).reshape(-1, 1)))
#                   yield [np.array(a), np.array(images)],[np.array(p1s), np.array(p2s), np.array(p3s)]
#                   images, a, p1s, p2s, p3s = [], [], [], [], []
#                elif len(vars)==4:
#                   p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
#                   p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
#                   p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
#                   p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
#                   a = np.squeeze(CSaux[0].transform(np.array(a).reshape(-1, 1)))
#                   yield [np.array(a), np.array(images)],[np.array(p1s), np.array(p2s), np.array(p3s), np.array(p4s)]
#                   images, a, p1s, p2s, p3s, p4s = [], [], [], [], [], []
#                elif len(vars)==5:
#                   p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
#                   p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
#                   p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
#                   p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
#                   p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
#                   a = np.squeeze(CSaux[0].transform(np.array(a).reshape(-1, 1)))
#                   yield [np.array(a), np.array(images)],[np.array(p1s), np.array(p2s), np.array(p3s),
#                         np.array(p4s), np.array(p5s)]
#                   images, a, p1s, p2s, p3s, p4s, p5s = \
#                   [], [], [], [], [], [], []
#                elif len(vars)==6:
#                   p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
#                   p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
#                   p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
#                   p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
#                   p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
#                   p6s = np.squeeze(CS[5].transform(np.array(p6s).reshape(-1, 1)))
#                   a = np.squeeze(CSaux[0].transform(np.array(a).reshape(-1, 1)))
#                   yield [np.array(a), np.array(images)],[np.array(p1s), np.array(p2s), np.array(p3s),
#                         np.array(p4s), np.array(p5s), np.array(p6s)]
#                   images, a, p1s, p2s, p3s, p4s, p5s, p6s = \
#                   [], [], [], [], [], [], [], []
#                elif len(vars)==7:
#                   p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
#                   p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
#                   p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
#                   p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
#                   p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
#                   p6s = np.squeeze(CS[5].transform(np.array(p6s).reshape(-1, 1)))
#                   p7s = np.squeeze(CS[6].transform(np.array(p7s).reshape(-1, 1)))
#                   a = np.squeeze(CSaux[0].transform(np.array(a).reshape(-1, 1)))
#                   yield [np.array(a), np.array(images)],[np.array(p1s), np.array(p2s), np.array(p3s),
#                         np.array(p4s), np.array(p5s), np.array(p6s), np.array(p7s)]
#                   images, a, p1s, p2s, p3s, p4s, p5s, p6s, p7s = \
#                   [], [], [], [], [], [], [], [], []
#                elif len(vars)==8:
#                   p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
#                   p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
#                   p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
#                   p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
#                   p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
#                   p6s = np.squeeze(CS[5].transform(np.array(p6s).reshape(-1, 1)))
#                   p7s = np.squeeze(CS[6].transform(np.array(p7s).reshape(-1, 1)))
#                   p8s = np.squeeze(CS[7].transform(np.array(p8s).reshape(-1, 1)))
#                   a = np.squeeze(CSaux[0].transform(np.array(a).reshape(-1, 1)))
#                   yield [np.array(a), np.array(images)],[np.array(p1s), np.array(p2s), np.array(p3s),
#                         np.array(p4s), np.array(p5s), np.array(p6s),
#                         np.array(p7s), np.array(p8s)]
#                   images, a, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s = \
#                   [], [], [], [], [], [], [], [], [], []
#                elif len(vars)==9:
#                   p1s = np.squeeze(CS[0].transform(np.array(p1s).reshape(-1, 1)))
#                   p2s = np.squeeze(CS[1].transform(np.array(p2s).reshape(-1, 1)))
#                   p3s = np.squeeze(CS[2].transform(np.array(p3s).reshape(-1, 1)))
#                   p4s = np.squeeze(CS[3].transform(np.array(p4s).reshape(-1, 1)))
#                   p5s = np.squeeze(CS[4].transform(np.array(p5s).reshape(-1, 1)))
#                   p6s = np.squeeze(CS[5].transform(np.array(p6s).reshape(-1, 1)))
#                   p7s = np.squeeze(CS[6].transform(np.array(p7s).reshape(-1, 1)))
#                   p8s = np.squeeze(CS[7].transform(np.array(p8s).reshape(-1, 1)))
#                   p9s = np.squeeze(CS[8].transform(np.array(p9s).reshape(-1, 1)))
#                   a = np.squeeze(CSaux[0].transform(np.array(a).reshape(-1, 1)))
#                   try:
#                      yield [np.array(a), np.array(images)],[np.array(p1s), np.array(p2s), np.array(p3s),
#                            np.array(p4s), np.array(p5s), np.array(p6s),
#                            np.array(p7s), np.array(p8s), np.array(p9s)]
#                   except GeneratorExit:
#                      print(" ") #pass
#                   images, a, p1s, p2s, p3s, p4s, p5s, p6s, p7s, p8s, p9s = \
#                   [], [], [], [], [], [], [], [], [], [], []
#         if not for_training:
#             break

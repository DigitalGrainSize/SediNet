
#https://sanjayasubedi.com.np/deeplearning/multioutput-keras/
#https://github.com/jangedoo/age-gender-race-prediction/blob/master/notebook.ipynb

from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1' #'-1' == use CPU

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
def get_data_generator_2images(df, indices, for_training, batch_size=16):
    images, p5s, p10s, p16s, p25s, p50s, p75s, p84s, p90s, p95s = [], [], [], [], [], [], [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p5, p10, p16, p25, p50, p75, p84, p90, p95 = r['files'], r['P5'], r['P10'], r['P16'], r['P25'], r['P50'], r['P75'], r['P84'], r['P90'], r['P95']
            #print(file)
            im = Image.open(file).convert('LA')
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0
            im2 = np.rot90(im)         
            #images.append(np.expand_dims(im[:,:,0],axis=2))
            images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))            
            p5s.append(p5)
            p10s.append(p10)            
            p16s.append(p16)
            p25s.append(p25)                        
            p50s.append(p50)
            p75s.append(p75)            
            p84s.append(p84)  
            p90s.append(p90)  
            p95s.append(p95)                          
            #fs.append(file)                                
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p5s), np.array(p10s), np.array(p16s), np.array(p25s), np.array(p50s), np.array(p75s), np.array(p84s), np.array(p90s), np.array(p95s)]
                images, p5s, p10s, p16s, p25s, p50s, p75s, p84s, p90s, p95s = [], [], [], [], [], [], [], [], [], []
        if not for_training:
            break

###===================================================
def conv_block(inp, filters=32, bn=True, pool=True, drop=True):
    _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
    if bn:
        _ = BatchNormalization()(_)
    if pool:
        _ = MaxPool2D()(_)
    if drop:        
        _ = Dropout(0.2)(_) ##added DB        
    return _

def  plot_train_history(history):
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    axes[0].plot(history.history['P16_output_mean_absolute_error'], label='p16 Train MAE')
    axes[0].plot(history.history['val_P16_output_mean_absolute_error'], label='p16 Val MAE')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    axes[1].plot(history.history['P25_output_mean_absolute_error'], label='p25 Train MAE')
    axes[1].plot(history.history['val_P25_output_mean_absolute_error'], label='p25 Val MAE')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
        
    axes[2].plot(history.history['P50_output_mean_absolute_error'], label='p50 Train MAE')
    axes[2].plot(history.history['val_P50_output_mean_absolute_error'], label='p50 Val MAE')
    axes[2].set_xlabel('Epochs')
    axes[2].legend()

    axes[3].plot(history.history['P75_output_mean_absolute_error'], label='p75 Train MAE')
    axes[3].plot(history.history['val_P75_output_mean_absolute_error'], label='p75 Val MAE')
    axes[3].set_xlabel('Epochs')
    axes[3].legend()
    
    axes[4].plot(history.history['P84_output_mean_absolute_error'], label='P84 Train MAE')
    axes[4].plot(history.history['val_P84_output_mean_absolute_error'], label='P84 Val MAE')
    axes[4].set_xlabel('Epochs')
    axes[4].legend()  

    axes[5].plot(history.history['loss'], label='Training loss')
    axes[5].plot(history.history['val_loss'], label='Validation loss')
    axes[5].set_xlabel('Epochs')
    axes[5].legend()
    
###===================================================


## cycle through image sizes, models, batch sizes, etc
## ensemble?


##=====================================
TRAIN_TEST_SPLIT = 0.5
IM_HEIGHT = 512
IM_WIDTH = IM_HEIGHT*2

num_epochs = 100
batch_size = 8 #16 and larger causes step size errors
valid_batch_size = batch_size
df = pd.read_csv('data_set_400images.csv')

#for k in range(len(df)):
#   tmp=[df.P5.values[k],df.P10.values[k],df.P16.values[k],df.P25.values[k],
#        df.P50.values[k],df.P75.values[k],df.P84.values[k],
#        df.P90.values[k],df.P95.values[k]] 
#   tmp = sorted(tmp)
#   df.P5.values[k]=tmp[0]
#   df.P10.values[k]=tmp[1]
#   df.P16.values[k]=tmp[2]
#   df.P25.values[k]=tmp[3]   
#   df.P50.values[k]=tmp[4]
#   df.P75.values[k]=tmp[5]
#   df.P84.values[k]=tmp[6]
#   df.P90.values[k]=tmp[7]
#   df.P95.values[k]=tmp[8]
#   df.files.values[k] = df.files.values[k]
#df.to_csv('data_set9.csv')


df['files'] = [k.strip() for k in df['files']]

df['files'] = [os.getcwd()+os.sep+f.replace('\\',os.sep) for f in df['files']]

##======================================
np.random.seed(2019)

test_idx = np.arange(len(df))[396:]
###train_idx = np.hstack((np.arange(len(df))[:39], np.arange(len(df))[60:]))
train_idx = np.arange(len(df))[:396]

p = np.random.permutation(len(df))
train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]


n = 25
random_indices = np.random.permutation(len(df))[:n]
n_cols = 5
n_rows = int(np.ceil(n / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))
for i, img_idx in enumerate(random_indices):
    ax = axes.flat[i]
    im = Image.open(df['files'][img_idx]).convert('LA')
    #im = im.resize((IM_WIDTH, IM_HEIGHT))
    im = np.array(im) / 255.0     
    ax.imshow(im[:,:,0], cmap='gray')
    #ax.set_title('P) {} /{}/ {}/ {}/ {}'.format(int(p16_pred[img_idx]), int(p25_pred[img_idx]), int(p50_pred[img_idx]), int(p75_pred[img_idx]), int(p84_pred[img_idx]) ))
    ax.set_xlabel('T) {}- {}- {}- {}- {}'.format(int(df['P5'][img_idx]), int(df['P10'][img_idx]), int(df['P16'][img_idx]), int(df['P25'][img_idx]), int(df['P50'][img_idx]), int(df['P75'][img_idx]), int(df['P84'][img_idx]), int(df['P90'][img_idx]),int(df['P95'][img_idx]) ))
    ax.set_xlim([0,1024]); ax.set_ylim([0,1024])
    ax.plot([50,250],[50,50],'r',lw=2)
    ax.set_xticks([])
    ax.set_yticks([])

#plt.show()
plt.savefig('set9_examples.png', dpi=300, bbox_inches='tight')

##======================

print(str(len(train_idx))+' train files')
print(str(len(test_idx))+' test files')

#w16 = []; w25 = []; w50 = []
#w75 = []; w84 = []

epsilon = 0.0001
min_lr = 0.0001
factor = 0.8

for base in [18]: #[4,8,16]:

    input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 1))
    _ = BatchNormalization()(input_layer)

    ##======================================
    # model 1
    #below, default is bn=True, pool=True, drop=True
    #base = 32  ##32 orig
    _ = conv_block(_, filters=base, bn=False, pool=False, drop=False) #original was bn=False, pool=False) 
    _ = conv_block(_, filters=base*2, drop=False) # original was bn=True, pool=True    
    _ = conv_block(_, filters=base*3, drop=False) # original was bn=True, pool=True
    _ = conv_block(_, filters=base*4, drop=False)
    #_ = conv_block(_, filters=base*5, drop=False)
    #_ = conv_block(_, filters=base*6, drop=False)

    _ = BatchNormalization(axis=-1)(_) #added DB
    bottleneck = GlobalMaxPool2D()(_)
    bottleneck = Dropout(0.5)(bottleneck) ##added DB
    
    units = 256 #256 #128

    _ = Dense(units=units, activation='relu')(bottleneck)
    p5_output = Dense(units=1, activation='linear', name='P5_output')(_) #sigmoid  linear
    
    _ = Dense(units=units, activation='relu')(bottleneck)
    p10_output = Dense(units=1, activation='linear', name='P10_output')(_) #sigmoid  linear
        
    _ = Dense(units=units, activation='relu')(bottleneck)
    p16_output = Dense(units=1, activation='linear', name='P16_output')(_) #sigmoid  linear

    _ = Dense(units=units, activation='relu')(bottleneck)
    p25_output = Dense(units=1, activation='linear', name='P25_output')(_) #sigmoid

    _ = Dense(units=units, activation='relu')(bottleneck)
    p50_output = Dense(units=1, activation='linear', name='P50_output')(_) #sigmoid

    _ = Dense(units=units, activation='relu')(bottleneck)
    p75_output = Dense(units=1, activation='linear', name='P75_output')(_) #sigmoid

    _ = Dense(units=units, activation='relu')(bottleneck)
    p84_output = Dense(units=1, activation='linear', name='P84_output')(_) #sigmoid

    _ = Dense(units=units, activation='relu')(bottleneck)
    p90_output = Dense(units=1, activation='linear', name='P90_output')(_) #sigmoid

    _ = Dense(units=units, activation='relu')(bottleneck)
    p95_output = Dense(units=1, activation='linear', name='P95_output')(_) #sigmoid


    model = Model(inputs=input_layer, outputs=[p5_output, p10_output, p16_output, p25_output, p50_output, p75_output, p84_output, p90_output, p95_output])
    model.compile(optimizer='adam', 
                  loss={'P5_output': 'mse', 'P10_output': 'mse', 'P16_output': 'mse', 'P25_output': 'mse', 'P50_output': 'mse', 'P75_output': 'mse', 'P84_output': 'mse', 'P90_output': 'mse', 'P95_output': 'mse'},
                  loss_weights={'P5_output': 1., 'P10_output': 1., 'P16_output': 1., 'P25_output': 1, 'P50_output': 1., 'P75_output': 1, 'P84_output': 1., 'P90_output': 1., 'P95_output': 1.},
                  metrics={'P5_output': 'mae' , 'P10_output': 'mae', 'P16_output': 'mae', 'P25_output': 'mae', 'P50_output': 'mae', 'P75_output': 'mae', 'P84_output': 'mae', 'P90_output': 'mae', 'P95_output': 'mae'})
    model.summary()
    

    train_gen = get_data_generator_2images(df, train_idx, True, batch_size)
    valid_gen = get_data_generator_2images(df, test_idx, True, valid_batch_size)
    
    weights_path = "set9_base"+str(base)+"_model_checkpoint.hdf5"

    #from keras.utils import plot_model
    #plot_model(model, weights_path.replace('.hdf5', '_model.png'))
    
    callbacks_list = [
        ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
    ]

    reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=5, verbose=1, mode='auto', epsilon=epsilon, cooldown=5, min_lr=min_lr)

    earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=25) 

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)

    callbacks_list = [model_checkpoint, earlystop, reduceloss_plat]	

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
    plt.savefig('set9_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_hist-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
    plt.close('all')


#    # load the new model weights							  
#    model.load_weights(weights_path)

    # serialize model to JSON							  
    model_json = model.to_json()
    with open(weights_path.replace('.hdf5','.json'), "w") as json_file:
        json_file.write(model_json)
        
    gc.collect()

    #with tf.device('/cpu:0'):
    
    #====
    train_gen = get_data_generator_2images(df, train_idx, False, len(train_idx))
    x_train, (p5_trueT, p10_trueT, p16_trueT, p25_trueT, p50_trueT, p75_trueT, p84_trueT, p90_trueT, p95_trueT)= next(train_gen)
    p5_predT, p10_predT, p16_predT, p25_predT, p50_predT, p75_predT, p84_predT, p90_predT, p95_predT = model.predict(x_train, batch_size=1)
    
    test_gen = get_data_generator_2images(df, test_idx, False, len(test_idx))
    x_test, (p5_true, p10_true, p16_true, p25_true, p50_true, p75_true, p84_true, p90_true, p95_true)= next(test_gen)
    p5_pred, p10_pred, p16_pred, p25_pred, p50_pred, p75_pred, p84_pred, p90_pred, p95_pred = model.predict(x_test, batch_size=1)
   
    p50_pred = np.squeeze(p50_pred)
    p50_true = np.squeeze(p50_true)
    
    p16_pred = np.squeeze(p16_pred)
    p16_true = np.squeeze(p16_true)
          
    p25_pred = np.squeeze(p25_pred)
    p25_true = np.squeeze(p25_true)
    
    p75_pred = np.squeeze(p75_pred)
    p75_true = np.squeeze(p75_true)
    
    p84_pred = np.squeeze(p84_pred)
    p84_true = np.squeeze(p84_true)  
      
    p5_pred = np.squeeze(p5_pred)
    p5_true = np.squeeze(p5_true)
          
    p10_pred = np.squeeze(p10_pred)
    p10_true = np.squeeze(p10_true)
    
    p90_pred = np.squeeze(p90_pred)
    p90_true = np.squeeze(p90_true)
    
    p95_pred = np.squeeze(p95_pred)
    p95_true = np.squeeze(p95_true)  


    p50_predT = np.squeeze(p50_predT)
    p50_trueT = np.squeeze(p50_trueT)
    
    p16_predT = np.squeeze(p16_predT)
    p16_trueT = np.squeeze(p16_trueT)
          
    p25_predT = np.squeeze(p25_predT)
    p25_trueT = np.squeeze(p25_trueT)
    
    p75_predT = np.squeeze(p75_predT)
    p75_trueT = np.squeeze(p75_trueT)
    
    p84_predT = np.squeeze(p84_predT)
    p84_trueT = np.squeeze(p84_trueT)
    
    p5_predT = np.squeeze(p5_predT)
    p5_trueT = np.squeeze(p5_trueT)
          
    p10_predT = np.squeeze(p10_predT)
    p10_trueT = np.squeeze(p10_trueT)
    
    p90_predT = np.squeeze(p90_predT)
    p90_trueT = np.squeeze(p90_trueT)
    
    p95_predT = np.squeeze(p95_predT)
    p95_trueT = np.squeeze(p95_trueT) 


#    d = {'files': df.files.values[test_idx], 'P5': p5_pred, 'P10': p10_pred, 'P16': p16_pred, 'P25': p25_pred, 'P50': p50_pred, 'P75': p75_pred,'P84': p84_pred, 'P90': p90_pred, 'P95': p95_pred }
#    df2 = pd.DataFrame(d)
#    df2.to_csv('data_set9_new2.csv')  
# 
                
    gc.collect()
    rms16 = np.sqrt(np.nanmean((p16_pred- p16_true)**2))		
    rms25 = np.sqrt(np.nanmean((p25_pred- p25_true)**2))
    rms50 = np.sqrt(np.nanmean((p50_pred- p50_true)**2))
    rms75 = np.sqrt(np.nanmean((p75_pred- p75_true)**2))
    rms84 = np.sqrt(np.nanmean((p84_pred- p84_true)**2))
    rms5 = np.sqrt(np.nanmean((p5_pred- p5_true)**2))
    rms10 = np.sqrt(np.nanmean((p10_pred- p10_true)**2))    
    rms90 = np.sqrt(np.nanmean((p90_pred- p90_true)**2))
    rms95 = np.sqrt(np.nanmean((p95_pred- p95_true)**2))    
            
    print([rms5, rms10, rms16, rms25, rms50, rms75, rms84, rms90, rms95])
    

#    ind = np.where(100*(np.squeeze(p5_pred) / np.squeeze(p5_true))>200)[0]
#    test_files = next(file_generator(df, test_idx, batch_size=len(test_idx)))
#    print(sorted(test_files[ind]))

    
    fig = plt.figure(figsize=(12,12))
    
    plt.subplot(331)
    plt.plot(p5_trueT, p5_predT, 'ko', markersize=3)#, alpha=0.5)
    plt.plot(p5_true, p5_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 1000], [5, 1000], 'k', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(p5_pred - p5_true) / p5_true)))[:5]+' %', fontsize=8, color='b')    
    plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(p5_predT - p5_trueT) / p5_trueT)))[:5]+' %', fontsize=8)
    plt.xlim(10,1300); plt.ylim(10,1300)
    plt.title(r'A) P$_{5}$', fontsize=8, loc='left')
    
    plt.subplot(332)
    plt.plot(p10_trueT, p10_predT, 'ko', markersize=3)#, alpha=0.5)
    plt.plot(p10_true, p10_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 1000], [5, 1000], 'k', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(p10_pred - p10_true) / p10_true)))[:5]+' %', fontsize=8, color='b')    
    plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(p10_predT - p10_trueT) / p10_trueT)))[:5]+' %', fontsize=8)
    plt.xlim(10,1300); plt.ylim(10,1300)
    plt.title(r'B) P$_{10}$', fontsize=8, loc='left')
                
    plt.subplot(333)
    plt.plot(p16_trueT, p16_predT, 'ko', markersize=3)#, alpha=0.5)
    plt.plot(p16_true, p16_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 1000], [5, 1000], 'k', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(p16_pred - p16_true) / p16_true)))[:5]+' %', fontsize=8, color='b')    
    plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(p16_predT - p16_trueT) / p16_trueT)))[:5]+' %', fontsize=8)
    plt.xlim(10,1300); plt.ylim(10,1300)
    plt.title(r'C) P$_{15}$', fontsize=8, loc='left')
        
    plt.subplot(334)
    plt.plot(p25_trueT, p25_predT, 'ko', markersize=3)#, alpha=0.5)
    plt.plot(p25_true, p25_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 1000], [5, 1000], 'k', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(p25_pred - p25_true) / p25_true)))[:5]+' %', fontsize=8, color='b')    
    plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(p25_predT - p25_trueT) / p25_trueT)))[:5]+' %', fontsize=8)
    plt.xlim(10,1300); plt.ylim(10,1300)
    plt.title(r'D) P$_{25}$', fontsize=8, loc='left')
            
    plt.subplot(335)
    plt.plot(p50_trueT, p50_predT, 'ko', markersize=3)#, alpha=0.5)
    plt.plot(p50_true, p50_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 1000], [5, 1000], 'k', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(p50_pred - p50_true) / p50_true)))[:5]+' %', fontsize=8, color='b')    
    plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(p50_predT - p50_trueT) / p50_trueT)))[:5]+' %', fontsize=8)
    plt.xlim(10,1300); plt.ylim(10,1300)
    plt.title(r'E) P$_{50}$', fontsize=8, loc='left')
            
    plt.subplot(336)
    plt.plot(p75_trueT, p75_predT, 'ko', markersize=3)#, alpha=0.5)
    plt.plot(p75_true, p75_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 1000], [5, 1000], 'k', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(p75_pred - p75_true) / p75_true)))[:5]+' %', fontsize=8, color='b')    
    plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(p75_predT - p75_trueT) / p75_trueT)))[:5]+' %', fontsize=8)
    plt.xlim(10,1300); plt.ylim(10,1300)
    plt.title(r'F) P$_{75}$', fontsize=8, loc='left')
            
    plt.subplot(337)
    plt.plot(p84_trueT, p84_predT, 'ko', markersize=3)#, alpha=0.5)
    plt.plot(p84_true, p84_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 1000], [5, 1000], 'k', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(p84_pred - p84_true) / p84_true)))[:5]+' %', fontsize=8, color='b')    
    plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(p84_predT - p84_trueT) / p84_trueT)))[:5]+' %', fontsize=8)
    plt.xlim(10,1300); plt.ylim(10,1300)
    plt.title(r'G) P$_{84}$', fontsize=8, loc='left')
    plt.xlabel('Observed grain size (pixels)', fontsize=8)
    plt.ylabel('Estimated grain size (pixels)', fontsize=8)
            
    plt.subplot(338)
    plt.plot(p90_trueT, p90_predT, 'ko', markersize=3)#, alpha=0.5)
    plt.plot(p90_true, p90_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 1000], [5, 1000], 'k', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(p90_pred - p90_true) / p90_true)))[:5]+' %', fontsize=8, color='b')    
    plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(p90_predT - p90_trueT) / p90_trueT)))[:5]+' %', fontsize=8)
    plt.xlim(10,1300); plt.ylim(10,1300)
    plt.title(r'H) P$_{90}$', fontsize=8, loc='left')
        
    plt.subplot(339)
    plt.plot(p95_trueT, p95_predT, 'ko', markersize=3)#, alpha=0.5)
    plt.plot(p95_true, p95_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 1000], [5, 1000], 'k', lw=2)
    plt.xscale('log'); plt.yscale('log')
    plt.text(11,700,'Test : '+str(np.mean(100*(np.abs(p95_pred - p95_true) / p95_true)))[:5]+' %', fontsize=8, color='b')    
    plt.text(11,1000,'Train : '+str(np.mean(100*(np.abs(p95_predT - p95_trueT) / p95_trueT)))[:5]+' %', fontsize=8)
    plt.xlim(10,1300); plt.ylim(10,1300)
    plt.title(r'I) P$_{95}$', fontsize=8, loc='left')
                    
    #plt.show()
    plt.savefig('set9_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_xy-base'+str(base)+'_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    del fig   
    
    ####=============================
    fig = plt.figure(figsize=(12,12))
    
    plt.subplot(331)
    plt.plot(p5_trueT, p5_predT, 'k.', markersize=5)#, alpha=0.5)
    plt.plot(p5_true, p5_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 750], [5, 750], 'k', lw=2)
    ##plt.xscale('log'); plt.yscale('log')
    plt.title(str(np.mean(100*(np.abs(p5_pred - p5_true) / p5_true)))[:5]+' %', fontsize=10)
    plt.xlim(3,1100); plt.ylim(3,1100)
    
    plt.subplot(332)
    plt.plot(p10_trueT, p10_predT, 'k.', markersize=5)#, alpha=0.5)
    plt.plot(p10_true, p10_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 750], [5, 750], 'k', lw=2)
    ##plt.xscale('log'); plt.yscale('log')
    plt.title(str(np.mean(100*(np.abs(p10_pred - p10_true) / p10_true)))[:5]+' %', fontsize=10)
    plt.xlim(3,1100); plt.ylim(3,1100)
            
    plt.subplot(333)
    plt.plot(p16_trueT, p16_predT, 'k.', markersize=5)#, alpha=0.5)
    plt.plot(p16_true, p16_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 750], [5, 750], 'k', lw=2)
    #plt.xscale('log'); plt.yscale('log')
    plt.title(str(np.mean(100*(np.abs(p16_pred - p16_true) / p16_true)))[:5]+' %', fontsize=10)
    plt.xlim(3,1100); plt.ylim(3,1100)
    
    plt.subplot(334)
    plt.plot(p25_trueT, p25_predT, 'k.', markersize=5)#, alpha=0.5)
    plt.plot(p25_true, p25_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 750], [5, 750], 'k', lw=2)
    ##plt.xscale('log'); plt.yscale('log')
    plt.title(str(np.mean(100*(np.abs(p25_pred - p25_true) / p25_true)))[:5]+' %', fontsize=10)
    plt.xlim(3,1100); plt.ylim(3,1100)
        
    plt.subplot(335)
    plt.plot(p50_trueT, p50_predT, 'k.', markersize=5)#, alpha=0.5)
    plt.plot(p50_true, p50_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 750], [5, 750], 'k', lw=2)
    ##plt.xscale('log'); plt.yscale('log')
    plt.title(str(np.mean(100*(np.abs(p50_pred - p50_true) / p50_true)))[:5]+' %', fontsize=10)
    plt.xlim(3,1100); plt.ylim(3,1100)
        
    plt.subplot(336)
    plt.plot(p75_trueT, p75_predT, 'k.', markersize=5)#, alpha=0.5)
    plt.plot(p75_true, p75_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 750], [5, 750], 'k', lw=2)
    ##plt.xscale('log'); plt.yscale('log')
    plt.title(str(np.mean(100*(np.abs(p75_pred - p75_true) / p75_true)))[:5]+' %', fontsize=10)
    plt.xlim(3,1100); plt.ylim(3,1100)
        
    plt.subplot(337)
    plt.plot(p84_trueT, p84_predT, 'k.', markersize=5)#, alpha=0.5)
    plt.plot(p84_true, p84_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 750], [5, 750], 'k', lw=2)
    ##plt.xscale('log'); plt.yscale('log')
    plt.title(str(np.mean(100*(np.abs(p84_pred - p84_true) / p84_true)))[:5]+' %', fontsize=10)
    plt.xlim(3,1100); plt.ylim(3,1100)

    plt.subplot(338)
    plt.plot(p90_trueT, p90_predT, 'k.', markersize=5)#, alpha=0.5)
    plt.plot(p90_true, p90_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 750], [5, 750], 'k', lw=2)
    ##plt.xscale('log'); plt.yscale('log')
    plt.title(str(np.mean(100*(np.abs(p90_pred - p90_true) / p90_true)))[:5]+' %', fontsize=10)
    plt.xlim(3,1100); plt.ylim(3,1100)
    
    plt.subplot(339)
    plt.plot(p95_trueT, p95_predT, 'k.', markersize=5)#, alpha=0.5)
    plt.plot(p95_true, p95_pred, 'bx', markersize=5)#, alpha=0.5)
    plt.plot([5, 750], [5, 750], 'k', lw=2)
    ##plt.xscale('log'); plt.yscale('log')
    plt.title(str(np.mean(100*(np.abs(p95_pred - p95_true) / p95_true)))[:5]+' %', fontsize=10)
    plt.xlim(3,1100); plt.ylim(3,1100)
                
    #plt.show()
    plt.savefig('set9_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_xy-base'+str(base)+'_lin.png', dpi=300, bbox_inches='tight')
    plt.close()
    del fig 

    K.clear_session()  




#    f5= (p5_true/p5_pred) + np.abs(np.random.randn(len(p10_pred))/15)
#    f10= (p10_true/p10_pred)+ np.abs(np.random.randn(len(p10_pred))/15)
#    f16= (p16_true/p16_pred)+ np.abs(np.random.randn(len(p10_pred))/15)
#    f25= (p25_true/p25_pred)+ np.abs(np.random.randn(len(p10_pred))/15)
#    f50= (p50_true/p50_pred)+ np.abs(np.random.randn(len(p10_pred))/15)
#    f75= (p75_true/p75_pred)+ np.abs(np.random.randn(len(p10_pred))/15)
#    f84= (p84_true/p84_pred)+ np.abs(np.random.randn(len(p10_pred))/15)
#    f90= (p90_true/p90_pred)+ np.abs(np.random.randn(len(p10_pred))/15)
#    f95= (p95_true/p95_pred)+ np.abs(np.random.randn(len(p10_pred))/15)
#                           
#    d = {'files': df.files.values[test_idx], 'P5': f5*p5_pred, 'P10': f10*p10_pred, 'P16': f16*p16_pred, 'P25': f25*p25_pred, 'P50': f50*p50_pred, 'P75': f75*p75_pred,'P84': f84*p84_pred, 'P90': f90*p90_pred, 'P95': f95*p95_pred }
#    df2 = pd.DataFrame(d)
#    df2.to_csv('data_set9_new.csv')  
# 
#    f5= (p5_trueT/p5_predT) + np.abs(np.random.randn(len(p10_predT))/15)
#    f10= (p10_trueT/p10_predT)+ np.abs(np.random.randn(len(p10_predT))/15)
#    f16= (p16_trueT/p16_predT)+ np.abs(np.random.randn(len(p10_predT))/15)
#    f25= (p25_trueT/p25_predT)+ np.abs(np.random.randn(len(p10_predT))/15)
#    f50= (p50_trueT/p50_predT)+ np.abs(np.random.randn(len(p10_predT))/15)
#    f75= (p75_trueT/p75_predT)+ np.abs(np.random.randn(len(p10_predT))/15)
#    f84= (p84_trueT/p84_predT)+ np.abs(np.random.randn(len(p10_predT))/15)
#    f90= (p90_trueT/p90_predT)+ np.abs(np.random.randn(len(p10_predT))/15)
#    f95= (p95_trueT/p95_predT)+ np.abs(np.random.randn(len(p10_predT))/15)
#    










# ______     ______     _____     __     __   __     ______     ______  
#/\  ___\   /\  ___\   /\  __-.  /\ \   /\ "-.\ \   /\  ___\   /\__  _\ 
#\ \___  \  \ \  __\   \ \ \/\ \ \ \ \  \ \ \-.  \  \ \  __\   \/_/\ \/ 
# \/\_____\  \ \_____\  \ \____-  \ \_\  \ \_\\"\_\  \ \_____\    \ \_\ 
#  \/_____/   \/_____/   \/____/   \/_/   \/_/ \/_/   \/_____/     \/_/ 
#
# By Dr Daniel Buscombe,
# daniel.buscombe@nau.edu

###===================================================
# import libraries
from sedinet_utils import *

###===================================================
def run_continuous_training(vars, csvfile, base, name, res_folder, dropout, add_bn):
   """
   This function generates, trains and evaluates a SediNet model for continuous prediction
   """

   # ensemble models tend to be more stable so below trains 3 models using different random
   # batches of images and slightly different base values
   models = []
   for base in [base-2,base,base+2]:
      ##======================================
      ## this randomly selects imagery for training and testing imagery sets
      ## while also making sure that both training and tetsing sets have at least 3 examples of each category
      train_idx, test_idx, df = random_sample(csvfile)  
   
      ##==============================================
      ## create a SediNet model to estimate sediment category
      model = make_cont_sedinet(base, vars, add_bn, dropout)

      ##==============================================
      ## train model
      model, weights_path = train_sedinet_cont(model, df, train_idx, test_idx, base, name, vars)
      models.append(model)
      
   # test model
   predict_test_train_cont(df, train_idx, test_idx, vars, models, weights_path, name, base)

   K.clear_session()   
      
   ##==============================================    
   ## move model files and plots to the results folder    
   tidy(res_folder, name, base)

###===================================================
def run_categorical_training(N, var, csvfile, base, ID_MAP, res_folder, dropout):
   """
   This function generates, trains and evaluates a SediNet model for categorical prediction
   """

   # ensemble models tend to be more stable so below trains 3 models using different random
   # batches of images and slightly different base values
   models = []
   for base in [base-2,base,base+2]:
      ##======================================
      ## this randomly selects imagery for training and testing imagery sets
      ## while also making sure that both training and tetsing sets have at least 3 examples of each category
      train_idx, test_idx, df = stratify_random_sample(N, var, csvfile)
 
      ##==============================================
      ## create a SediNet model to estimate sediment category
      model = make_cat_sedinet(base, ID_MAP, dropout)

      ##==============================================
      ## train model
      model, weights_path = train_sedinet_cat(model, df, train_idx, test_idx, ID_MAP, base, var)
      models.append(model)

   classes = np.arange(len(ID_MAP))

   # test model
   # predicts using all 3 models and uses the mode as the prediction
   predict_test_train_cat(df, train_idx, test_idx, var, models, classes, weights_path)

   K.clear_session()   
      
   ##==============================================    
   ## move model files and plots to the results folder    
   tidy(res_folder, '', '')

###===================================================
def conv_block(inp, filters=32, bn=True, pool=True, drop=True):
   """
   This function generates a SediNet convolutional block
   """
   _ = Conv2D(filters=filters, kernel_size=3, activation='relu', kernel_initializer='he_uniform')(inp)
   if bn:
       _ = BatchNormalization()(_)
   if pool:
       _ = MaxPool2D()(_)
   if drop:        
       _ = Dropout(0.2)(_)        
   return _


###===================================================
def train_sedinet_cont(model, df, train_idx, test_idx, base, name, vars):
    """
    This function trains an implementation of SediNet
    """
    ##==============================================
    ## create training and testing file generators, set the weights path, plot the model, and create a callback list for model training
    
    if len(vars)==1:        
       train_gen = get_data_generator_1vars(df, train_idx, True, vars, batch_size)
       valid_gen = get_data_generator_1vars(df, test_idx, True, vars, valid_batch_size)    
    elif len(vars)==2:        
       train_gen = get_data_generator_2vars(df, train_idx, True, vars, batch_size)
       valid_gen = get_data_generator_2vars(df, test_idx, True, vars, valid_batch_size)    
    elif len(vars)==3:        
       train_gen = get_data_generator_3vars(df, train_idx, True, vars, batch_size)
       valid_gen = get_data_generator_3vars(df, test_idx, True, vars, valid_batch_size)
    elif len(vars)==4:        
       train_gen = get_data_generator_4vars(df, train_idx, True, vars, batch_size)
       valid_gen = get_data_generator_4vars(df, test_idx, True, vars, valid_batch_size)
    elif len(vars)==5:        
       train_gen = get_data_generator_5vars(df, train_idx, True, vars, batch_size)
       valid_gen = get_data_generator_5vars(df, test_idx, True, vars, valid_batch_size)
    elif len(vars)==6:        
       train_gen = get_data_generator_6vars(df, train_idx, True, vars, batch_size)
       valid_gen = get_data_generator_6vars(df, test_idx, True, vars, valid_batch_size)
    elif len(vars)==7:        
       train_gen = get_data_generator_7vars(df, train_idx, True, vars, batch_size)
       valid_gen = get_data_generator_7vars(df, test_idx, True, vars, valid_batch_size)
    elif len(vars)==8:        
       train_gen = get_data_generator_8vars(df, train_idx, True, vars, batch_size)
       valid_gen = get_data_generator_8vars(df, test_idx, True, vars, valid_batch_size)       
    elif len(vars)==9:        
       train_gen = get_data_generator_9vars(df, train_idx, True, vars, batch_size)
       valid_gen = get_data_generator_9vars(df, test_idx, True, vars, valid_batch_size)
                 
    weights_path = name+"_base"+str(base)+"_model_checkpoint.hdf5"
    
    plot_model(model, weights_path.replace('.hdf5', '_model.png'), show_shapes=True, show_layer_names=True)
    
    callbacks_list = [
         ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
     ]

    print("==========================================")
    print("[INFORMATION] schematic of the model has been written out to: "+weights_path.replace('.hdf5', '_model.png'))
    print("[INFORMATION] weights will be written out to: "+weights_path)
    
    ##==============================================
    ## set checkpoint file and parameters that control early stopping, 
    ## and reduction of learning rate if and when validation scores plateau upon successive epochs
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
    if len(vars)==9:
       plot_train_history_9var(history, vars)
       plt.savefig(name+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
       plt.close('all')       
    elif len(vars)==8:
       plot_train_history_8var(history, vars)     
       plt.savefig(name+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
       plt.close('all')
    elif len(vars)==7:
       plot_train_history_7var(history, vars)     
       plt.savefig(name+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
       plt.close('all')
    elif len(vars)==6:
       plot_train_history_6var(history, vars)     
       plt.savefig(name+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
       plt.close('all')
    elif len(vars)==5:
       plot_train_history_5var(history, vars)     
       plt.savefig(name+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
       plt.close('all')
    elif len(vars)==4:
       plot_train_history_4var(history, vars)     
       plt.savefig(name+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
       plt.close('all')       
    elif len(vars)==3:
       plot_train_history_3var(history, vars)     
       plt.savefig(name+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
       plt.close('all')
    elif len(vars)==2:
       plot_train_history_2var(history, vars)     
       plt.savefig(name+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
       plt.close('all')
    elif len(vars)==1:
       plot_train_history_1var_mae(history)     
       plt.savefig(name+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
       plt.close('all')
                                          
    # serialize model to JSON to use later to predict							  
    model_json = model.to_json()
    with open(weights_path.replace('.hdf5','.json'), "w") as json_file:
       json_file.write(model_json)

    ## do some garbage collection    
    gc.collect()

    return model, weights_path
    

###===================================================
def train_sedinet_cat(model, df, train_idx, test_idx, ID_MAP, base, var):
    """
    This function trains an implementation of SediNet
    """
    ##==============================================
    ## create training and testing file generators, set the weights path, plot the model, and create a callback list for model training
    train_gen = get_data_generator_1image(df, train_idx, True, ID_MAP, var, batch_size)
    valid_gen = get_data_generator_1image(df, test_idx, True, ID_MAP, var, valid_batch_size)
   
    weights_path = var+"_base"+str(base)+"_model_checkpoint.hdf5"
    
    plot_model(model, weights_path.replace('.hdf5', '_model.png'), show_shapes=True, show_layer_names=True)
    
    callbacks_list = [
         ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
     ]

    print("==========================================")
    print("[INFORMATION] schematic of the model has been written out to: "+weights_path.replace('.hdf5', '_model.png'))
    print("[INFORMATION] weights will be written out to: "+weights_path)
    
    ##==============================================
    ## set checkpoint file and parameters that control early stopping, 
    ## and reduction of learning rate if and when validation scores plateau upon successive epochs
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
    plt.savefig(var+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history-base'+str(base)+'.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    # serialize model to JSON to use later to predict							  
    model_json = model.to_json()
    with open(weights_path.replace('.hdf5','.json'), "w") as json_file:
       json_file.write(model_json)

    ## do some garbage collection    
    gc.collect()

    return model, weights_path

###===================================================
def make_cat_sedinet(base, ID_MAP, dropout):
    """
    This function creates an implementation of SediNet for estimating sediment category
    """
    input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
    _ = conv_block(input_layer, filters=base, bn=False, pool=False, drop=False)
    _ = conv_block(_, filters=base*2, bn=False, pool=True,drop=False)    
    _ = conv_block(_, filters=base*3, bn=False, pool=True,drop=False) 
    _ = conv_block(_, filters=base*4, bn=False, pool=True,drop=False)

    bottleneck = GlobalMaxPool2D()(_)
    bottleneck = Dropout(dropout)(bottleneck) 

     # for class prediction
    _ = Dense(units=128, activation='relu')(bottleneck)
    output = Dense(units=len(ID_MAP), activation='softmax', name='output')(_) 
                     
    model = Model(inputs=input_layer, outputs=[output]) 
    model.compile(optimizer='adam',
                  loss={'output': 'categorical_crossentropy'}, 
                  loss_weights={'output': 1.}, 
                  metrics={'output': 'accuracy'})                   
    print("==========================================")
    print('[INFORMATION] Model summary:')                                  
    model.summary()
    return model
    

###===================================================
def make_cont_sedinet(base, vars, add_bn, dropout):
    """
    This function creates an implementation of SediNet for estimating sediment metric on a continuous scale
    """
    input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH*2, 1))
    _ = conv_block(input_layer, filters=base, bn=False, pool=False, drop=False)
    _ = conv_block(_, filters=base*2, bn=False, pool=True,drop=False)    
    _ = conv_block(_, filters=base*3, bn=False, pool=True,drop=False) 
    _ = conv_block(_, filters=base*4, bn=False, pool=True,drop=False)

    if add_bn == True:
       _ = BatchNormalization(axis=-1)(_) 
    bottleneck = GlobalMaxPool2D()(_)
    bottleneck = Dropout(dropout)(bottleneck) 

    units = 512
    _ = Dense(units=units, activation='relu')(bottleneck)
        
    outputs = []
    for var in vars:
       outputs.append(Dense(units=1, activation='linear', name=var+'_output')(_) )

    loss = dict(zip([k+"_output" for k in vars], ['mse' for k in vars]))
    metrics = dict(zip([k+"_output" for k in vars], ['mae' for k in vars]))
        
    model = Model(inputs=input_layer, outputs=outputs) 
    model.compile(optimizer='adam',loss=loss, metrics=metrics)               
    print("==========================================")
    print('[INFORMATION] Model summary:')                                  
    model.summary()
    return model
    
    
            

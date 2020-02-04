
## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.2 (Feb 4 2020)

from sedinet_models import *

###===================================================
def run_training_siso_simo(vars, train_csvfile, test_csvfile, name, res_folder, mode, greyscale, dropout, numclass):
   """
   This function generates, trains and evaluates a sedinet model for continuous prediction
   """    

   if numclass>0:
      ID_MAP = dict(zip(np.arange(numclass), [str(k) for k in range(numclass)]))

   ##======================================
   ## this randomly selects imagery for training and testing imagery sets
   ## while also making sure that both training and tetsing sets have at least 3 examples of each category
   train_idx, train_df = get_df(train_csvfile)
   test_idx, test_df = get_df(test_csvfile)   

   ##==============================================
   ## create a sedinet model to estimate category
   if numclass>0:
      SM = make_cat_sedinet(ID_MAP, dropout)
   else:
      SM = make_sedinet_siso_simo(vars, greyscale, dropout) 

      CS = []
      for var in vars:
         cs = RobustScaler() #MinMaxScaler()
         cs.fit_transform(np.r_[train_df[var].values, test_df[var].values].reshape(-1,1))
         CS.append(cs)
         del cs
       
   ##==============================================
   ## train model
   if numclass==0:
      SM, weights_path = train_sedinet_siso_simo(SM, train_df, test_df, 
                                                  train_idx, test_idx, name, vars, mode, greyscale, CS, dropout) 
   else:
      SM, weights_path = train_sedinet_cat(SM, train_df, test_df, train_idx, test_idx, ID_MAP, vars, greyscale)
      classes = np.arange(len(ID_MAP))


   # test model
   if numclass==0:
      predict_test_train_siso_simo(train_df, test_df, train_idx, test_idx, vars, 
                                SM, weights_path, name, mode, greyscale, CS, dropout)

   else:
      predict_test_train_cat(train_df, test_df, train_idx, test_idx, vars[0], SM, classes, weights_path, greyscale)

   K.clear_session()

   ##==============================================
   ## move model files and plots to the results folder
   if numclass==0:
      tidy(res_folder, name) 
   else:
      tidy(res_folder, '')


###===================================================
def train_sedinet_cat(SM, train_df, test_df, train_idx, test_idx, ID_MAP, vars, greyscale):
    """
    This function trains an implementation of SediNet
    """
    ##==============================================
    ## create training and testing file generators, set the weights path, plot the model, and create a callback list for model training
    train_gen = get_data_generator_1image(train_df, train_idx, True, ID_MAP, vars[0], batch_size, greyscale)
    valid_gen = get_data_generator_1image(test_df, test_idx, True, ID_MAP, vars[0], valid_batch_size, greyscale)

    weights_path = vars[0]+"_model_checkpoint.hdf5"

    try:
       plot_model(SM, weights_path.replace('.hdf5', '_model.png'), show_shapes=True, show_layer_names=True)
    except:
       pass

    callbacks_list = [
         ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
     ]

    print("==========================================")
    print("[INFORMATION] schematic of the model has been written out to: "+weights_path.replace('.hdf5', '_model.png'))
    print("[INFORMATION] weights will be written out to: "+weights_path)

    ##==============================================
    ## set checkpoint file and parameters that control early stopping,
    ## and reduction of learning rate if and when validation scores plateau upon successive epochs
    reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=5, verbose=1, mode='auto', min_delta=min_delta, cooldown=5, min_lr=min_lr)

    earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=15)

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)

    callbacks_list = [model_checkpoint, reduceloss_plat, earlystop]

    ##==============================================
    ## train the model
    history = SM.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=num_epochs,
                    callbacks=callbacks_list,
                    validation_data=valid_gen,
                    validation_steps=len(test_idx)//valid_batch_size,
                    max_queue_size=10)

    ###===================================================
    ## Plot the loss and accuracy as a function of epoch
    plot_train_history_1var(history)
    plt.savefig(vars[0]+'_'+str(IM_HEIGHT)+'_batch'+str(batch_size)+'_history.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    # serialize model to JSON to use later to predict
    model_json = SM.to_json()
    with open(weights_path.replace('.hdf5','.json'), "w") as json_file:
       json_file.write(model_json)

    ## do some garbage collection
    gc.collect()

    return SM, weights_path

###===================================================
def train_sedinet_siso_simo(SM, train_df, test_df, train_idx, test_idx, name, vars, mode, greyscale, CS, dropout): 
    """
    This function trains an implementation of sedinet
    """

    ##==============================================
    ## create training and testing file generators, set the weights path, 
    ## plot the model, and create a callback list for model training

    train_gen = get_data_generator_Nvars_siso_simo(train_df, train_idx, True, 
                                                   vars, batch_size, greyscale, CS)
    valid_gen = get_data_generator_Nvars_siso_simo(test_df, test_idx, True, 
                                                   vars, valid_batch_size, greyscale, CS)
    
    varstring = ''.join([str(k)+'_' for k in vars])
    weights_path = name+"_"+mode+"_batch"+str(batch_size)+"_"+varstring+"_checkpoint.hdf5" 

    joblib.dump(CS, weights_path.replace('.hdf5','_scaler.pkl')) 

    
    try:
        plot_model(SM, weights_path.replace('.hdf5', '_model.png'), show_shapes=True, show_layer_names=True)
        print("[INFORMATION] model schematic written to: "+weights_path.replace('.hdf5', '_model.png'))
    except:
        pass

    print("==========================================")
    print("[INFORMATION] weights will be written out to: "+weights_path)

    ##==============================================
    ## set checkpoint file and parameters that control early stopping,
    ## and reduction of learning rate if and when validation scores plateau upon successive epochs
    reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, 
                                        patience=5, verbose=1, mode='auto', 
                                        min_delta=min_delta, cooldown=5, min_lr=min_lr)

    earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=stop_patience)

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, 
                                       save_best_only=True, mode='min', save_weights_only = True)

    callbacks_list = [model_checkpoint, reduceloss_plat, earlystop]


    try:
        with open(weights_path.replace('.hdf5','') + '_report.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            SM.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.close()       
        print("[INFORMATION] model summary written to: "+ weights_path.replace('.hdf5','') + '_report.txt')      
        with open(weights_path.replace('.hdf5','') + '_report.txt','r') as fh:
            tmp = fh.readlines()
        print("===============================================")   
        print("Total parameters: %s" % (''.join(tmp).split('Total params:')[-1].split('\n')[0]))
        fh.close()       
        print("===============================================")   
    except:
        pass     

    ##==============================================
    ## train the model
    history = SM.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=num_epochs,
                    callbacks=callbacks_list,
                    validation_data=valid_gen,
                    validation_steps=len(test_idx)//valid_batch_size,
                    use_multiprocessing=True)

    ###===================================================
    ## Plot the loss and accuracy as a function of epoch
    if len(vars)==1:
       plot_train_history_1var_mae(history)
    else:
       plot_train_history_Nvar(history, vars, len(vars))
       
    varstring = ''.join([str(k)+'_' for k in vars])
    plt.savefig(weights_path.replace('.hdf5', '_history.png'), dpi=300, bbox_inches='tight')
    plt.close('all')

    # serialize model to JSON to use later to predict
    model_json = SM.to_json()
    with open(weights_path.replace('.hdf5','.json'), "w") as json_file:
       json_file.write(model_json)

    ## do some garbage collection
    gc.collect()

    return SM, weights_path

      
###===================================================
def run_training_miso_mimo(vars, train_csvfile, test_csvfile, name, res_folder, mode, greyscale, auxin, dropout):
   """
   This function generates, trains and evaluates a sedinet model for continuous prediction
   """    
   ##======================================
   ## this randomly selects imagery for training and testing imagery sets
   ## while also making sure that both training and tetsing sets have at least 3 examples of each category
   train_idx, train_df = get_df(train_csvfile)
   test_idx, test_df = get_df(test_csvfile)   

   ##==============================================
   ## create a sedinet model to estimate category
   cnn = make_sedinet_miso_mimo(False, dropout)

   CS = []
   for var in vars:
      cs = RobustScaler() #MinMaxScaler()
      cs.fit_transform(np.r_[train_df[var].values, test_df[var].values].reshape(-1,1))
      CS.append(cs)
      del cs

   CSaux = []
   cs = RobustScaler() #MinMaxScaler()
   cs.fit_transform(np.r_[train_df[auxin].values, test_df[auxin].values].reshape(-1,1))
   CSaux.append(cs)
   del cs
       
   ##==============================================
   ## train model
   SM, weights_path = train_sedinet_miso_mimo(cnn, train_df, test_df, 
                                                  train_idx, test_idx, name, vars, 
                                                  auxin, mode, greyscale, CS, CSaux) 

   # test model
   predict_test_train_miso_mimo(train_df, test_df, train_idx, test_idx, vars, 
                                auxin, SM, weights_path, name, mode, greyscale, CS, CSaux)
                                
   K.clear_session()

   ##==============================================
   ## move model files and plots to the results folder
   tidy(res_folder, name) 

###===================================================
def train_sedinet_miso_mimo(cnn, train_df, test_df, train_idx, test_idx, name, vars, auxin, mode, greyscale, CS, CSaux): #dense_neurons
    """
    This function trains an implementation of sedinet
    """
    
    dense_neurons = 4

    ##==============================================
    ## create training and testing file generators, set the weights path, plot the model, and create a callback list for model training
    varstring = ''.join([str(k)+'_' for k in vars])
    weights_path = name+"_"+auxin+"_"+mode+"_batch"+str(batch_size)+"_"+varstring+"_checkpoint.hdf5" 
 
    # Create the MLP and CNN models
    mlp = make_mlp(1) #dense_neurons
 
    # Create the input to the final set of layers as the output of both the MLP and CNN
    combinedInput = concatenate([mlp.output, cnn.output])

    # The final fully-connected layer head will have two dense layers (one relu and one sigmoid)
    x = Dense(dense_neurons, activation="relu")(combinedInput)
    x = Dense(1, activation="sigmoid")(x)
    
    ## The final model accepts numerical data on the MLP input and 
    ## images on the CNN input, outputting a single value  
    outputs = []
    for var in vars:
       outputs.append(Dense(units=1, activation='linear', name=var+'_output')(x) )

    loss = dict(zip([k+"_output" for k in vars], ['mse' for k in vars]))
    metrics = dict(zip([k+"_output" for k in vars], ['mae' for k in vars]))

    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input
    SM = Model(inputs=[mlp.input, cnn.input], outputs=outputs)

    SM.compile(optimizer=opt, loss=loss, metrics=metrics)

    try:
        plot_model(SM, weights_path.replace('.hdf5', '_model.png'), show_shapes=True, show_layer_names=True)
        print("[INFORMATION] model schematic written to: "+weights_path.replace('.hdf5', '_model.png'))
    except:
        pass

    print("==========================================")
    print("[INFORMATION] weights will be written out to: "+weights_path)


    try:
        with open(weights_path.replace('.hdf5','') + '_report.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            SM.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.close()       
        print("[INFORMATION] model summary written to: "+ weights_path.replace('.hdf5','') + '_report.txt')      
        with open(weights_path.replace('.hdf5','') + '_report.txt','r') as fh:
            tmp = fh.readlines()
        print("===============================================")   
        print("Total parameters: %s" % (''.join(tmp).split('Total params:')[-1].split('\n')[0]))
        fh.close()       
        print("===============================================")   
    except:
        pass     
 
    
    reduceloss_plat = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=5, 
                                        verbose=1, mode='auto', min_delta=min_delta, 
                                        cooldown=5, min_lr=min_lr)

    earlystop = EarlyStopping(monitor="val_loss", mode="auto", patience=stop_patience)

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, 
                                       save_best_only=True, mode='min', save_weights_only = True) 


    callbacks_list = [model_checkpoint, reduceloss_plat, earlystop]
    
    #aux_mean = train_df[auxin].mean()
    #aux_std =  train_df[auxin].std()
    
    train_gen = get_data_generator_Nvars_miso_mimo(train_df, train_idx, True, 
                                                   vars, auxin, batch_size, greyscale, CS, CSaux)
    valid_gen = get_data_generator_Nvars_miso_mimo(test_df, test_idx, True, 
                                                   vars, auxin, valid_batch_size, greyscale, CS, CSaux)

    ##==============================================
    ## train the model
    history = SM.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=num_epochs,
                    callbacks=callbacks_list,
                    validation_data=valid_gen,
                    validation_steps=len(test_idx)//valid_batch_size)
                    
    ###===================================================
    ## Plot the loss and accuracy as a function of epoch
    if len(vars)==1:
       plot_train_history_1var_mae(history)
    else:
       plot_train_history_Nvar(history, vars, len(vars))
       
    varstring = ''.join([str(k)+'_' for k in vars])
    plt.savefig(weights_path.replace('.hdf5', '_history.png'), dpi=300, bbox_inches='tight')
    plt.close('all')

    # serialize model to JSON to use later to predict
    model_json = SM.to_json()
    with open(weights_path.replace('.hdf5','.json'), "w") as json_file:
       json_file.write(model_json)

    ## do some garbage collection
    gc.collect()

    return SM, weights_path



### the following is a workflow to use keras ImageDataGenerator in-built function
### the disadvtange is not being able to use validation loss to stop training early and set learning rate
          
#    if greyscale==True:
#        train_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
#            train_df, x_col='files', shuffle=False, class_mode=None, color_mode='grayscale',
#            target_size=(IM_WIDTH, IM_HEIGHT), batch_size=len(train_df))    

#        valid_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
#            test_df, x_col='files', shuffle=False, class_mode=None, color_mode='grayscale',
#            target_size=(IM_WIDTH, IM_HEIGHT), batch_size=len(test_df))
#            
#    else:
#        train_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
#            train_df, x_col='files', shuffle=False, class_mode=None, color_mode='rgb',
#            target_size=(IM_WIDTH, IM_HEIGHT), batch_size=len(train_df))

#        valid_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
#            test_df, x_col='files', shuffle=False, class_mode=None, color_mode='rgb',
#            target_size=(IM_WIDTH, IM_HEIGHT), batch_size=len(test_df))
#              
#    trainImagesX = next(train_gen)
#    testImagesX = next(valid_gen)

#    # Setting the labels for y as the safe column
#    trainY = train_df[vars].values
#    testY = test_df[vars].values

#    # attribute variables used to train
#    trainAttrX = train_df[auxin].values
#    testAttrX = test_df[auxin].values

#    # Process the structured data
#    trainAttrX, testAttrX, cs = process_structured_data(trainAttrX, testAttrX)

#    ##==============================================
#    ## set checkpoint file and parameters that control early stopping,
#    ## and reduction of learning rate if and when validation scores plateau upon successive epochs
#    reduceloss_plat = ReduceLROnPlateau(monitor='loss', factor=factor, patience=5, verbose=1, mode='auto', min_delta=min_delta, cooldown=5, min_lr=min_lr)

#    earlystop = EarlyStopping(monitor="loss", mode="auto", patience=25) #val_loss

#    model_checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True) #val_loss

#    callbacks_list = [model_checkpoint, reduceloss_plat, earlystop]

#    ##==============================================
#    ## train the model    
#    history = SM.fit([trainAttrX, trainImagesX], trainY, 
#                      validation_data=([testAttrX, testImagesX], testY), 
#                      epochs=num_epochs,
#                      steps_per_epoch=len(train_idx)//batch_size,
#                      validation_steps=len(test_idx)//valid_batch_size,
#                      callbacks=callbacks_list, 
#                      batch_size=batch_size)

                    

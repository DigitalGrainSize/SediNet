
## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.2 (Feb 4 2020)

###===================================================
# import libraries
from sedinet_models import *

###===================================================
def get_data_generator_1vars(df, indices, for_training, vars, greyscale, batch_size=16):
    """
    This function generates data for a batch of images and 1 associated metric
    """
    images, p1s = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, p1 = r['files'], r[vars[0]]
            #im = Image.open(file).convert('LA')
            #im = im.resize((IM_HEIGHT, IM_HEIGHT))
            #im = np.array(im) / 255.0
            #im2 = np.rot90(im)

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

            #images.append(np.expand_dims(np.hstack((im[:,:,0], im2[:,:,0])),axis=2))
            
            if greyscale==True:            
               images.append(np.expand_dims(im[:,:,0], axis=2)) 
            else:
               images.append(im)            
            
            p1s.append(p1)
            if len(images) >= batch_size:
                yield np.array(images), [np.array(p1s)]
                images, p1s = [], []
        if not for_training:
            break

###===================================================
def estimate_categorical(vars, csvfile, res_folder, dropout, numclass, greyscale, weights_path):
   """
   This function uses a SediNet model for categorical prediction
   """
   
   ID_MAP = dict(zip(np.arange(numclass), [str(k) for k in range(numclass)]))
   
   ###===================================================
   ## read the data set in, clean and modify the pathnames so they are absolute
   df = pd.read_csv(csvfile)
   df['files'] = [k.strip() for k in df['files']]
   df['files'] = [os.getcwd()+os.sep+f.replace('\\',os.sep) for f in df['files']]

   train_idx = np.arange(len(df))

   train_gen = get_data_generator_1vars(df, train_idx, True, vars, greyscale, len(df))
   
   #models = []
   #for base in [base-2,base,base+2]:
   #weights_path = var+"_model_checkpoint.hdf5"
   ##==============================================
   ## create a SediNet model to estimate sediment category
   #model = make_cat_sedinet(base, ID_MAP, dropout)
   SM = make_cat_sedinet(ID_MAP, dropout)
   SM.load_weights(os.getcwd()+os.sep+weights_path)
   #models.append(model)

   x_train, (trueT)= next(train_gen)
   trueT = trueT[0] #np.squeeze(np.asarray(trueT).argmax(axis=-1) )

   #P = []; PT = []
   #for model in models:
   predT = SM.predict(x_train, batch_size=1)
   
   del x_train, train_gen
   
   predT = np.asarray(predT).argmax(axis=-1)
   #PT.append(predT)

   #predT = np.squeeze(mode(np.asarray(PT), axis=0)[0])

   ##==============================================
   try:
      ## print a classification report to screen, showing f1, precision, recall and accuracy
      print("==========================================")
      print("Classification report for "+vars[0])
      print(classification_report(trueT, predT))
   except:
      pass

   classes = np.arange(len(ID_MAP))
   ##==============================================
   ## create figures showing confusion matrices for data set
   plot_confmat(predT, trueT, vars[0]+'T',classes)
   plt.savefig(weights_path.replace('.hdf5','_cm_predict.png'), dpi=300, bbox_inches='tight')
   plt.close('all')



###===================================================
def estimate_siso_simo(vars, csvfile, greyscale, weights_path, dropout): 
   """
   This function uses a sedinet model for continuous prediction
   """
   ###===================================================
   ## read the data set in, clean and modify the pathnames so they are absolute
   df = pd.read_csv(csvfile)
   df['files'] = [k.strip() for k in df['files']]

   train_idx = np.arange(len(df))

   CS = joblib.load(weights_path.replace('.hdf5','_scaler.pkl')) 

   train_gen = get_data_generator_Nvars_siso_simo(df, train_idx, False, vars, len(df), greyscale, CS) #
   
   try:
      x_train, tmp = next(train_gen)

      if len(vars)>1:
         counter = 0
         for v in vars:
            exec(v+'_trueT = np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1)))')
            counter +=1
      else:
         exec(vars[0]+'_trueT = np.squeeze(CS[0].inverse_transform(tmp[0].reshape(-1,1)))')
   
   except:
      df['files'] = [os.getcwd()+os.sep+f.replace('\\',os.sep) for f in df['files']]
      train_gen = get_data_generator_Nvars_siso_simo(df, train_idx, False, vars, len(df), greyscale, CS) #   
      x_train, tmp = next(train_gen)
         
      if len(vars)>1:
         counter = 0
         for v in vars:
            exec(v+'_trueT = np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1)))')
            counter +=1
      else:
         exec(vars[0]+'_trueT = np.squeeze(CS[0].inverse_transform(tmp[0].reshape(-1,1)))')

   
   varstring = ''.join([str(k)+'_' for k in vars])
      
   ##==============================================
   ## create a sedinet model to estimate category
   model = make_sedinet_siso_simo(vars, greyscale, dropout) 
   model.load_weights(os.getcwd()+os.sep+weights_path)

   for v in vars:
      exec(v+'_PT = []')

   tmp = model.predict(x_train, batch_size=1)

   if len(vars)>1:
       counter = 0
       for v in vars:
          exec(v+'_PT.append(np.squeeze(CS[counter].inverse_transform(tmp[counter].reshape(-1,1))))')
          counter +=1
   else:
       exec(vars[0]+'_PT.append(np.asarray(np.squeeze(CS[0].inverse_transform(tmp.reshape(-1,1)))))') 


   if len(vars)>1:
      for k in range(len(vars)):
         exec(vars[k]+'_predT = np.squeeze(np.mean(np.asarray('+vars[k]+'_PT), axis=0))')
   else:
      exec(vars[0]+'_predT = np.squeeze(np.mean(np.asarray('+vars[0]+'_PT), axis=0))')


   Z = joblib.load(weights_path.replace('.hdf5','_bias.pkl'))


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
      nrows = 1; ncols = 3
   elif len(vars)==2:
      nrows = 2; ncols = 1
   elif len(vars)==1:
      nrows = 1; ncols = 1

   ## make a plot
   fig = plt.figure()
   labs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
   for k in range(1,1+(nrows*ncols)):
      try:
         plt.subplot(nrows,ncols,k)
         x = eval(vars[k-1]+'_trueT')
         y = eval(vars[k-1]+'_predT')

         y = np.polyval(Z[k-1],y) #apply bias correction
         y = np.abs(y) #make sure no negative values

         plt.plot(x, y, 'ko', markersize=5)

         plt.plot([ np.min(np.hstack((x,y))),  np.max(np.hstack((x,y)))], [ np.min(np.hstack((x,y))),  np.max(np.hstack((x,y)))], 'k', lw=2)

         plt.text(1.02*np.min(np.hstack((x,y))), 0.98*np.max(np.hstack((x,y))),'Test : '+str(np.mean(100*(np.abs(eval(vars[k-1]+'_predT') - eval(vars[k-1]+'_trueT')) / eval(vars[k-1]+'_trueT'))))[:5]+' %', fontsize=8)

         plt.title(r''+labs[k-1]+') '+vars[k-1], fontsize=8, loc='left')
         plt.xlabel('Actual '+vars[k-1])
         plt.ylabel('Estimated '+vars[k-1])
      except:
         pass
      
   varstring = ''.join([str(k)+'_' for k in vars])
   plt.savefig(weights_path.replace('.hdf5', '_predict.png'), dpi=300, bbox_inches='tight')
   plt.close()
   del fig


#   if len(vars)>1:
#      counter = 0
#      for v in vars:
#         exec(v+'_trueT = np.squeeze(tmp[counter])')
#         counter +=1
#   else:
#      exec(vars[0]+'_trueT = np.squeeze(tmp)')

#   if len(vars)>1:
#      counter = 0
#      for v in vars:
#         exec(v+'_PT.append(np.squeeze(tmp[counter]))')
#         counter +=1
#   else:
#      exec(vars[0]+'_PT.append(np.asarray(np.squeeze(tmp)))')


###===================================================
def estimate_miso_mimo(vars, test_csvfile, greyscale, auxin, weights_path): 
   """
   This function uses a sedinet model for continuous prediction
   """
   test_idx, test_df = get_df(test_csvfile)   

   if greyscale==True:
       valid_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
           test_df, x_col='files', shuffle=False, class_mode=None, color_mode='grayscale',
           target_size=(IM_WIDTH, IM_HEIGHT), batch_size=len(test_df))
            
   else:
       valid_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
           test_df, x_col='files', shuffle=False, class_mode=None, color_mode='rgb',
           target_size=(IM_WIDTH, IM_HEIGHT), batch_size=len(test_df))
              
   testImagesX = next(valid_gen)

   ##==============================================
   
   if len(vars)>1:
      counter = 0
      for v in vars:
         exec(v+'_trueT = np.squeeze(tmp[counter])')
         counter +=1
   else:
      exec(vars[0]+'_trueT = np.squeeze(tmp)')
   
   varstring = ''.join([str(k)+'_' for k in vars])
      
   ##==============================================
   ## create a sedinet model to estimate category
   model = make_sedinet_miso_mimo(False)
   model.load_weights(os.getcwd()+os.sep+weights_path)

   for v in vars:
      exec(v+'_PT = []')

   tmp = model.predict(testImagesX, batch_size=128)
   if len(vars)>1:
      counter = 0
      for v in vars:
         exec(v+'_PT.append(np.squeeze(tmp[counter]))')
         counter +=1
   else:
      exec(vars[0]+'_PT.append(np.asarray(np.squeeze(tmp)))')

   if len(vars)>1:
      for k in range(len(vars)):
         exec(vars[k]+'_predT = np.squeeze(np.mean(np.asarray('+vars[k]+'_PT), axis=0))')
   else:
      exec(vars[0]+'_predT = np.squeeze(np.mean(np.asarray('+vars[0]+'_PT), axis=0))')


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
      nrows = 1; ncols = 3
   elif len(vars)==2:
      nrows = 2; ncols = 1
   elif len(vars)==1:
      nrows = 1; ncols = 1

   ## make a plot
   fig = plt.figure()
   labs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
   for k in range(1,1+(nrows*ncols)):
      plt.subplot(nrows,ncols,k)
      x = eval(vars[k-1]+'_trueT')
      y = eval(vars[k-1]+'_predT')
      plt.plot(x, y, 'ko', markersize=5)
      plt.plot([ np.min(np.hstack((x,y))),  np.max(np.hstack((x,y)))], [ np.min(np.hstack((x,y))),  np.max(np.hstack((x,y)))], 'k', lw=2)

      plt.text(1.02*np.min(np.hstack((x,y))), 0.98*np.max(np.hstack((x,y))),'Test : '+str(np.mean(100*(np.abs(eval(vars[k-1]+'_predT') - eval(vars[k-1]+'_trueT')) / eval(vars[k-1]+'_trueT'))))[:5]+' %', fontsize=8)

      plt.title(r''+labs[k-1]+') '+vars[k-1], fontsize=8, loc='left')
      plt.xlabel('Actual '+vars[k-1])
      plt.ylabel('Estimated '+vars[k-1])
            
   varstring = ''.join([str(k)+'_' for k in vars])
   plt.savefig(weights_path.replace('.hdf5', '_predict.png'), dpi=300, bbox_inches='tight')
   plt.close()
   del fig



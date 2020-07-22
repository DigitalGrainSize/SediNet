
## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.3 (July 2020)

###===================================================
# import libraries
from sedinet_models import *

###===================================================
def get_data_generator(df, indices, greyscale, batch_size=16):
    """
    This function generates data for a batch of images and no metric, for  # "unseen" samples
    """

    for_training = False
    images = []
    while True:
        for i in indices:
            r = df.iloc[i]
            file = r['files']

            if greyscale==True:
               im = Image.open(file).convert('LA')
            else:
               im = Image.open(file)
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0

            if greyscale==True:
               images.append(np.expand_dims(im[:,:,0], axis=2))
            else:
               images.append(im)

            if len(images) >= batch_size:
                yield np.array(images)
                images = []
        if not for_training:
            break

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

            if greyscale==True:
               im = Image.open(file).convert('LA')
            else:
               im = Image.open(file)
            im = im.resize((IM_HEIGHT, IM_HEIGHT))
            im = np.array(im) / 255.0

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
def estimate_categorical(vars, csvfile, res_folder, dropout,
                         numclass, greyscale, name, mode):
   """
   This function uses a SediNet model for categorical prediction
   """

   ID_MAP = dict(zip(np.arange(numclass), [str(k) for k in range(numclass)]))

   ##======================================
   ## this randomly selects imagery for training and testing imagery sets
   ## while also making sure that both training and tetsing sets have
   ## at least 3 examples of each category
   test_idx, test_df = get_df(csvfile)

   # for 16GB RAM, used maximum of 200 samples to test on
   # need to change batch gnerator into a better keras one

   valid_gen = get_data_generator_1image(test_df, test_idx, True, ID_MAP,
                vars[0], np.min((200, len(test_idx))), greyscale, False)

   if SHALLOW is True:
      if DO_AUG is True:
          weights_path = name+"_"+mode+"_batch"+str(BATCH_SIZE)+"_im"+str(IM_HEIGHT)+\
                   "_"+str(IM_WIDTH)+"_shallow_"+vars[0]+"_"+CAT_LOSS+"_aug.hdf5"
      else:
          weights_path = name+"_"+mode+"_batch"+str(BATCH_SIZE)+"_im"+str(IM_HEIGHT)+\
                   "_"+str(IM_WIDTH)+"_shallow_"+vars[0]+"_"+CAT_LOSS+"_noaug.hdf5"
   else:
      if DO_AUG is True:
           weights_path = name+"_"+mode+"_batch"+str(BATCH_SIZE)+"_im"+str(IM_HEIGHT)+\
                   "_"+str(IM_WIDTH)+"_"+vars[0]+"_"+CAT_LOSS+"_aug.hdf5"
      else:
           weights_path = name+"_"+mode+"_batch"+str(BATCH_SIZE)+"_im"+str(IM_HEIGHT)+\
                   "_"+str(IM_WIDTH)+"_"+vars[0]+"_"+CAT_LOSS+"_noaug.hdf5"


   if not os.path.exists(weights_path):
       weights_path = res_folder + os.sep+ weights_path
       print("Using %s" % (weights_path))

   if numclass>0:
      ID_MAP = dict(zip(np.arange(numclass), [str(k) for k in range(numclass)]))


   if type(BATCH_SIZE)==list:
      SMs = []; 
      for batch_size, valid_batch_size, wp in zip(BATCH_SIZE, VALID_BATCH_SIZE, weights_path):
         sm = make_cat_sedinet(ID_MAP, dropout, greyscale)
         sm.load_weights(wp)
         SMs.append(sm)

   else:
     SM = make_cat_sedinet(ID_MAP, dropout, greyscale)

   if type(BATCH_SIZE)==list:
       predict_test_train_cat(test_df, None, test_idx, None, vars[0],
                         SMs, [i for i in ID_MAP.keys()], weights_path, greyscale,
                         name, DO_AUG)
   else:
       predict_test_train_cat(test_df, None, test_idx, None, vars[0],
                         SM, [i for i in ID_MAP.keys()], weights_path, greyscale,
                         name, DO_AUG)

   K.clear_session()

   ##===================================
   ## move model files and plots to the results folder
   tidy(name, res_folder)


###===================================================
def estimate_siso_simo(vars, csvfile, greyscale,
                       dropout, numclass, scale, name, mode, res_folder, batch_size, weights_path): #,
   """
   This function uses a sedinet model for continuous prediction
   """

   # if not os.path.exists(weights_path):
   #     weights_path = res_folder + os.sep+ weights_path
   #     print("Using %s" % (weights_path))

   ##======================================
   ## this randomly selects imagery for training and testing imagery sets
   ## while also making sure that both training and tetsing sets have
   ## at least 3 examples of each category
   test_idx, test_df = get_df(csvfile)

   ##==============================================
   ## create a sedinet model to estimate category
   # if type(BATCH_SIZE)==list:
   #     SMs = []
   #     for k in BATCH_SIZE:
   #         SMs.append(make_sedinet_siso_simo(vars, greyscale, dropout))
   #
   # else:
   SM = make_sedinet_siso_simo(vars, greyscale, dropout)

   if scale==True:
       CS = []
       for var in vars:
          cs = RobustScaler() #MinMaxScaler()
          cs.fit_transform(
            np.r_[test_df[var].values].reshape(-1,1)
            )
          CS.append(cs)
          del cs
   else:
       CS = []

   # test model
   # if numclass==0:
   if type(BATCH_SIZE)==list:
       predict_test_train_siso_simo(test_df, None, test_idx, None, vars,
                            SM, weights_path, name, mode, greyscale, CS,
                            dropout, scale, DO_AUG)
   else:
       predict_test_train_siso_simo(test_df, None, test_idx, None, vars,
                            SM, weights_path, name, mode, greyscale, CS,
                            dropout, scale, DO_AUG)

   K.clear_session()

   ##===================================
   ## move model files and plots to the results folder
   tidy(name, res_folder)


## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.3 (July 2020)

###===================================================
# import libraries
from sedinet_utils import *

###===================================================
def conv_block2(inp, filters=32, bn=True, pool=True, drop=True):
   """
   This function generates a SediNet convolutional block
   """
   # _ = Conv2D(filters=filters, kernel_size=3, activation='relu',
   #            kernel_initializer='he_uniform')(inp)

   _ = SeparableConv2D(filters=filters, kernel_size=3, activation='relu')(inp) #kernel_initializer='he_uniform'
   if bn:
       _ = BatchNormalization()(_)
   if pool:
       _ = MaxPool2D()(_)
   if drop:
       _ = Dropout(0.2)(_)
   return _

###===================================================
def make_cat_sedinet(ID_MAP, dropout, greyscale):
    """
    This function creates an implementation of SediNet for estimating
	sediment category
    """

    base = BASE_CAT ##30
    if greyscale==True:
       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 1))
    else:
       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))

    _ = conv_block2(input_layer, filters=base, bn=False, pool=False, drop=False) #x #
    _ = conv_block2(_, filters=base*2, bn=False, pool=True,drop=False)
    _ = conv_block2(_, filters=base*3, bn=False, pool=True,drop=False)
    _ = conv_block2(_, filters=base*4, bn=False, pool=True,drop=False)

    if not SHALLOW:
       _ = conv_block2(_, filters=base*5, bn=False, pool=True,drop=False)
       _ = conv_block2(_, filters=base*6, bn=False, pool=True,drop=False)

    bottleneck = GlobalMaxPool2D()(_)
    bottleneck = Dropout(dropout)(bottleneck)

     # for class prediction
    _ = Dense(units=CAT_DENSE_UNITS, activation='relu')(bottleneck)  ##128
    output = Dense(units=len(ID_MAP), activation='softmax', name='output')(_)

    model = Model(inputs=input_layer, outputs=[output])

    if CAT_LOSS == 'focal':
       model.compile(optimizer=OPT,
                  loss={'output': tfa.losses.SigmoidFocalCrossEntropy() },
                  metrics={'output': 'accuracy'})
    else:
       model.compile(optimizer=OPT, #'adam',
                  loss={'output': CAT_LOSS}, #'categorical_crossentropy'
                  metrics={'output': 'accuracy'})


    print("==========================================")
    print('[INFORMATION] Model summary:')
    model.summary()
    return model


###===================================================
def make_sedinet_siso_simo(vars, greyscale, dropout):
    """
    This function creates an implementation of SediNet for estimating
	sediment metric on a continuous scale
    """

    base = BASE_CONT ##30 ## suggested range = 20 -- 40
    if greyscale==True:
       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 1))
    else:
       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))

    _ = conv_block2(input_layer, filters=base, bn=False, pool=False, drop=False) #x #
    _ = conv_block2(_, filters=base*2, bn=False, pool=True,drop=False)
    _ = conv_block2(_, filters=base*3, bn=False, pool=True,drop=False)
    _ = conv_block2(_, filters=base*4, bn=False, pool=True,drop=False)
    _ = conv_block2(_, filters=base*5, bn=False, pool=True,drop=False)

    if not SHALLOW:
       _ = conv_block2(_, filters=base*6, bn=False, pool=True,drop=False)
       _ = conv_block2(_, filters=base*7, bn=False, pool=True,drop=False)

    _ = BatchNormalization(axis=-1)(_)
    bottleneck = GlobalMaxPool2D()(_)
    bottleneck = Dropout(dropout)(bottleneck)

    units = CONT_DENSE_UNITS ## suggested range 512 -- 1024
    _ = Dense(units=units, activation='relu')(bottleneck)

    outputs = []
    for var in vars:
       outputs.append(Dense(units=1, activation='linear', name=var+'_output')(_) )

    if CONT_LOSS == 'pinball':
       loss = dict(zip([k+"_output" for k in vars], [tfa.losses.PinballLoss(tau=.5) for k in vars]))
    else: ## 'mse'
       loss = dict(zip([k+"_output" for k in vars], ['mse' for k in vars])) #loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)  # Sum of squared error

    metrics = dict(zip([k+"_output" for k in vars], ['mae' for k in vars]))

    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer=OPT,loss=loss, metrics=metrics)
    #print("==========================================")
    #print('[INFORMATION] Model summary:')
    #model.summary()
    return model


# ###===================================================
# def conv_block_mbn(x, filters=32, alpha=1):
#    """
#    This function generates a sedinet convolutional block based on a
#    mobilenet base model
#    """
#    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#    x = Conv2D(int(filters * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#    return x


# ###===================================================
# def make_mlp(dim): #dense_neurons
# 	# define our MLP network
# 	dense_neurons = 4
# 	mlp = Sequential()
# 	mlp.add(Dense(8, input_dim=dim, activation="relu"))
# 	mlp.add(Dense(dense_neurons, activation="relu"))
# 	return mlp

# ###===================================================
# def conv_block(x, filters=32):
#    """
#    This function generates a custom sedinet convolutional block
#    """
#    x = Conv2D(filters=filters, kernel_size=3, activation='relu',
#               kernel_initializer='he_uniform')(x)
#    #x = BatchNormalization()(x)
#    x = MaxPool2D()(x)
#    #x = Dropout(0.2)(x)
#    return x

#
# ###===================================================
# def make_sedinet_miso_mimo(greyscale, dropout):
#     """
#     This function creates a mobilenetv1 style implementation of sedinet
# 	for estimating metric on a continuous scale
#     """
#
#     # create the sedinet model
#     if greyscale==True:
#        input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 1))
#     else:
#        input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
#
#     img_input = BatchNormalization(axis=-1)(input_layer) #x #
#     alpha=1
#
#     x = Conv2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#
#     for k in [64,128,128,256,256,512]:
#        x = conv_block_mbn(x, filters=k, alpha=alpha)
#
#     if not SHALLOW:
#         for i in range(5):
#             x = conv_block_mbn(x, filters=512, alpha=alpha)
#
#     for k in [1024,1024]:
#        x = conv_block_mbn(x, filters=k, alpha=alpha)
#
#     x = MaxPool2D()(x)
#
#     x = BatchNormalization(axis=-1)(x)
#     bottleneck = GlobalMaxPool2D()(x)
#     bottleneck = Dropout(dropout)(bottleneck)
#
#     model = Model(input_layer, bottleneck)
#
#     return model
#


#########

####===================================================
#def make_sedinet_custom_siso_simo(vars, greyscale):
#    """
#    This function creates a custom implementation of sedinet
#    for estimating metric on a continuous scale
#    """
#
#    base = 16

#    if greyscale==True:
#       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 1))
#    else:
#       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))

#    input_layer = BatchNormalization(axis=-1)(input_layer)
#
#    x = conv_block(input_layer, filters=base)
#    x = conv_block(x, filters=base*2)
#    x = conv_block(x, filters=base*3)
#    x = conv_block(x, filters=base*4)
#
#    x = BatchNormalization(axis=-1)(x)
#    bottleneck = GlobalMaxPool2D()(x)
#    bottleneck = Dropout(dropout)(bottleneck)

#    units = 1024
#    x = Dense(units=units, activation='relu')(bottleneck)

#    outputs = []
#    for var in vars:
#       outputs.append(Dense(units=1, activation='linear', name=var+'_output')(x) )

#    loss = dict(zip([k+"_output" for k in vars], ['mse' for k in vars]))
#    metrics = dict(zip([k+"_output" for k in vars], ['mae' for k in vars]))

#    model = Model(inputs=input_layer, outputs=outputs)
#    model.compile(optimizer=opt, loss=loss, metrics=metrics)
#    #print("==========================================")
#    #print('[INFORMATION] Model summary:')
#    #model.summary()
#    return model

####===================================================
#def make_sedinet_siso_simo(vars, greyscale, dropout):
#    """
#    This function creates a mobilenetv1 style implementation of sedinet
#    for estimating metric on a continuous scale
#    """

#    if greyscale==True:
#       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 1))
#    else:
#       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
#
#    img_input = BatchNormalization(axis=-1)(input_layer)
#    alpha=1
#
#    x = Conv2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#
#    for k in [64,128,128,256,256,512]:
#       x = conv_block_mbn(x, filters=k, alpha=alpha)

#    if not shallow:
#        for i in range(5):
#            x = conv_block_mbn(x, filters=512, alpha=alpha)

#    for k in [1024,1024]:
#       x = conv_block_mbn(x, filters=k, alpha=alpha)
#
#    x = MaxPool2D()(x)
#
#    x = BatchNormalization(axis=-1)(x)
#    bottleneck = GlobalMaxPool2D()(x)
#    bottleneck = Dropout(dropout)(bottleneck)

#    units = 1024
#    x = Dense(units=units, activation='relu')(bottleneck)

#    outputs = []
#    for var in vars:
#       outputs.append(Dense(units=1, activation='linear', name=var+'_output')(x) )

#    loss = dict(zip([k+"_output" for k in vars], ['mse' for k in vars]))
#    metrics = dict(zip([k+"_output" for k in vars], ['mae' for k in vars]))

#    model = Model(inputs=input_layer, outputs=outputs)
#    model.compile(optimizer=opt, loss=loss, metrics=metrics)
#    #print("==========================================")
#    #print('[INFORMATION] Model summary:')
#    #model.summary()
#    return model

####===================================================
#def make_sedinet_custom_miso_mimo(vars, greyscale):
#    """
#    This function creates a custom implementation of sedinet for estimating metric on a continuous scale
#    """
#
#    base = 16
#    if greyscale==True:
#       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 1))
#    else:
#       input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
#
#    input_layer = BatchNormalization(axis=-1)(input_layer)
#
#    x = conv_block(input_layer, filters=base)
#    x = conv_block(x, filters=base*2)
#    x = conv_block(x, filters=base*3)
#    x = conv_block(x, filters=base*4)
#
#    x = BatchNormalization(axis=-1)(x)
#    bottleneck = GlobalMaxPool2D()(x)
#    bottleneck = Dropout(dropout)(bottleneck)

#    model = Model(input_layer, bottleneck)
#    return model
#





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
        _ = Dropout(0.2)(_) ##added DB        
    return _

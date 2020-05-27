try:
    from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Dropout, concatenate
    from keras.models import Model, load_model
except:
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Dropout
    from tensorflow.keras.models import Model


# Unet model for segmentation of color/greyscale images https://github.com/zhixuhao/unet
def unet(input_shape, num_classes=5, droprate=None, linear=False):
    model_name = 'unet'

    if droprate:
        droprate = droprate
        # model_name = 'unet_' + str(droprate).replace('.','')

    inputs = Input(shape=input_shape)
    # 1. down
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(droprate)(pool1)

    # 2. down
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    # 3. down
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    # 4. down
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(droprate)(pool4)

    # 5. down
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    # Upsampling
    # 1. up
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    add6 = concatenate([conv4, up6], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    # 2. up
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    add7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    # 3. up
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    add8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    # 4. up
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    add9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    if num_classes == 1:
        if linear:
            conv10 = Conv2D(num_classes, 1, activation='linear')(conv9)
            # reshape1 = Reshape((num_pixels, num_classes))(conv10)
        else:
            conv10 = Conv2D(num_classes, 1, activation='sigmoid')(conv9)
            # reshape1 = Reshape((num_pixels, num_classes))(conv10)

    else:
        if linear:
            conv10 = Conv2D(num_classes, 1, activation='linear')(conv9)
            # reshape1 = Reshape((num_pixels, num_classes))(conv10)
        else:
            conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)
            # reshape1 = Reshape((num_pixels, num_classes))(conv10)

    model = Model(inputs=inputs, outputs=conv10)
    return model, model_name

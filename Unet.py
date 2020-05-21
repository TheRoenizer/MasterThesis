

try:
    from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, add, UpSampling2D, Dropout, Reshape
    from keras.models import Model, load_model
except:
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, add, UpSampling2D, Dropout, Reshape
    from tensorflow.keras.models import Model

'''
def load_data(data_path, dtype=np.float32):
    n = 99            # Number of images
    m = 5             # Number of labels
    dim = (480, 640)  # Image dimensions

    images = np.empty((n, *dim, 3), dtype=dtype)
    labels = np.empty((n, *dim, m), dtype=dtype)

    for i in range(n):
        image_path = os.path.join(data_path, 'Jigsaw annotations/Images/Suturing ({}).png'.format(i + 1))
        images[i] = cv.imread(image_path).astype(dtype)
        images[i] = cv.normalize(images[i], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

        for j in range(m):
            label_path = os.path.join(data_path, 'Jigsaw annotations/Annotated/Suturing ({})/data/00{}.png'.format(i + 1, j))
            labels[i, ..., j] = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
            labels[i, ..., j] = cv.threshold(labels[i, ..., j], dst=None, thresh=1, maxval=255, type=cv.THRESH_BINARY)[1]
            labels[i, ..., j] = cv.normalize(labels[i, ..., j], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

    return images, labels
'''

"""Unet model for segmentation of color/greyscale images https://github.com/zhixuhao/unet"""
def unet(input_shape, num_classes=5, droprate=None, linear=False):
    model_name = 'unet'

    if droprate:
        droprate = droprate
        # model_name = 'unet_' + str(droprate).replace('.','')

    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(droprate)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(droprate)(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    add6 = add([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    add7 = add([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    add8 = add([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    add9 = add([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

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

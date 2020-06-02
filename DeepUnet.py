import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

try:
    from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, add, UpSampling2D, Dropout, Reshape, \
        concatenate, ZeroPadding2D, Cropping2D
    from keras.models import Model, load_model
except:
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, add, UpSampling2D, Dropout, Reshape
    from tensorflow.keras.models import Model

def deep_unet(input_shape, num_classes=5, padding=0):
    model_name = 'deep_unet'
    inputs = Input(shape=input_shape)

    # add zero padding such to the height so it is divisible by 128 (2*7)
    x = ZeroPadding2D((padding, 0))(inputs)

    # Down pooling stage
    # 1. down
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    conv1_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    # 2. down
    conv2_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_1)
    y2 = add([pool1, conv2_2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(y2)

    # 3. down
    conv3_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_1)
    y3 = add([pool2, conv3_2])
    pool3 = MaxPooling2D(pool_size=(2, 2))(y3)

    # 4. down
    conv4_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_1)
    y4 = add([pool3, conv4_2])
    pool4 = MaxPooling2D(pool_size=(2, 2))(y4)

    # 4. down
    conv5_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_1)
    y5 = add([pool4, conv5_2])
    pool5 = MaxPooling2D(pool_size=(2, 2))(y5)

    # 6. down
    conv6_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6_1)
    y6 = add([pool5, conv6_2])
    pool6 = MaxPooling2D(pool_size=(2, 2))(y6)

    # 7. down
    conv7_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool6)
    conv7_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_1)
    y7 = add([pool6, conv7_2])
    pool7 = MaxPooling2D(pool_size=(2, 2))(y7)

    # Up sampling stage
    # 1. up
    # no concat in beginning of first block due to no conv/filters
    conv8_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool7)
    conv8_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_1)
    y8 = add([pool7, conv8_2])
    up1 = UpSampling2D(size=(2, 2))(y8)

    # 2. up
    concat2 = concatenate([pool6, up1], axis=-1)
    conv9_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat2)
    conv9_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_1)
    y9 = add([pool6, conv9_2])
    up2 = UpSampling2D(size=(2, 2))(y9)

    # 3. up
    concat3 = concatenate([pool5, up2], axis=-1)
    conv10_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat3)
    conv10_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10_1)
    y10 = add([pool5, conv10_2])
    up3 = UpSampling2D(size=(2, 2))(y10)

    # 4. up
    concat4 = concatenate([pool4, up3], axis=-1)
    conv11_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat4)
    conv11_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11_1)
    y11 = add([pool4, conv11_2])
    up4 = UpSampling2D(size=(2, 2))(y11)

    # 5. up
    concat5 = concatenate([pool3, up4], axis=-1)
    conv12_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat5)
    conv12_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12_1)
    y12 = add([pool3, conv12_2])
    up5 = UpSampling2D(size=(2, 2))(y12)

    # 6. up
    concat6 = concatenate([pool2, up5], axis=-1)
    conv13_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat6)
    conv13_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13_1)
    y13 = add([pool2, conv13_2])
    up6 = UpSampling2D(size=(2, 2))(y13)

    # 7. up
    concat7 = concatenate([pool1, up6], axis=-1)
    conv14_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat7)
    conv14_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14_1)
    y14 = add([pool1, conv14_2])
    up7 = UpSampling2D(size=(2, 2))(y14)

    conv_final = Conv2D(num_classes, 1, activation='softmax')(up7)
    x = Cropping2D((padding, 0))(conv_final)

    model = Model(inputs=inputs, outputs=x)
    return model, model_name






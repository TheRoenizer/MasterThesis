import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Hvis du vil bruge "kort 1":
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ellers:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# hvis du træne på CPU'en:
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
from keras import backend as K
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

try:
    from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, add, UpSampling2D, Dropout, Reshape, concatenate
    from keras.models import Model, load_model
except:
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, add, UpSampling2D, Dropout, Reshape
    from tensorflow.keras.models import Model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Christoffer:
# PATH = 'C:/Users/chris/Google Drive/'
# Linux:
PATH = '/home/jsteeen/'

batch_size = 1
num_epochs = 100
#num_pixels = 480 * 640
weights = [.5, 1.5, 1.5, 1, 1] # [background, gripper, gripper, shaft, shaft]


def deep_unet(input_shape, num_classes=5, droprate=None, linear=False):
    model_name = 'deep_unet'
    inputs = Input(shape=input_shape)

    # Down pooling stage
    # 1. down
    #conv1_0 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
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
    # no concat in beginning of first layer due to no conv/filters
    conv8_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool7)
    conv8_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_1)
    y8 = add([pool7, conv8_2])
    up1 = UpSampling2D(size=(2, 2))(y8)
    concat1 = concatenate([pool6, up1], axis=-1)

    up2 = UpSampling2D(size=(2, 2))(y8)
    add2 = add([y6, up2])
    conv9_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add2)
    conv9_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_1)
    y9 = conv9_2 * tf.keras.activations.relu(conv9_2, alpha=0.0, max_value=None, threshold=0)(conv9_1 * y8) + y8

    up3 = UpSampling2D(size=(2, 2))(y9)
    add3 = add([y5, up3])
    conv10_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add3)
    conv10_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10_1)
    y10 = conv10_2 * tf.keras.activations.relu(conv10_2, alpha=0.0, max_value=None, threshold=0)(conv10_1 * y9) + y9

    up4 = UpSampling2D(size=(2, 2))(y10)
    add4 = add([y4, up4])
    conv11_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add4)
    conv11_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11_1)
    y11 = conv11_2 * tf.keras.activations.relu(conv11_2, alpha=0.0, max_value=None, threshold=0)(conv11_1 * y10) + y10

    up5 = UpSampling2D(size=(2, 2))(y11)
    add5 = add([y3, up5])
    conv12_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add5)
    conv12_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12_1)
    y12 = conv12_2 * tf.keras.activations.relu(conv12_2, alpha=0.0, max_value=None, threshold=0)(conv12_1 * y11) + y11

    up6 = UpSampling2D(size=(2, 2))(y12)
    add6 = add([y2, up6])
    conv13_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add6)
    conv13_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13_1)
    y13 = conv13_2 * tf.keras.activations.relu(conv13_2, alpha=0.0, max_value=None, threshold=0)(conv13_1 * y12) + y12

    up7 = UpSampling2D(size=(2, 2))(y13)
    add7 = add([y1, up7])
    conv14_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(add7)
    conv14_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14_1)
    y14 = conv14_2 * tf.keras.activations.relu(conv14_2, alpha=0.0, max_value=None, threshold=0)(conv14_1 * y13) + y13

    conv_final = Conv2D(num_classes, 1, activation='softmax')(y14)

    model = Model(inputs=inputs, outputs=conv_final)
    return model, model_name

'''
def down_block(i, inputs1):
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    y = conv2 * tf.keras.activations.relu(conv2, alpha=0.0, max_value=None, threshold=0)(conv1 * inputs1) + inputs1
    pool1 = MaxPooling2D(pool_size=(2, 2))(y)
    return pool, y


def up_block(inputs1):
    up1 = UpSampling2D(size=(2, 2))(inputs1)
    add1 = add([y7, up1])
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    y = conv2 * tf.keras.activations.relu(conv2, alpha=0.0, max_value=None, threshold=0)(conv1 * pool) + pool

    return conv1
'''

def weighted_categorical_crossentropy(weights=[1]):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * -K.log(y_pred) * weights
        loss = K.sum(loss, -1)
        return loss

    return loss

def display(display_list, epoch_display):
    fig = plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask after epoch {}'.format(epoch_display + 1)]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig("Pictures/afterEpoch{}.png".format(epoch_display + 1))
    # plt.show()
    plt.close(fig)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(epoch_show_predictions, image_num=1):
    pred_mask = deep_unet.predict(imgs_val[image_num][tf.newaxis, ...]) * 255
    display([imgs_val[image_num], lbls_val[image_num], create_mask(pred_mask)], epoch_show_predictions)


class DisplayCallback(tf.keras.callbacks.Callback):
    # @staticmethod
    def on_epoch_end(self, epoch_callback, logs=None):
        clear_output(wait=True)
        show_predictions(epoch_callback)
        print('\nSample Prediction after epoch {}\n'.format(epoch_callback + 1))

# Load images
imgs_train = np.zeros((79, 480, 640, 3))
print('Loading images...')
for i in range(1, 80):
    # print('Progress: ' + str(i) + ' of 79')
    path = PATH + '/Jigsaw annotations/Images/Suturing (' + str(i) + ').png'
    img = np.array(Image.open(path))[np.newaxis]
    img = img / 255
    imgs_train[i-1] = img

imgs_val = np.zeros((10, 480, 640, 3))
for i in range(80, 90):
    # print('Progress: ' + str(i) + ' of 89')
    path = PATH + '/Jigsaw annotations/Images/Suturing (' + str(i) + ').png'
    img = np.array(Image.open(path))[np.newaxis]
    img = img / 255
    imgs_val[i-80] = img

imgs_test = np.zeros((10, 480, 640, 3))
for i in range(90, 100):
    # print('Progress: ' + str(i) + ' of 89')
    path = PATH + '/Jigsaw annotations/Images/Suturing (' + str(i) + ').png'
    img = np.array(Image.open(path))[np.newaxis]
    img = img / 255
    imgs_test[i-90] = img

print('Images loaded!')
print('Loading labels...')

# Load labels
lbls_train = np.zeros((79, 480, 640))
sample_weight = np.zeros((79, 480, 640))
for i in range(1, 80):
    # print('Progress: ' + str(i) + ' of 79')
    path1 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/000.png'
    path2 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/002.png'
    path3 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/003.png'
    path4 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/001.png'
    img1 = cv.imread(path1, 2)
    img2 = cv.imread(path2, 2)
    img3 = cv.imread(path3, 2)
    img4 = cv.imread(path4, 2)
    change1_to = np.where(img1[:, :] != 0)
    change2_to = np.where(img2[:, :] != 0)
    change3_to = np.where(img3[:, :] != 0)
    change4_to = np.where(img4[:, :] != 0)
    img1[change1_to] = 1
    img2[change2_to] = 2
    img3[change3_to] = 3
    img4[change4_to] = 4
    weight1 = np.zeros((480, 640))
    weight2 = np.zeros((480, 640))
    weight3 = np.zeros((480, 640))
    weight4 = np.zeros((480, 640))
    weight1[change1_to] = 10
    weight2[change2_to] = 1
    weight3[change3_to] = 10
    weight4[change4_to] = 1
    img = img1 + img2 + img3 + img4
    weight = weight1 + weight2 + weight3 + weight4
    change_5 = np.where(img[:, :] == 5)
    img[change_5] = 0
    change_overlap = np.where(weight[:, :] == 11)
    weight[change_overlap] = 0
    lbls_train[i-1] = img
    sample_weight[i-1] = weight

lbls_train_onehot = tf.keras.utils.to_categorical(lbls_train, num_classes=5, dtype='float32')
lbls_train = lbls_train.reshape((79, 480, 640, -1))

lbls_val = np.zeros((10, 480, 640))
for i in range(80, 90):
    # print('Progress: ' + str(i) + ' of 89')
    path1 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/000.png'
    path2 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/002.png'
    path3 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/003.png'
    path4 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/001.png'
    img1 = cv.imread(path1, 2)
    img2 = cv.imread(path2, 2)
    img3 = cv.imread(path3, 2)
    img4 = cv.imread(path4, 2)
    change1_to = np.where(img1[:, :] != 0)
    change2_to = np.where(img2[:, :] != 0)
    change3_to = np.where(img3[:, :] != 0)
    change4_to = np.where(img4[:, :] != 0)
    img1[change1_to] = 1
    img2[change2_to] = 2
    img3[change3_to] = 3
    img4[change4_to] = 4
    img = img1 + img2 + img3 + img4
    change_5 = np.where(img[:, :] == 5)
    img[change_5] = 0
    lbls_val[i-80] = img

lbls_val_onehot = tf.keras.utils.to_categorical(lbls_val, num_classes=5, dtype='float32')
lbls_val = lbls_val.reshape((10, 480, 640, -1))

lbls_test = np.zeros((10, 480, 640))
for i in range(90, 100):
    # print('Progress: ' + str(i) + ' of 89')
    path1 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/000.png'
    path2 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/002.png'
    path3 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/003.png'
    path4 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/001.png'
    img1 = cv.imread(path1, 2)
    img2 = cv.imread(path2, 2)
    img3 = cv.imread(path3, 2)
    img4 = cv.imread(path4, 2)
    change1_to = np.where(img1[:, :] != 0)
    change2_to = np.where(img2[:, :] != 0)
    change3_to = np.where(img3[:, :] != 0)
    change4_to = np.where(img4[:, :] != 0)
    img1[change1_to] = 1
    img2[change2_to] = 2
    img3[change3_to] = 3
    img4[change4_to] = 4
    img = img1 + img2 + img3 + img4
    change_5 = np.where(img[:, :] == 5)
    img[change_5] = 0
    lbls_test[i - 90] = img

lbls_test_onehot = tf.keras.utils.to_categorical(lbls_test, num_classes=5, dtype='float32')
lbls_test = lbls_test.reshape((10, 480, 640, -1))

print('Labels loaded!')

imgs_train2 = np.zeros((480, 640, 3))
(deep_unet, name) = deep_unet(imgs_train2.shape, num_classes=5, droprate=0.0, linear=False)

deep_unet.compile(optimizer='adam', loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])

show_predictions(-1)

history = deep_unet.fit(imgs_train, lbls_train_onehot, validation_data=[imgs_val, lbls_val_onehot],
                  batch_size=batch_size,
                  epochs=num_epochs,
                  verbose=1,
                  shuffle=True,
                  callbacks=[DisplayCallback()])

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

try:
    from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, add, UpSampling2D, Dropout
    from keras.models import Model
except:
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, add, UpSampling2D, Dropout
    from tensorflow.keras.models import Model

"""Unet model for segmentation of color/greyscale images https://github.com/zhixuhao/unet"""


# Christoffer:
PATH = 'C:/Users/chris/Google Drive/'
# Jonathan:
#PATH = '/Users/jonathansteen/Google Drive/'
# Linux:
# PATH = '/home/jsteeen/'


def unet(input_shape, num_classes=1, droprate=None, linear=False):
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
        else:
            conv10 = Conv2D(num_classes, 1, activation='sigmoid')(conv9)

    else:
        if linear:
            conv10 = Conv2D(num_classes, 1, activation='linear')(conv9)
        else:
            conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model, model_name


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask shaft', 'True Mask gripper', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


imgs_train = np.zeros((79, 480, 640, 3))
for i in range(1, 80):
    print('Progress: ' + str(i) + ' of 79')
    path = PATH + '/Jigsaw annotations/Images/Suturing (' + str(i) + ').png'
    img = np.array(Image.open(path))[np.newaxis]
    img = img / 255
    print(img.shape)
    imgs_train[i-1] = img


# print(imgs_train)

imgs_val = np.zeros((10, 480, 640, 3))
for i in range(80, 90):
    print('Progress: ' + str(i) + ' of 89')
    path = PATH + '/Jigsaw annotations/Images/Suturing (' + str(i) + ').png'
    img = np.array(Image.open(path))[np.newaxis]
    img = img / 255
    imgs_val[i-80] = img
    # print(imgs_val)

# Label for shafts
lbls_train_shaft = np.zeros((79, 480, 640))
for i in range(1, 80):
    print('Progress: ' + str(i) + ' of 79')
    path1 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/003.png'
    path2 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/001.png'
    img1 = cv2.imread(path1, 2)
    img2 = cv2.imread(path2, 2)
    img = img1 + img2
    ret, binary_img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    binary_img = binary_img.astype('float32') / 255
    final_img = np.array(binary_img)[np.newaxis]
    lbls_train_shaft[i-1] = final_img

lbls_train_shaft = lbls_train_shaft.reshape((79, 480, 640, -1))
#print(lbls_train_shaft.shape)

lbls_val_shaft = np.zeros((10, 480, 640))
for i in range(80, 90):
    print('Progress: ' + str(i) + ' of 89')
    path1 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/003.png'
    path2 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/001.png'
    img1 = cv2.imread(path1, 2)
    img2 = cv2.imread(path2, 2)
    img = img1 + img2
    ret, binary_img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    binary_img = binary_img.astype('float32') / 255
    final_img = np.array(binary_img)[np.newaxis]
    lbls_val_shaft[i-80] = final_img

lbls_val_shaft = lbls_val_shaft.reshape((10, 480, 640, -1))
# print(imgs_train.shape)
# print(imgs_val.shape)
# print(lbls_train.shape)
# print(lbls_val.shape)

# Label for grippers
lbls_train_gripper = np.zeros((79, 480, 640))
for i in range(1, 80):
    print('Progress: ' + str(i) + ' of 79')
    path1 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/000.png'
    path2 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/002.png'
    img1 = cv2.imread(path1, 2)
    img2 = cv2.imread(path2, 2)
    img = img1 + img2
    ret, binary_img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    binary_img = binary_img.astype('float32') / 255
    final_img = np.array(binary_img)[np.newaxis]
    print('SE OGSÅ HER:')
    print(final_img.shape)
    lbls_train_gripper[i-1] = final_img

lbls_train_gripper = lbls_train_gripper.reshape((79, 480, 640, -1))
print('SE HER:')
print(lbls_train_gripper.shape)

lbls_val_gripper = np.zeros((10, 480, 640))
for i in range(80, 90):
    print('Progress: ' + str(i) + ' of 89')
    path1 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/000.png'
    path2 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/002.png'
    img1 = cv2.imread(path1, 2)
    img2 = cv2.imread(path2, 2)
    img = img1 + img2
    ret, binary_img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    binary_img = binary_img.astype('float32') / 255
    final_img = np.array(binary_img)[np.newaxis]
    lbls_val_gripper[i-80] = final_img

lbls_val_gripper = lbls_val_gripper.reshape((10, 480, 640, -1))

imgs_train2 = np.zeros((480, 640, 3))
(unet, name) = unet(imgs_train2.shape, num_classes=1, droprate=0.0, linear=False)

unet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#tf.keras.metrics.MeanIoU(num_classes=2)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(image_num=0):
    pred_mask = unet.predict(imgs_train[image_num][tf.newaxis, ...]) * 255
    # print(pred_mask.shape)
    display([imgs_train[image_num], lbls_train_shaft[image_num], lbls_train_gripper[image_num], pred_mask[0]])




class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


epoch = 10
show_predictions()

unet.fit(imgs_train, [lbls_train_shaft, lbls_train_gripper], validation_data=[imgs_val, lbls_val_shaft, lbls_val_gripper],
         batch_size=1,
         epochs=epoch,
         verbose=1,
         shuffle=True,
         callbacks=[DisplayCallback()])

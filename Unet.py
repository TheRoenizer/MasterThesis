import tensorflow as tf
from keras import backend as K
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
# PATH = 'C:/Users/chris/Google Drive/'
# Jonathan:
PATH = '/Users/jonathansteen/Google Drive/'
# Linux:
# PATH = '/home/jsteeen/'
# PATH = '/home/croen/'

epoch = 100
class_weight = {
    0: 1.,
    1: 50.,
    2: 2.,
    3: 1.,
    4: 1.
}

Loss_function = 1   # 1=focal_loss, 2=dice_loss, 3=jaccard_loss, 4=tversky_loss

FL_alpha = .25      # Focal loss alpha
FL_gamma = 2.       # Focal loss gamma
TL_beta = 3         # Tversky loss beta


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probabilities of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        focal_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(focal_loss, axis=1)

    return categorical_focal_loss_fixed


def dice_loss():
    def dice_loss_fixed(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

        return 1 - (numerator+1) / (denominator+1)

    return dice_loss_fixed


def jaccard_loss():
    def jaccard_loss_fixed(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.reduce_sum(y_true + y_pred - y_true * y_pred, axis=-1)

        return 1 - (numerator + 1) / (denominator + 1)

    return jaccard_loss_fixed


def tversky_loss(beta):
    def tversky_loss_fixed(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.reduce_sum(y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred), axis=-1)

        return 1 - (numerator + 1) / (denominator + 1)

    return tversky_loss_fixed


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
        else:
            conv10 = Conv2D(num_classes, 1, activation='sigmoid')(conv9)

    else:
        if linear:
            conv10 = Conv2D(num_classes, 1, activation='linear')(conv9)
        else:
            conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model, model_name


def display(display_list, epoch_display):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask after epoch {}'.format(epoch_display + 1)]
    for j in range(len(display_list)):
        plt.subplot(1, len(display_list), j + 1)
        plt.title(title[j])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[j]))
        plt.axis('off')
    plt.savefig("Pictures/afterEpoch{}.png".format(epoch_display + 1))
    plt.show()


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

print('Done!')
print('Loading labels...')
# Labels
lbls_train = np.zeros((79, 480, 640))
for i in range(1, 80):
    # print('Progress: ' + str(i) + ' of 79')
    path1 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/000.png'
    path2 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/002.png'
    path3 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/003.png'
    path4 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/001.png'
    img1 = cv2.imread(path1, 2)
    img2 = cv2.imread(path2, 2)
    img3 = cv2.imread(path3, 2)
    img4 = cv2.imread(path4, 2)
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
    lbls_train[i-1] = img

lbls_train_onehot = tf.keras.utils.to_categorical(lbls_train, num_classes=5, dtype='float32')
lbls_train = lbls_train.reshape((79, 480, 640, -1))

lbls_val = np.zeros((10, 480, 640))
for i in range(80, 90):
    # print('Progress: ' + str(i) + ' of 89')
    path1 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/000.png'
    path2 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/002.png'
    path3 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/003.png'
    path4 = PATH + '/Jigsaw annotations/Annotated/Suturing (' + str(i) + ')' + '/data/001.png'
    img1 = cv2.imread(path1, 2)
    img2 = cv2.imread(path2, 2)
    img3 = cv2.imread(path3, 2)
    img4 = cv2.imread(path4, 2)
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

print('Done!')
lbls_val_onehot = tf.keras.utils.to_categorical(lbls_val, num_classes=5, dtype='float32')
lbls_val = lbls_val.reshape((10, 480, 640, -1))

imgs_train2 = np.zeros((480, 640, 3))
(unet, name) = unet(imgs_train2.shape, num_classes=5, droprate=0.0, linear=False)

if Loss_function == 1:
    print('Categorical Focal Loss with gamma = ' + str(FL_gamma) + ' and alpha = ' + str(FL_alpha))
    unet.compile(optimizer='adam', loss=[categorical_focal_loss(gamma=FL_gamma, alpha=FL_alpha)], metrics=['accuracy'])
elif Loss_function == 2:
    print('Dice Loss')
    unet.compile(optimizer='adam', loss=[dice_loss()], metrics=['accuracy'])
elif Loss_function == 3:
    print('Jaccard Loss')
    unet.compile(optimizer='adam', loss=[jaccard_loss()], metrics=['accuracy'])
elif Loss_function == 4:
    print('Tversky Loss with beta = ' + str(TL_beta))
    unet.compile(optimizer='adam', loss=[tversky_loss(beta=TL_beta)], metrics=['accuracy'])
else:
    print('No loss function')

# tf.keras.metrics.MeanIoU(num_classes=2)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(epoch_show_predictions, image_num=1):
    pred_mask = unet.predict(imgs_val[image_num][tf.newaxis, ...]) * 255
    # print(pred_mask.shape)
    display([imgs_val[image_num], lbls_val[image_num], create_mask(pred_mask)], epoch_show_predictions)


class DisplayCallback(tf.keras.callbacks.Callback):
    # @staticmethod
    def on_epoch_end(self, epoch_callback, logs=None):
        clear_output(wait=True)
        show_predictions(epoch_callback)
        print('\nSample Prediction after epoch {}\n'.format(epoch_callback + 1))
        if logs:
            print('logs')


show_predictions(-1)

model_history = unet.fit(imgs_train, lbls_train_onehot, validation_data=[imgs_val, lbls_val_onehot],
                         batch_size=1,
                         epochs=epoch,
                         verbose=1,
                         shuffle=True,
                         callbacks=[DisplayCallback()],
                         class_weight=class_weight)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(epoch)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Pictures/Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 5])
plt.legend()
plt.show()

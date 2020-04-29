from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from Unet import *
from Loss_functions import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Hvis du vil bruge "kort 1":
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ellers:
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# hvis du træne på CPU'en:
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_data(data_path, dtype=np.float32):
    N = 99            # Number of images
    M = 5             # Number of labels
    DIM = (480, 640)  # Image dimensions

    images = np.empty((N, *DIM, 3), dtype=dtype)
    labels = np.empty((N, *DIM, M), dtype=dtype)
    labels_display = np.empty((N, *DIM, 1), dtype=dtype)

    for i in range(N):
        image_path = os.path.join(data_path, 'Images/Suturing ({}).png'.format(i + 1))
        images[i] = cv.imread(image_path).astype(dtype)
        images[i] = cv.normalize(images[i], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

        #d = {}

        for j in range(M):
            label_path = os.path.join(data_path, 'Annotated/Suturing ({})/data/00{}.png'.format(i + 1, j))
            labels[i,...,j] = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
            labels_display[i, ..., 0] += labels[i, ..., j]
            labels[i,...,j] = cv.threshold(labels[i,...,j], dst=None, thresh=1, maxval=255, type=cv.THRESH_BINARY)[1]
            labels[i,...,j] = cv.normalize(labels[i,...,j], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)


    return images, labels, labels_display

# Functions used to display images after each epoch
def display(display_list, epoch_display):
    fig = plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask after epoch {}'.format(epoch_display + 1)]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        # img = tf.keras.preprocessing.image.array_to_img(display_list[i])
        # img.save("afterEpoch{}.png".format(epoch))
        plt.axis('off')

    plt.savefig("Pictures/afterEpoch{}.png".format(epoch_display + 1))
    # plt.show()
    plt.close(fig)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(epoch_show_predictions, image_num=1):
    pred_mask = unet.predict(imgs_val[image_num][tf.newaxis, ...]) * 255
    display([imgs_val[image_num], lbls_display_val[image_num], create_mask(pred_mask)], epoch_show_predictions)


class DisplayCallback(tf.keras.callbacks.Callback):
    # @staticmethod
    def on_epoch_end(self, epoch_callback, logs=None):
        clear_output(wait=True)
        show_predictions(epoch_callback)
        print('\nSample Prediction after epoch {}\n'.format(epoch_callback + 1))


# A little test:

epoch = 100
weights = [.5, 1.5, 1.5, 1, 1]

#images, labels, labels_display = load_data('/home/jsteeen/Jigsaw annotations')
images, labels, labels_display = load_data('C:/Users/chris/Google Drive/Jigsaw annotations')

imgs_train = images[0:79]
imgs_val = images[79:89]
imgs_test = images[89:99]

lbls_train = labels[0:79]
lbls_val = labels[79:89]
lbls_test = labels[89:99]

lbls_display_train = labels_display[0:79]
lbls_display_val = labels_display[79:89]
lbls_display_test = labels_display[89:99]

print("imgs_val: " + str(imgs_val.shape))
print("lbls_display_val: " + str(lbls_display_val.shape))
imgs_train2 = np.zeros((480, 640, 3))
(unet, name) = unet(imgs_train2.shape, num_classes=5, droprate=0.0, linear=False)

unet.compile(optimizer='adam',
                     loss=weighted_categorical_crossentropy(weights),
                     metrics=['accuracy', iou_coef, dice_coef])

model_history = unet.fit(imgs_train, lbls_train, validation_data=[imgs_val, lbls_val],
                             batch_size = 1,
                             epochs=epoch,
                             verbose=1,
                             shuffle=True,
                             callbacks=[DisplayCallback()])
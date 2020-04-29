from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Hvis du vil bruge "kort 1":
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ellers:
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# hvis du træne på CPU'en:
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from Unet import *
from Loss_functions import *

print('Tensorflow version: '+tf.__version__)
print('Numpy version: '+np.__version__)
print('Keras version: '+tf.keras.__version__)

which_path = 1 # 1 = local c, 2 = local j, 3 = remote
train = True
epoch = 100
num_pixels = 480 * 640
weights = [.5, 1.5, 1.5, 1, 1] # [background, gripper, gripper, shaft, shaft]
# sample_weight = np.zeros((79, num_pixels))

Loss_function = 5   # 1=focal_loss, 2=dice_loss, 3=jaccard_loss, 4=tversky_loss, 5=weighted_categorical_crossentropy 6=categorical_cross_entropy

FL_alpha = .25      # Focal loss alpha
FL_gamma = 2.       # Focal loss gamma
TL_beta = 3         # Tversky loss beta

if which_path == 1:
    # Christoffer:
    PATH = 'C:/Users/chris/Google Drive/'
elif which_path ==2:
    # Jonathan
    PATH = '/Users/jonathansteen/Google Drive/'
elif which_path == 3:
    # Linux:
    PATH = '/home/jsteeen/'

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
print("lbls_train_onehot: " + str(lbls_train_onehot.shape))
print("lbls_train: " + str(lbls_train.shape))
lbls_train = lbls_train.reshape((79, 480, 640, -1))
print("lbls_train: " + str(lbls_train.shape))

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

# print(imgs_train.shape)
# print(imgs_val.shape)
# print(lbls_train.shape)
# print(lbls_val.shape)

#imgs_train2 = np.zeros((480, 640, 3))
#(unet, name) = unet(imgs_train2.shape, num_classes=1, droprate=0.0, linear=False)

#unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

if train:
    imgs_train2 = np.zeros((480, 640, 3))
    (unet, name) = unet(imgs_train2.shape, num_classes=5, droprate=0.0, linear=False)

    unet.summary()

    if Loss_function == 1:
        print('Categorical Focal Loss with gamma = ' + str(FL_gamma) + ' and alpha = ' + str(FL_alpha))
        unet.compile(optimizer='adam',
                     loss=categorical_focal_loss(gamma=FL_gamma, alpha=FL_alpha),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 2:
        print('Dice Loss')
        unet.compile(optimizer='adam',
                     loss=dice_loss(),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 3:
        print('Jaccard Loss')
        unet.compile(optimizer='adam',
                     loss=jaccard_loss(),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 4:
        print('Tversky Loss with beta = ' + str(TL_beta))
        unet.compile(optimizer='adam',
                     loss=tversky_loss(beta=TL_beta),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 5:
        print('Weighted categorical crossentropy with weights = ' + str(weights))
        unet.compile(optimizer='adam',
                     loss=weighted_categorical_crossentropy(weights),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 6:
        print('Categorical crossentropy')
        unet.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy', iou_coef, dice_coef])
    else:
        print('No loss function')

    # tf.keras.metrics.MeanIoU(num_classes=2)

    train_dataset = tf.data.Dataset.from_tensor_slices((imgs_train, lbls_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((imgs_val, lbls_val))

    # Slice the dataset
    train_dataset = train_dataset.shuffle(100).batch(1)
    val_dataset = val_dataset.batch(1)

    show_predictions(-1)
    # imgs_train = imgs_train.reshape((79, num_pixels, 3))
    # imgs_val = imgs_val.reshape((10, num_pixels, 3))
    # print(sample_weight.shape)
    # print(lbls_train_onehot.shape)
    # print(imgs_train.shape)
    # print(lbls_val_onehot.shape)
    # print(imgs_val.shape)
    #model_history = unet.fit(train_dataset, epochs=epoch)

    print("imgs_val: " + str(imgs_val.shape))
    print("lbls_display: " + str(lbls_val.shape))

    model_history = unet.fit(imgs_train, lbls_train_onehot, validation_data=[imgs_val, lbls_val_onehot],
                             batch_size = 1,
                             epochs=epoch,
                             verbose=2,
                             shuffle=True,
                             callbacks=[DisplayCallback()])
                             #sample_weight=sample_weight)

    show_predictions(101, 2)
    show_predictions(102, 3)
    show_predictions(103, 4)
    show_predictions(104, 5)
    show_predictions(105, 6)

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    accuracy = model_history.history['accuracy']
    val_accuracy = model_history.history['val_accuracy']
    iou_metric = model_history.history['iou_coef']
    val_iou_metric = model_history.history['val_iou_coef']
    dice_metric = model_history.history['dice_coef']
    val_dice_coef = model_history.history['val_dice_coef']

    # Save metric data to file
    f = open("Pictures/Metrics.txt", "w+")
    f.write("loss" + str(loss))
    f.write("\nval_loss: " + str(val_loss))
    f.write("\naccuracy: " + str(accuracy))
    f.write("\nval_accuracy: " + str(val_accuracy))
    f.write("\niou_coef: " + str(iou_metric))
    f.write("\nval_iou_coef: " + str(val_iou_metric))
    f.write("\ndice_coef: " + str(dice_metric))
    f.write("\nval_dice_coef: " + str(val_dice_coef))
    f.close()

    epochs = range(epoch)

    # Plot statistics
    graph = plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig('Pictures/Training and Validation Loss')
    plt.show()
    plt.close(graph)
    '''
    # Evaluate model
    print('\n# Evaluate on test data')
    start_time = time.time()
    results = unet.evaluate(imgs_test, lbls_test_onehot, batch_size=1)
    stop_time = time.time()
    print("--- %s seconds ---" % (stop_time - start_time))
    print("%s: %.2f%%" % (unet.metrics_names[1], results[1] * 100))
    '''
    # Save model to file
    unet.save('Unet_model.h5')
    print("Saved model to disk")

elif not train:
    # Load model from file
    unet = load_model('Unet_model.h5', compile=False)

    # Compile loaded model
    if Loss_function == 1:
        print('Categorical Focal Loss with gamma = ' + str(FL_gamma) + ' and alpha = ' + str(FL_alpha))
        unet.compile(optimizer='adam',
                     loss=categorical_focal_loss(gamma=FL_gamma, alpha=FL_alpha),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 2:
        print('Dice Loss')
        unet.compile(optimizer='adam',
                     loss=dice_loss(),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 3:
        print('Jaccard Loss')
        unet.compile(optimizer='adam',
                     loss=jaccard_loss(),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 4:
        print('Tversky Loss with beta = ' + str(TL_beta))
        unet.compile(optimizer='adam',
                     loss=tversky_loss(beta=TL_beta),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 5:
        print('Weighted categorical crossentropy with weights = ' + str(weights))
        unet.compile(optimizer='adam',
                     loss=weighted_categorical_crossentropy(weights),
                     metrics=['accuracy', iou_coef, dice_coef])
    elif Loss_function == 6:
        print('Categorical crossentropy')
        unet.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy', iou_coef, dice_coef])
    else:
        print('No loss function')


# Evaluate loaded model
print('\n# Evaluate on test data')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test_onehot, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f" % (unet.metrics_names[0], results[0]))
print("%s: %.2f" % (unet.metrics_names[1], results[1]))
print("%s: %.2f" % (unet.metrics_names[2], results[2]))
print("%s: %.2f" % (unet.metrics_names[3], results[3]))

print('\n# Evaluate on test data 2')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test_onehot, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (unet.metrics_names[1], results[1] * 100))

print('\n# Evaluate on test data 3')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test_onehot, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (unet.metrics_names[1], results[1] * 100))

print('\n# Evaluate on test data 4')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test_onehot, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (unet.metrics_names[1], results[1] * 100))

print('\n# Evaluate on test data 5')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test_onehot, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (unet.metrics_names[1], results[1] * 100))
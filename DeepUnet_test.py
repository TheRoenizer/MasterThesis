import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from contextlib import redirect_stdout

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Hvis du vil bruge "kort 1":
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ellers:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# hvis du træne på CPU'en:
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from DeepUnet import *
from functions import *

train = True
which_data = 1 # 1 = jigsaw, 2 = EndoVis
which_path = 1 # 1 = local, 2 = remote
batch_size = 1
num_epochs = 20
#num_pixels = 480 * 640

if which_data == 1:
    weights = [.5, 1.5, 1.5, 1, 1]  # [background, gripper, gripper, shaft, shaft]
if which_data == 2:
    weights = [.5, 2, 2, 2]  # [background, shaft, wrist, fingers]

if which_path == 1:
    # Christoffer:
    PATH = 'C:/Users/chris/Google Drive/'
elif which_path == 2:
    # Linux:
    PATH = '/home/jsteeen/'


# functions used to display images after each epoch
def display(display_list, epoch_display):
    fig = plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask after epoch {}'.format(epoch_display + 1)]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig("pictures_deepunet/afterEpoch{}.png".format(epoch_display + 1))
    # plt.show()
    plt.close(fig)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(epoch_show_predictions, image_num=1):
    pred_mask = deep_unet.predict(imgs_val[image_num][tf.newaxis, ...]) * 255
    display([imgs_val[image_num], lbls_display_val[image_num], create_mask(pred_mask)], epoch_show_predictions)


class DisplayCallback(tf.keras.callbacks.Callback):
    # @staticmethod
    def on_epoch_end(self, epoch_callback, logs=None):
        clear_output(wait=True)
        show_predictions(epoch_callback)
        print('\nSample Prediction after epoch {}\n'.format(epoch_callback + 1))


# Load images and labels
if which_data == 1:
    images, labels, labels_display = load_data(PATH)

    cv.imwrite("pictures_deepunet_j/labels_display0.png", labels_display[0])
    cv.imwrite("pictures_deepunet_j/label0.png", labels[0,...,0])
    cv.imwrite("pictures_deepunet_j/label1.png", labels[0,...,1])
    cv.imwrite("pictures_deepunet_j/label2.png", labels[0,...,2])
    cv.imwrite("pictures_deepunet_j/label3.png", labels[0,...,3])
    cv.imwrite("pictures_deepunet_j/label4.png", labels[0,...,4])
    print("images saved")

    imgs_train = images[0:79]
    imgs_val = images[79:89]
    imgs_test = images[89:99]

    lbls_train = labels[0:79]
    lbls_val = labels[79:89]
    lbls_test = labels[89:99]

    lbls_display_train = labels_display[0:79]
    lbls_display_val = labels_display[79:89]
    lbls_display_test = labels_display[89:99]

if which_data == 2:
    images, labels, labels_display = load_data_EndoVis17(PATH)

    imgs_train = images[0:175]
    imgs_val = images[175:200]
    imgs_test = images[200:225]
    print("imgs_train: " + str(imgs_train.shape))
    print("imgs_val: " + str(imgs_val.shape))

    lbls_train = labels[0:175]
    lbls_val = labels[175:200]
    lbsl_test = labels[200:225]

    lbls_display_train = labels_display[0:175]
    lbls_display_val = labels_display[175:200]
    lbls_display_test = labels_display[200:225]

print('Images and labels loaded!')

if train:
    imgs_train2 = np.zeros((480, 640, 3))
    imgs_train3 = np.zeros((1024, 1280, 3))

    if which_data == 1:
        (deep_unet, name) = deep_unet(imgs_train2.shape, num_classes=5, padding=16)
    elif which_data == 2:
        (deep_unet, name) = deep_unet(imgs_train3.shape, num_classes=4)

    deep_unet.summary()

    with open('DeepUnetModelSummary.txt', 'w') as f:
        with redirect_stdout(f):
            deep_unet.summary()

    deep_unet.compile(optimizer='adam', loss=weighted_categorical_crossentropy(weights), metrics=['accuracy', jaccard, iou_coef_mean, iou_coef, dice_coef])

    tf.keras.utils.plot_model(deep_unet,
                              to_file='DeepUnetModelPlot.png',
                              show_shapes=True,
                              show_layer_names=True,
                              rankdir='TB')

    show_predictions(-1)

    model_history = deep_unet.fit(imgs_train, lbls_train, validation_data=[imgs_val, lbls_val],
                                  batch_size=batch_size,
                                  epochs=num_epochs,
                                  verbose=1,
                                  shuffle=True,
                                  callbacks=[DisplayCallback()])

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    accuracy = model_history.history['accuracy']
    val_accuracy = model_history.history['val_accuracy']
    iou_metric = model_history.history['iou_coef']
    val_iou_metric = model_history.history['val_iou_coef']
    dice_metric = model_history.history['dice_coef']
    val_dice_coef = model_history.history['val_dice_coef']

    # Save metric data to file
    f = open("pictures_deepunet/Metrics.txt", "w+")
    f.write("loss" + str(loss))
    f.write("\nval_loss: " + str(val_loss))
    f.write("\naccuracy: " + str(accuracy))
    f.write("\nval_accuracy: " + str(val_accuracy))
    f.write("\niou_coef: " + str(iou_metric))
    f.write("\nval_iou_coef: " + str(val_iou_metric))
    f.write("\ndice_coef: " + str(dice_metric))
    f.write("\nval_dice_coef: " + str(val_dice_coef))
    f.write("\nweights: " + str(weights))
    f.close()

    epochs = range(num_epochs)

    # Plot statistics
    graph = plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig('pictures_deepunet/Training and Validation Loss')
    plt.show()
    plt.close(graph)

    deep_unet.save('deep_unet_model.h5')
    print("Saved model to disk")

elif not train:
    # Load model from file
    deep_unet = load_model('deep_unet_model.h5', compile=False)

    #compile saved model
    deep_unet.compile(optimizer='adam', loss=weighted_categorical_crossentropy(weights), metrics=['accuracy', iou_coef_mean, iou_coef, dice_coef])
'''
#evaluate model
print('\n# Predict on test data 1')
start_time = time.time()
results = deep_unet.predict(imgs_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print(results.shape)
fps = 1 / ((stop_time - start_time) / 10)
print("fps: %s" % fps)

predicted_labels = np.empty((10, 480, 640, 1), dtype=np.float32)
for i in range(10):
    for j in range(5):
        mask = cv.threshold(results[i,...,j], dst=None, thresh=0.5, maxval=255, type=cv.THRESH_BINARY)[1]
        k = np.where(mask == 255)
        predicted_labels[i][k] = (j + 1) * 50  # set pixel value here

cv.imwrite("Pictures_DeepUnet/results/predicted.png", predicted_labels[0])
cv.imwrite("Pictures_DeepUnet/results/result0_0.png", results[0,...,0])
cv.imwrite("Pictures_DeepUnet/results/result0_1.png", results[0,...,1])
cv.imwrite("Pictures_DeepUnet/results/result0_2.png", results[0,...,2])
cv.imwrite("Pictures_DeepUnet/results/result0_3.png", results[0,...,3])
cv.imwrite("Pictures_DeepUnet/results/result0_4.png", results[0,...,4])


print('\n# Predict on test data 2')
start_time = time.time()
results = deep_unet.predict(imgs_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds---" % (stop_time - start_time))
fps = 1 / ((stop_time - start_time) / 10)
print("fps: %s" % fps)

print('\n# Predict on test data 3')
start_time = time.time()
results = deep_unet.predict(imgs_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
fps = 1 / ((stop_time - start_time) / 10)
print("fps: %s" % fps)

print('\n# Predict on test data 4')
start_time = time.time()
results = deep_unet.predict(imgs_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
fps = 1 / ((stop_time - start_time) / 10)
print("fps: %s" % fps)
'''
'''
print('\n# Evaluate on test data')
start_time = time.time()
results = deep_unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f" % (deep_unet.metrics_names[0], results[0]))
print("%s: %.2f" % (deep_unet.metrics_names[1], results[1]))
print("%s: %.2f" % (deep_unet.metrics_names[2], results[2]))
print("%s: %.2f" % (deep_unet.metrics_names[3], results[3]))

print('\n# Evaluate on test data 2')
start_time = time.time()
results = deep_unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (deep_unet.metrics_names[1], results[1] * 100))


print('\n# Evaluate on test data')
start_time = time.time()
results = deep_unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f" % (deep_unet.metrics_names[0], results[0]))
print("%s: %.2f" % (deep_unet.metrics_names[1], results[1]))
print("%s: %.2f" % (deep_unet.metrics_names[2], results[2]))
print("%s: %.2f" % (deep_unet.metrics_names[3], results[3]))

print('\n# Evaluate on test data 2')
start_time = time.time()
results = deep_unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (deep_unet.metrics_names[1], results[1] * 100))
'''

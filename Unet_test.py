from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
from contextlib import redirect_stdout
import time

from Unet import *
from functions import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Hvis du vil bruge "kort 1":
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ellers:
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# hvis du træne på CPU'en:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


print('TensorFlow version: '+tf.__version__)
print('Numpy version: '+np.__version__)
print('Keras version: '+tf.keras.__version__)

which_path = 3  # 1 = local c, 2 = local j, 3 = remote
train = True
epoch = 100
num_pixels = 480 * 640
weights = [.5, 1.5, 1, 1.5, 1]  # [background, right gripper, right shaft, left gripper, left shaft]

if which_path == 1:
    # Christoffer:
    PATH = 'C:/Users/chris/Google Drive/'
elif which_path == 2:
    # Jonathan
    PATH = '/Users/jonathansteen/Google Drive/'
else:
    # Linux:
    PATH = '/home/jsteeen/'

metrics = ['accuracy',
           iou_coef_mean, iou_coef0, iou_coef1, iou_coef2, iou_coef3, iou_coef4,
           dice_coef_mean, dice_coef0, dice_coef1, dice_coef2, dice_coef3, dice_coef4]

Loss_function = 5   # 1=focal_loss, 2=dice_loss, 3=jaccard_loss, 4=tversky_loss, 5=weighted_categorical_crossentropy 6=categorical_cross_entropy

FL_alpha = .25      # Focal loss alpha
FL_gamma = 2.       # Focal loss gamma
TL_beta = 3         # Tversky loss beta

if Loss_function == 1:
    loss_function = categorical_focal_loss(gamma=FL_gamma, alpha=FL_alpha)
elif Loss_function == 2:
    loss_function = dice_loss()
elif Loss_function == 3:
    loss_function = jaccard_loss()
elif Loss_function == 4:
    loss_function = tversky_loss(beta=TL_beta)
elif Loss_function == 5:
    loss_function = weighted_categorical_crossentropy(weights)
else:
    loss_function = 'categorical_crossentropy'


# Functions used to display images after each epoch
def display(display_list, epoch_display):
    fig = plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask after epoch {}'.format(epoch_display + 1)]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.savefig("pictures_unet/afterEpoch{}.png".format(epoch_display + 1))
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


es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = tf.keras.callbacks.ModelCheckpoint('best_model_unet.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)


# Load images and labels
print('Loading images and labels...')
images, labels, labels_display = load_data(PATH)

imgs_train = images[0:79]
imgs_val = images[79:89]
imgs_test = images[89:99]

lbls_train = labels[0:79]
lbls_val = labels[79:89]
lbls_test = labels[89:99]

lbls_display_train = labels_display[0:79]
lbls_display_val = labels_display[79:89]
lbls_display_test = labels_display[89:99]

print('Images and labels loaded!')

if train:
    input_shape = np.empty((480, 640, 3))
    (unet, name) = unet(input_shape.shape, num_classes=5, droprate=0.0, linear=False)

    unet.summary()

    with open('UnetModelSummary.txt', 'w') as f:
        with redirect_stdout(f):
            unet.summary()

    unet.compile(optimizer='adam', loss=loss_function, metrics=metrics)

    tf.keras.utils.plot_model(unet,
                              to_file='UnetModelPlot.png',
                              show_shapes=True,
                              show_layer_names=True,
                              rankdir='TB')

    model_history = unet.fit(imgs_train, lbls_train, validation_data=[imgs_val, lbls_val],
                             batch_size=1,
                             epochs=epoch,
                             verbose=2,
                             shuffle=True,
                             callbacks=[DisplayCallback(), es, mc])

    '''
    show_predictions(101, 2)
    show_predictions(102, 3)
    show_predictions(103, 4)
    show_predictions(104, 5)
    show_predictions(105, 6)
    '''
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    accuracy = model_history.history['accuracy']
    val_accuracy = model_history.history['val_accuracy']
    #iou_metric = model_history.history['iou_coef']
    #val_iou_metric = model_history.history['val_iou_coef']
    #dice_metric = model_history.history['dice_coef']
    #val_dice_coef = model_history.history['val_dice_coef']

    # Save metric data to file
    f = open("pictures_unet/Metrics.csv", "w+")
    f.write("loss" + str(loss))
    f.write("\nval_loss: " + str(val_loss))
    f.write("\naccuracy: " + str(accuracy))
    f.write("\nval_accuracy: " + str(val_accuracy))
    #f.write("\niou_coef: " + str(iou_metric))
    #f.write("\nval_iou_coef: " + str(val_iou_metric))
    #f.write("\ndice_coef: " + str(dice_metric))
    #f.write("\nval_dice_coef: " + str(val_dice_coef))
    #f.write("\nweights: " + str(weights))
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
    plt.savefig('pictures_unet/Training and Validation Loss')
    plt.show()
    plt.close(graph)

    # Save model to file
    unet.save('Unet_model.h5')
    print("Saved model to disk")

else:
    # Load model from file
    unet = load_model('Unet_model.h5', compile=False)

    # Compile loaded model
    unet.compile(optimizer='adam', loss=loss_function, metrics=metrics)


# Evaluate model
#display test images
def display_test(display_list, image_num):
    fig = plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.savefig("pictures_unet/test_image{}.png".format(image_num + 1))
    plt.close(fig)

def show_predictions_test(image_num=1):
    pred_mask = unet.predict(imgs_test[image_num][tf.newaxis, ...]) * 255
    display_test([imgs_test[image_num], lbls_display_test[image_num], create_mask(pred_mask)], image_num)


for i in range(10):
    show_predictions_test(i)


print('\n# Evaluate on test data')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f" % (unet.metrics_names[0], results[0]))
print("%s: %.2f" % (unet.metrics_names[1], results[1]))
print("%s: %.2f" % (unet.metrics_names[2], results[2]))
print("%s: %.2f" % (unet.metrics_names[3], results[3]))



print('\n# predict on test data 2')
start_time = time.time()
results = unet.predict(imgs_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (unet.metrics_names[1], results[1] * 100))

for i in range(10):
    cv.imwrite('pictures_unet/test_image_{}'.format(i), results[i])

print('\n# Evaluate on test data 3')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (unet.metrics_names[1], results[1] * 100))

print('\n# Evaluate on test data 4')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (unet.metrics_names[1], results[1] * 100))

print('\n# Evaluate on test data 5')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))
print("%s: %.2f%%" % (unet.metrics_names[1], results[1] * 100))

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
which_data = 2  # 1 = JIGSAWS, 2 = MICCAI2017
model_name = 'best_model_unet_fl_endo.hdf5'
train = False
epoch = 100

if which_data == 1:
    weights = [.5, 1.5, 1, 1.5, 1]  # [background, right gripper, right shaft, left gripper, left shaft]
    metrics = ['accuracy',
               iou_coef_mean, iou_coef0, iou_coef1, iou_coef2, iou_coef3, iou_coef4]
if which_data == 2:
    weights = [.5, 2, 2, 2]  # [background, shaft, wrist, fingers]
    metrics = ['accuracy',
               iou_coef_mean, iou_coef0, iou_coef1, iou_coef2, iou_coef3]

if which_path == 1:
    # Christoffer:
    PATH = 'C:/Users/chris/Google Drive/'
elif which_path == 2:
    # Jonathan
    PATH = '/Users/jonathansteen/Google Drive/'
else:
    # Linux:
    PATH = '/home/jsteeen/'

Loss_function = 1   # 1=focal_loss, 2=weighted_categorical_crossentropy 3=categorical_cross_entropy

FL_alpha = .25      # Focal loss alpha
FL_gamma = 2.       # Focal loss gamma


if Loss_function == 1:
    loss_function = categorical_focal_loss(gamma=FL_gamma, alpha=FL_alpha)
elif Loss_function == 2:
    loss_function = weighted_categorical_crossentropy(weights)
else:
    loss_function = 'categorical_crossentropy'


# Functions used to display images after each epoch
def display(display_list, epoch_display):
    fig = plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask after epoch {}'.format(epoch_display)]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.savefig("pictures_unet/afterEpoch{}.png".format(epoch_display))
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


# Callback functions
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
csv = tf.keras.callbacks.CSVLogger('pictures_unet/metrics.csv', separator=',', append=False)


# Load images and labels
if which_data == 1:
    print('Loading images and labels from JIGGSAWS...')
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

elif which_data == 2:
    print('Loading images and labels from EndoVis17...')
    images, labels, labels_display = load_data_EndoVis17(PATH)

    imgs_train = images[0:175]
    imgs_val = images[175:200]
    imgs_test = images[200:225]

    lbls_train = labels[0:175]
    lbls_val = labels[175:200]
    lbls_test = labels[200:225]

    lbls_display_train = labels_display[0:175]
    lbls_display_val = labels_display[175:200]
    lbls_display_test = labels_display[200:225]

print('Images and labels loaded!')

if train:
    # Build model
    input_shape_jigsaw = np.empty((480, 640, 3))
    input_shape_endovis = np.zeros((512, 640, 3))  # old values 1024,1280
    if which_data == 1:
        (unet, name) = unet(input_shape_jigsaw.shape, num_classes=5, droprate=0.0, linear=False)
    elif which_data == 2:
        (unet, name) = unet(input_shape_endovis.shape, num_classes=4, droprate=0.0, linear=False)

    unet.summary()

    with open('UnetModelSummary.txt', 'w') as f:
        with redirect_stdout(f):
            unet.summary()

    # Compile model
    unet.compile(optimizer='adam', loss=loss_function, metrics=metrics)

    tf.keras.utils.plot_model(unet,
                              to_file='UnetModelPlot.png',
                              show_shapes=True,
                              show_layer_names=True,
                              rankdir='TB')

    # Train model
    model_history = unet.fit(imgs_train, lbls_train, validation_data=[imgs_val, lbls_val],
                             batch_size=1,
                             epochs=epoch,
                             verbose=2,
                             shuffle=True,
                             callbacks=[DisplayCallback(), es, mc, csv])

# Load best model from file
unet = load_model(model_name, compile=False)

# Compile loaded model
unet.compile(optimizer='adam', loss=loss_function, metrics=metrics)


# Evaluate model
# display test images
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


for i in range(len(imgs_test)):
    show_predictions_test(i)


print('\n# Evaluate on test data')
start_time = time.time()
results = unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % ((stop_time - start_time)/len(imgs_test)))

# Save metric data to file
f = open("pictures_unet/test_metrics.txt", "w+")
f.write("%s: %.4f" % (unet.metrics_names[0], results[0]))
for i in range(1, len(results)):
    f.write("\n%s: %.4f" % (unet.metrics_names[i], results[i]))


print('\n# Evaluate on test data')
start_time = time.time()
unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % ((stop_time - start_time)/len(imgs_test)))

print(str(len(imgs_test)))

# Evaluate time and save to file
times = 10
total_time = 0.0
for i in range(times):
    print('\n# predict on test data ')
    start_time = time.time()
    unet.predict(imgs_test, batch_size=1)
    stop_time = time.time()
    total_time += ((stop_time - start_time) / len(imgs_test))
    f.write("\nSeconds per image: %.4f" % ((stop_time - start_time)/len(imgs_test)))
    print("--- %s seconds ---" % ((stop_time - start_time)/len(imgs_test)))

average = total_time / times
fps = 1.0 / average

f.write("\nAverage: %.4f" % average)
f.write("\nFPS: %.2f" % fps)
if Loss_function == 2:
    f.write("\nWeights: %s" % weights)
f.close()

print('DONE!')

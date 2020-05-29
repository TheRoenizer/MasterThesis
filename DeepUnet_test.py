import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from contextlib import redirect_stdout

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from DeepUnet import *
from functions import *

model_name = 'best_model_deepunet_wcc_endo.hdf5'
train = True
which_data = 2  # 1 = jigsaw, 2 = EndoVis
which_path = 2  # 1 = local, 2 = remote
batch_size = 1
num_epochs = 100

Loss_function = 2  # 1=focal_loss, 2=weighted_categorical_crossentropy, 3=categorical_cross_entropy

FL_alpha = .25      # Focal loss alpha
FL_gamma = 2.       # Focal loss gamma

if which_data == 1:
    weights = [.5, 3, 2, 3, 2]  # [background, right gripper, right shaft, left gripper, left shaft]
    metrics = ['accuracy',
               iou_coef_mean, iou_coef0, iou_coef1, iou_coef2, iou_coef3, iou_coef4,
               dice_coef_mean, dice_coef0, dice_coef1, dice_coef2, dice_coef3, dice_coef4]
if which_data == 2:
    weights = [.5, 2, 2, 2]  # [background, shaft, wrist, fingers]
    metrics = ['accuracy',
               iou_coef_mean, iou_coef0, iou_coef1, iou_coef2, iou_coef3,
               dice_coef_mean, dice_coef0, dice_coef1, dice_coef2, dice_coef3]

if which_path == 1:
    # Christoffer:
    PATH = 'C:/Users/chris/Google Drive/'
elif which_path == 2:
    # Linux:
    PATH = '/home/jsteeen/'



if Loss_function == 1:
    loss_function = categorical_focal_loss(gamma=FL_gamma, alpha=FL_alpha)
elif Loss_function == 2:
    loss_function = weighted_categorical_crossentropy(weights)
else:
    loss_function = 'categorical_crossentropy'

# functions used to display images after each epoch
def display(display_list, epoch_display):
    fig = plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask after epoch {}'.format(epoch_display)]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig("pictures_deepunet/afterEpoch{}.png".format(epoch_display))
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

# Callback functions
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
csv = tf.keras.callbacks.CSVLogger('pictures_deepunet/metrics.csv', separator=',', append=False)


# Load images and labels
if which_data == 1:
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

if which_data == 2:
    print('Loading images and labels...')
    images, labels, labels_display = load_data_EndoVis17_full(PATH)

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
    input_shape_jigsaws = np.zeros((480, 640, 3))
    input_shape_endovis = np.zeros((1024, 1280, 3))

    if which_data == 1:
        (deep_unet, name) = deep_unet(input_shape_jigsaws.shape, num_classes=5, padding=16)
    else:
        (deep_unet, name) = deep_unet(input_shape_endovis.shape, num_classes=4)

    deep_unet.summary()

    with open('DeepUnetModelSummary.txt', 'w') as f:
        with redirect_stdout(f):
            deep_unet.summary()

    deep_unet.compile(optimizer='adam', loss=loss_function, metrics=metrics)

    tf.keras.utils.plot_model(deep_unet,
                              to_file='DeepUnetModelPlot.png',
                              show_shapes=True,
                              show_layer_names=True,
                              rankdir='TB')


    model_history = deep_unet.fit(imgs_train, lbls_train, validation_data=[imgs_val, lbls_val],
                                  batch_size=batch_size,
                                  epochs=num_epochs,
                                  verbose=2,
                                  shuffle=True,
                                  callbacks=[DisplayCallback(), es, mc, csv])


# Load model from file
deep_unet = load_model(model_name, compile=False)

#compile saved model
deep_unet.compile(optimizer='adam', loss=loss_function, metrics=metrics)

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

    plt.savefig("pictures_deepunet/test_image{}.png".format(image_num + 1))
    plt.close(fig)


def show_predictions_test(image_num=1):
    pred_mask = deep_unet.predict(imgs_test[image_num][tf.newaxis, ...]) * 255
    display_test([imgs_test[image_num], lbls_display_test[image_num], create_mask(pred_mask)], image_num)


for i in range(10):
    show_predictions_test(i)


print('\n# Evaluate on test data')
start_time = time.time()
results = deep_unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % ((stop_time - start_time)/len(imgs_test)))

# Save metric data to file
f = open("pictures_deepunet/test_metrics.txt", "w+")
f.write("%s: %.4f" % (deep_unet.metrics_names[0], results[0]))
for i in range(1, len(results)):
    f.write("\n%s: %.4f" % (deep_unet.metrics_names[i], results[i]))


print('\n# Evaluate on test data')
start_time = time.time()
deep_unet.evaluate(imgs_test, lbls_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % ((stop_time - start_time)/len(imgs_test)))

print(str(len(imgs_test)))

times = 10
total_time = 0.0
for i in range(times):
    print('\n# predict on test data ')
    start_time = time.time()
    deep_unet.predict(imgs_test, batch_size=1)
    stop_time = time.time()
    total_time += ((stop_time - start_time) / len(imgs_test))
    f.write("\nSeconds per image: %.4f" % ((stop_time - start_time)/len(imgs_test)))
    print("--- %s seconds ---" % ((stop_time - start_time)/len(imgs_test)))

average = total_time / times
fps = 1.0 / average

f.write("\nAverage: %.4f" % average)
f.write("\nFPS: %.2f" % fps)
if Loss_function == 2:
    f.write("\nWeigts: %s" % weights)
f.close()

print('DONE!')
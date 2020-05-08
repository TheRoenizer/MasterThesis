import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

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
which_data = 2 # 1 = jigsaw, 2 = EndoVis
which_path = 2 # 1 = local, 2 = remote
batch_size = 1
num_epochs = 100
#num_pixels = 480 * 640

if which_data == 1:
    weights = [.5, 1.5, 1.5, 1, 1] # [background, gripper, gripper, shaft, shaft]
if which_data == 2:
    weights = [.5, 1.5, 1] # [background, gripper, shaft]

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
    images, labels, labels_display = load_data_EndoVis(PATH)
    cv.imwrite("pictures_deepunet/labels_display0.png", labels_display[50])
    cv.imwrite("pictures_deepunet/label0.png", labels[50, ..., 0])
    cv.imwrite("pictures_deepunet/label1.png", labels[50, ..., 1])
    cv.imwrite("pictures_deepunet/label2.png", labels[50, ..., 2])
    print("images saved")

    imgs_train = np.concatenate((images[0:35], images[40:75], images[80:115]))
    imgs_val = np.concatenate((images[35:40], images[75:80], images[115:120]))
    print("imgs_train: " + str(imgs_train.shape))
    print("imgs_val: " + str(imgs_val.shape))

    lbls_train = np.concatenate((labels[0:35], labels[40:75], labels[80:115]))
    lbls_val = np.concatenate((labels[35:40], labels[75:80], labels[115:120]))

    lbls_display_train = np.concatenate((labels_display[0:35], labels_display[40:75], labels_display[80:115]))
    lbls_display_val = np.concatenate((labels_display[35:40], labels_display[75:80], labels_display[115:120]))

'''
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
'''
print('Images and labels loaded!')

if train:
    imgs_train2 = np.zeros((480, 640, 3))

    if which_data == 1:
        (deep_unet, name) = deep_unet(imgs_train2.shape, num_classes=5, droprate=0.0, linear=False)
    elif which_data == 2:
        (deep_unet, name) = deep_unet(imgs_train2.shape, num_classes=3, droprate=0.0, linear=False)

    deep_unet.summary()

    deep_unet.compile(optimizer='adam', loss=weighted_categorical_crossentropy(weights), metrics=['accuracy', iou_coef, dice_coef])

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
    deep_unet.compile(optimizer='adam', loss=weighted_categorical_crossentropy(weights), metrics=['accuracy', iou_coef, dice_coef])
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
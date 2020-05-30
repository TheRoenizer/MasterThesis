from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv
import os
from contextlib import redirect_stdout
import time

try:
    from keras.layers import Input, Flatten, Dense, add, Reshape, Conv2D, Conv1D, BatchNormalization, MaxPooling2D, Dropout, Concatenate
    from keras.models import Model, Sequential, load_model
except:
    from tensorflow.keras.layers import Input, Flatten, Dense, add, Reshape, Conv2D, BatchNormalization, MaxPooling2D
    from tensorflow.keras.models import Model, load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

which_path = 0
epochs = 10
droprate = 0.5

if which_path == 1:
    # Christoffer:
    PATH = 'C:/Users/chris/Google Drive/Master Thesis/'
elif which_path == 2:
    # Jonathan:
    PATH = '/Users/jonathansteen/Google Drive/Master Thesis/'
else:
    # Linux:
    PATH = '/home/jsteeen/'

# Load images
print("Loading images...")
# Train images
imgs_train_left = np.zeros((64, 800, 1280, 3))
imgs_train_right = np.zeros((64, 800, 1280, 3))
imgs_train = np.empty((64, 1600, 1280, 3))
for i in range(0, 40):
    path_left = PATH + 'rosbag_annotations/img'+str(i)+'_left/data/002.png'
    path_right = PATH + 'rosbag_annotations/img'+str(i)+'_right/data/002.png'

    img_left = np.array(Image.open(path_left))[np.newaxis]
    img_right = np.array(Image.open(path_right))[np.newaxis]

    # Normalize data
    img_left = img_left / 255
    img_right = img_right / 255

    imgs_train_left[i] = img_left
    imgs_train_right[i] = img_right
    imgs_train[i] = np.concatenate((img_left, img_right), axis=1)
for i in range(50, 74):
    path_left = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/002.png'
    path_right = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/002.png'

    img_left = np.array(Image.open(path_left))[np.newaxis]
    img_right = np.array(Image.open(path_right))[np.newaxis]

    # Normalize data
    img_left = img_left / 255
    img_right = img_right / 255

    imgs_train_left[i-10] = img_left
    imgs_train_right[i-10] = img_right
    imgs_train[i-10] = np.concatenate((img_left, img_right), axis=1)

# Validation images
imgs_val_left = np.zeros((8, 800, 1280, 3))
imgs_val_right = np.zeros((8, 800, 1280, 3))
imgs_val = np.empty((8, 1600, 1280, 3))
for i in range(40, 45):
    path_left = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/002.png'
    path_right = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/002.png'

    img_left = np.array(Image.open(path_left))[np.newaxis]
    img_right = np.array(Image.open(path_right))[np.newaxis]

    # Normalize data
    img_left = img_left / 255
    img_right = img_right / 255

    imgs_val_left[i-40] = img_left
    imgs_val_right[i-40] = img_right
    imgs_val[i-40] = np.concatenate((img_left, img_right), axis=1)
for i in range(74, 77):
    path_left = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/002.png'
    path_right = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/002.png'

    img_left = np.array(Image.open(path_left))[np.newaxis]
    img_right = np.array(Image.open(path_right))[np.newaxis]

    # Normalize data
    img_left = img_left / 255
    img_right = img_right / 255

    imgs_val_left[i-69] = img_left
    imgs_val_right[i-69] = img_right
    imgs_val[i-69] = np.concatenate((img_left, img_right), axis=1)

# Test images
imgs_test_left = np.zeros((8, 800, 1280, 3))
imgs_test_right = np.zeros((8, 800, 1280, 3))
imgs_test = np.empty((8, 1600, 1280, 3))
for i in range(45, 50):
    path_left = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/002.png'
    path_right = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/002.png'

    img_left = np.array(Image.open(path_left))[np.newaxis]
    img_right = np.array(Image.open(path_right))[np.newaxis]

    # Normalize data
    img_left = img_left / 255
    img_right = img_right / 255

    imgs_test_left[i-45] = img_left
    imgs_test_right[i-45] = img_right
    imgs_test[i-45] = np.concatenate((img_left, img_right), axis=1)
for i in range(77, 80):
    path_left = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/002.png'
    path_right = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/002.png'

    img_left = np.array(Image.open(path_left))[np.newaxis]
    img_right = np.array(Image.open(path_right))[np.newaxis]

    # Normalize data
    img_left = img_left / 255
    img_right = img_right / 255

    imgs_test_left[i-72] = img_left
    imgs_test_right[i-72] = img_right
    imgs_test[i-72] = np.concatenate((img_left, img_right), axis=1)

print("Images loaded!")

# Load labels
print("Loading labels...")
# Train labels
lbls_train_left = np.zeros((64, 800, 1280))
lbls_train_right = np.zeros((64, 800, 1280))
lbls_train = np.empty((64, 1600, 1280, 1))
for i in range(0, 40):
    # Left
    path_left1 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/000.png'
    path_left2 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/001.png'

    lbl_left1 = cv.imread(path_left1, 2)
    lbl_left2 = cv.imread(path_left2, 2)

    change_left1_to = np.where(lbl_left1[:, :] != 0)
    change_left2_to = np.where(lbl_left2[:, :] != 0)

    lbl_left1[change_left1_to] = 1
    lbl_left2[change_left2_to] = 2

    lbl_left = lbl_left1 + lbl_left2

    change_left_overlap = np.where(lbl_left[:, :] == 3)
    lbl_left[change_left_overlap] = 0
    lbls_train_left[i] = lbl_left

    # Right
    path_right1 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/000.png'
    path_right2 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/001.png'

    lbl_right1 = cv.imread(path_right1, 2)
    lbl_right2 = cv.imread(path_right2, 2)

    change_right1_to = np.where(lbl_right1[:, :] != 0)
    change_right2_to = np.where(lbl_right2[:, :] != 0)

    lbl_right1[change_right1_to] = 1
    lbl_right2[change_right2_to] = 2

    lbl_right = lbl_right1 + lbl_right2

    change_right_overlap = np.where(lbl_right[:, :] == 3)
    lbl_right[change_right_overlap] = 0
    lbls_train_right[i] = lbl_right
    lbls_train[i] = np.concatenate((lbl_left[:, :, np.newaxis], lbl_right[:, :, np.newaxis]), axis=0)
for i in range(50, 74):
    # Left
    path_left1 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/000.png'
    path_left2 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/001.png'

    lbl_left1 = cv.imread(path_left1, 2)
    lbl_left2 = cv.imread(path_left2, 2)

    change_left1_to = np.where(lbl_left1[:, :] != 0)
    change_left2_to = np.where(lbl_left2[:, :] != 0)

    lbl_left1[change_left1_to] = 1
    lbl_left2[change_left2_to] = 2

    lbl_left = lbl_left1 + lbl_left2

    change_left_overlap = np.where(lbl_left[:, :] == 3)
    lbl_left[change_left_overlap] = 0
    lbls_train_left[i-10] = lbl_left

    # Right
    path_right1 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/000.png'
    path_right2 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/001.png'

    lbl_right1 = cv.imread(path_right1, 2)
    lbl_right2 = cv.imread(path_right2, 2)

    change_right1_to = np.where(lbl_right1[:, :] != 0)
    change_right2_to = np.where(lbl_right2[:, :] != 0)

    lbl_right1[change_right1_to] = 1
    lbl_right2[change_right2_to] = 2

    lbl_right = lbl_right1 + lbl_right2

    change_right_overlap = np.where(lbl_right[:, :] == 3)
    lbl_right[change_right_overlap] = 0
    lbls_train_right[i-10] = lbl_right
    lbls_train[i-10] = np.concatenate((lbl_left[:, :, np.newaxis], lbl_right[:, :, np.newaxis]), axis=0)


# Validation labels
lbls_val_left = np.zeros((8, 800, 1280))
lbls_val_right = np.zeros((8, 800, 1280))
lbls_val = np.empty((8, 1600, 1280, 1))
for i in range(40, 45):
    # Left
    path_left1 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/000.png'
    path_left2 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/001.png'

    lbl_left1 = cv.imread(path_left1, 2)
    lbl_left2 = cv.imread(path_left2, 2)

    change_left1_to = np.where(lbl_left1[:, :] != 0)
    change_left2_to = np.where(lbl_left2[:, :] != 0)

    lbl_left1[change_left1_to] = 1
    lbl_left2[change_left2_to] = 2

    lbl_left = lbl_left1 + lbl_left2

    change_left_overlap = np.where(lbl_left[:, :] == 3)
    lbl_left[change_left_overlap] = 0
    lbls_val_left[i-40] = lbl_left

    # Right
    path_right1 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/000.png'
    path_right2 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/001.png'

    lbl_right1 = cv.imread(path_right1, 2)
    lbl_right2 = cv.imread(path_right2, 2)

    change_right1_to = np.where(lbl_right1[:, :] != 0)
    change_right2_to = np.where(lbl_right2[:, :] != 0)

    lbl_right1[change_right1_to] = 1
    lbl_right2[change_right2_to] = 2

    lbl_right = lbl_right1 + lbl_right2

    change_right_overlap = np.where(lbl_right[:, :] == 3)
    lbl_right[change_right_overlap] = 0
    lbls_val_right[i-40] = lbl_right
    lbls_val[i-40] = np.concatenate((lbl_left[:, :, np.newaxis], lbl_right[:, :, np.newaxis]), axis=0)
for i in range(74, 77):
    # Left
    path_left1 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/000.png'
    path_left2 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/001.png'

    lbl_left1 = cv.imread(path_left1, 2)
    lbl_left2 = cv.imread(path_left2, 2)

    change_left1_to = np.where(lbl_left1[:, :] != 0)
    change_left2_to = np.where(lbl_left2[:, :] != 0)

    lbl_left1[change_left1_to] = 1
    lbl_left2[change_left2_to] = 2

    lbl_left = lbl_left1 + lbl_left2

    change_left_overlap = np.where(lbl_left[:, :] == 3)
    lbl_left[change_left_overlap] = 0
    lbls_val_left[i-69] = lbl_left

    # Right
    path_right1 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/000.png'
    path_right2 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/001.png'

    lbl_right1 = cv.imread(path_right1, 2)
    lbl_right2 = cv.imread(path_right2, 2)

    change_right1_to = np.where(lbl_right1[:, :] != 0)
    change_right2_to = np.where(lbl_right2[:, :] != 0)

    lbl_right1[change_right1_to] = 1
    lbl_right2[change_right2_to] = 2

    lbl_right = lbl_right1 + lbl_right2

    change_right_overlap = np.where(lbl_right[:, :] == 3)
    lbl_right[change_right_overlap] = 0
    lbls_val_right[i-69] = lbl_right
    lbls_val[i-69] = np.concatenate((lbl_left[:, :, np.newaxis], lbl_right[:, :, np.newaxis]), axis=0)


# Test labels
lbls_test_left = np.zeros((8, 800, 1280))
lbls_test_right = np.zeros((8, 800, 1280))
lbls_test = np.empty((8, 1600, 1280, 1))
for i in range(45, 50):
    # Left
    path_left1 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/000.png'
    path_left2 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/001.png'

    lbl_left1 = cv.imread(path_left1, 2)
    lbl_left2 = cv.imread(path_left2, 2)

    change_left1_to = np.where(lbl_left1[:, :] != 0)
    change_left2_to = np.where(lbl_left2[:, :] != 0)

    lbl_left1[change_left1_to] = 1
    lbl_left2[change_left2_to] = 2

    lbl_left = lbl_left1 + lbl_left2

    change_left_overlap = np.where(lbl_left[:, :] == 3)
    lbl_left[change_left_overlap] = 0
    lbls_test_left[i-45] = lbl_left

    # Right
    path_right1 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/000.png'
    path_right2 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/001.png'

    lbl_right1 = cv.imread(path_right1, 2)
    lbl_right2 = cv.imread(path_right2, 2)

    change_right1_to = np.where(lbl_right1[:, :] != 0)
    change_right2_to = np.where(lbl_right2[:, :] != 0)

    lbl_right1[change_right1_to] = 1
    lbl_right2[change_right2_to] = 2

    lbl_right = lbl_right1 + lbl_right2

    change_right_overlap = np.where(lbl_right[:, :] == 3)
    lbl_right[change_right_overlap] = 0
    lbls_test_right[i-45] = lbl_right
    lbls_test[i-45] = np.concatenate((lbl_left[:, :, np.newaxis], lbl_right[:, :, np.newaxis]), axis=0)
for i in range(77, 80):
    # Left
    path_left1 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/000.png'
    path_left2 = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/001.png'

    lbl_left1 = cv.imread(path_left1, 2)
    lbl_left2 = cv.imread(path_left2, 2)

    change_left1_to = np.where(lbl_left1[:, :] != 0)
    change_left2_to = np.where(lbl_left2[:, :] != 0)

    lbl_left1[change_left1_to] = 1
    lbl_left2[change_left2_to] = 2

    lbl_left = lbl_left1 + lbl_left2

    change_left_overlap = np.where(lbl_left[:, :] == 3)
    lbl_left[change_left_overlap] = 0
    lbls_test_left[i-72] = lbl_left

    # Right
    path_right1 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/000.png'
    path_right2 = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/001.png'

    lbl_right1 = cv.imread(path_right1, 2)
    lbl_right2 = cv.imread(path_right2, 2)

    change_right1_to = np.where(lbl_right1[:, :] != 0)
    change_right2_to = np.where(lbl_right2[:, :] != 0)

    lbl_right1[change_right1_to] = 1
    lbl_right2[change_right2_to] = 2

    lbl_right = lbl_right1 + lbl_right2

    change_right_overlap = np.where(lbl_right[:, :] == 3)
    lbl_right[change_right_overlap] = 0
    lbls_test_right[i-72] = lbl_right
    lbls_test[i-72] = np.concatenate((lbl_left[:, :, np.newaxis], lbl_right[:, :, np.newaxis]), axis=0)


print("Labels loaded!")

# Load poses
print("Loading poses...")

poses = np.load(PATH + "rosbag_annotations/pose_arr.npy")
print(poses.shape)
poses_train = np.concatenate((poses[:, :, 0:40], poses[:, :, 50:74]), axis=-1)
print(poses_train.shape)
poses_val = np.concatenate((poses[:, :, 40:45], poses[:, :, 74:77]), axis=-1)
print(poses_val.shape)
poses_test = np.concatenate((poses[:, :, 45:50], poses[:, :, 77:80]), axis=-1)
print(poses_test.shape)
print(poses_test[:, :, 0])
poses_train = poses_train.T
poses_val = poses_val.T
poses_test = poses_test.T
print(poses_test[0, :, :].T)

print("Poses loaded!")

# Build model
'''
inputs1 = Input(shape=(800, 1280, 3))
inputs2 = Input(shape=(800, 1280, 3))
# add = add([inputs1, inputs2])
concat = Concatenate(axis=-1)([inputs1, inputs2])
'''
input3 = Input(shape=(1600, 1280, 1))

conv1 = Conv2D(16, 3, activation='relu', padding='same')(input3)
conv1 = BatchNormalization(axis=-1)(conv1)
pool1 = MaxPooling2D(pool_size=2)(conv1)
pool1 = Dropout(droprate)(pool1)

conv2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
conv2 = BatchNormalization(axis=-1)(conv2)
pool2 = MaxPooling2D(pool_size=2)(conv2)
pool2 = Dropout(droprate)(pool2)

conv3 = Conv2D(64, 3, activation='relu', padding='same')(pool2)
conv3 = BatchNormalization(axis=-1)(conv3)
pool3 = MaxPooling2D(pool_size=2)(conv3)
pool3 = Dropout(droprate)(pool3)

conv4 = Conv2D(128, 3, activation='relu', padding='same')(pool3)
conv4 = BatchNormalization(axis=-1)(conv4)
pool4 = MaxPooling2D(pool_size=2)(conv4)
pool4 = Dropout(droprate)(pool4)

flat = Flatten()(pool4)
fc1 = Dense(32, activation='relu')(flat)
fc1 = BatchNormalization(axis=-1)(fc1)
fc1 = Dropout(0.5)(fc1)

output = Dense(16, activation='linear')(fc1)
output = Reshape((4, 4))(output)

model = Model(inputs=input3, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse', 'mae'])

tf.keras.utils.plot_model(model,
                          to_file='PoseEstimationPlot.png',
                          show_shapes=True,
                          rankdir='TB')

model.summary()

with open('PoseEstimationSummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Train model
history = model.fit(lbls_train, poses_train,
                    batch_size=1,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(lbls_val, poses_val))


predicted_poses = model.predict(lbls_test)
f = open("pose_estimation/predicted_poses.txt", "w+")
# f.write("True: " + poses_test[0, :, :].T)
# f.write("\nPredicted: " + predicted_poses[0, :, :].T)
for i in range(9):
    # f.write("True: " + poses_test[i, :, :].T + "\n")
    # f.write("Predicted: " + predicted_poses[i, :, :].T + "\n")
    # f.close()
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    f.write("True:\n" + np.array2string(poses_test[i, :, :].T, separator=', ') + "\n")
    f.write("Predicted:\n" + np.array2string(predicted_poses[i, :, :].T, separator=', ') + "\n")
f.close()

print('\n# Evaluate on test data')
start_time = time.time()
results = model.evaluate(lbls_test, poses_test, batch_size=1)
stop_time = time.time()
print("--- %s seconds ---" % ((stop_time - start_time)/len(imgs_test)))

f = open("pose_estimation/test_metrics.txt", "w+")
f.write("%s: %.4f" % (model.metrics_names[0], results[0]))
for i in range(1, len(results)):
    f.write("\n%s: %.4f" % (model.metrics_names[i], results[i]))

# Evaluate time and save to file
times = 10
total_time = 0.0
for i in range(times):
    print('\n# predict on test data ')
    start_time = time.time()
    model.predict(lbls_test, batch_size=1)
    stop_time = time.time()
    total_time += ((stop_time - start_time) / len(imgs_test))
    f.write("\nSeconds per image: %.4f" % ((stop_time - start_time)/len(imgs_test)))
    print("--- %s seconds ---" % ((stop_time - start_time)/len(imgs_test)))

average = total_time / times
fps = 1.0 / average

f.write("\nAverage: %.4f" % average)
f.write("\nFPS: %.2f" % fps)
f.close()


print("DONE!")

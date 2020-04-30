import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from PIL import Image
import cv2 as cv

try:
    from keras.layers import Input, Flatten, Dense, add, Reshape
    from keras.models import Model, load_model
except:
    from tensorflow.keras.layers import Input, Flatten, Dense, add, Reshape
    from tensorflow.keras.models import Model, load_model

which_path = 3
epochs = 10

if which_path == 1:
    # Christoffer:
    PATH = 'C:/Users/chris/Google Drive/Master Thesis/'
elif which_path == 2:
    # Jonathan:
    PATH = '/Users/jonathansteen/Google Drive/Master Thesis/'
elif which_path == 3:
    # Linux:
    PATH = '/home/jsteeen/'

# Load images
print("Loading images...")
# Train images
imgs_train_left = np.zeros((64, 800, 1280, 3))
imgs_train_right = np.zeros((64, 800, 1280, 3))
for i in range(0, 64):
    path_left = PATH + 'rosbag_annotations/img'+str(i)+'_left/data/002.png'
    path_right = PATH + 'rosbag_annotations/img'+str(i)+'_right/data/002.png'

    img_left = np.array(Image.open(path_left))[np.newaxis]
    img_right = np.array(Image.open(path_right))[np.newaxis]

    img_left = img_left / 255
    img_right = img_right / 255

    imgs_train_left[i] = img_left
    imgs_train_right[i] = img_right

# Validation images
imgs_val_left = np.zeros((8, 800, 1280, 3))
imgs_val_right = np.zeros((8, 800, 1280, 3))
for i in range(64, 72):
    path_left = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/002.png'
    path_right = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/002.png'

    img_left = np.array(Image.open(path_left))[np.newaxis]
    img_right = np.array(Image.open(path_right))[np.newaxis]

    img_left = img_left / 255
    img_right = img_right / 255

    imgs_val_left[i-64] = img_left
    imgs_val_right[i-64] = img_right

# Test images
imgs_test_left = np.zeros((8, 800, 1280, 3))
imgs_test_right = np.zeros((8, 800, 1280, 3))
for i in range(72, 80):
    path_left = PATH + 'rosbag_annotations/img' + str(i) + '_left/data/002.png'
    path_right = PATH + 'rosbag_annotations/img' + str(i) + '_right/data/002.png'

    img_left = np.array(Image.open(path_left))[np.newaxis]
    img_right = np.array(Image.open(path_right))[np.newaxis]

    img_left = img_left / 255
    img_right = img_right / 255

    imgs_test_left[i-72] = img_left
    imgs_test_right[i-72] = img_right

print("Images loaded!")

# Load labels
print("Loading labels...")
# Train labels
lbls_train_left = np.zeros((64, 800, 1280))
lbls_train_right = np.zeros((64, 800, 1280))
for i in range(0, 64):
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

# Validation labels
lbls_val_left = np.zeros((8, 800, 1280))
lbls_val_right = np.zeros((8, 800, 1280))
for i in range(64, 72):
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
    lbls_val_left[i-64] = lbl_left

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
    lbls_val_right[i-64] = lbl_right

# Test labels
lbls_test_left = np.zeros((8, 800, 1280))
lbls_test_right = np.zeros((8, 800, 1280))
for i in range(72, 80):
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

print("Labels loaded!")
print("Loading poses...")

poses = np.load(PATH + "rosbag_annotations/pose_arr.npy")
print(poses.shape)
poses_train = poses[:, :, 0:64]
poses_val = poses[:, :, 64:72]
poses_test = poses[:, :, 72:80]
print(poses_test[:, :, 0])
poses_train = np.reshape(poses_train, (64, 4, 4))
poses_val = np.reshape(poses_val, (8, 4, 4))
poses_test = np.reshape(poses_test, (8, 4, 4))
print(poses_test[:, :, 0])

print("Poses loaded!")

# Build model
in1 = Input(shape=(800, 1280, 3))
x1 = Flatten()(in1)
in2 = Input(shape=(800, 1280, 3))
x2 = Flatten()(in2)
add = add([x1, x2])
h1 = Dense(25, activation='relu')(add)
h2 = Dense(25, activation='relu')(h1)
h3 = Dense(25, activation='relu')(h2)
h4 = Dense(16, activation='relu')(h3)
out = Reshape((4, 4))(h4)

model = Model(inputs=[in1, in2], outputs=out)
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

model.summary()

# Train model
model.fit([imgs_train_left, imgs_train_right], poses_train,
          batch_size=1,
          epochs=epochs,
          verbose=1,
          validation_data=([imgs_val_left, imgs_val_right], poses_val))

predicted_pose = model.predict([imgs_test_left, imgs_test_right])
print(np.reshape(poses_test, (4, 4, 8)))
print(predicted_pose)
print("DONE!")

from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import cv2 as cv


def load_data(data_path, dtype=np.float32):
    N = 99            # Number of images
    M = 5             # Number of labels
    DIM = (480, 640)  # Image dimensions

    images = np.empty((N, *DIM, 3), dtype=dtype)
    labels = np.empty((N, *DIM, M), dtype=dtype)
    labels_display = np.empty((N, *DIM, 1), dtype=dtype)
    temp = np.empty((N, *DIM, 1), dtype=dtype)

    for i in range(N):
        image_path = os.path.join(data_path, 'Jigsaw annotations/Images/Suturing ({}).png'.format(i + 1))
        images[i] = cv.imread(image_path).astype(dtype)
        images[i] = cv.normalize(images[i], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

        for j in range(0,M-1):
            label_path = os.path.join(data_path, 'Jigsaw annotations/Annotated/Suturing ({})/data/00{}.png'.format(i + 1, j))
            labels[i,...,j+1] = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
            #labels_display[i, ..., 0] += labels[i, ..., j]
            labels[i,...,j+1] = cv.threshold(labels[i, ..., j + 1], dst=None, thresh=1, maxval=255, type=cv.THRESH_BINARY)[1]
            temp[i, ..., 0] += labels[i, ..., j+1]
            labels[i,...,j+1] = cv.normalize(labels[i, ..., j + 1], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

        for j in range(M-1):
            label_path = os.path.join(data_path, 'Jigsaw annotations/Annotated/Suturing ({})/data/00{}.png'.format(i + 1, j))
            im = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
            mask = cv.threshold(im, dst=None, thresh=1, maxval=255, type=cv.THRESH_BINARY)[1]
            k = np.where(mask == 255)
            labels_display[i][k] = (j + 1) * 30  # set pixel value here

        temp[i,...,0] = cv.threshold(temp[i, ..., 0], dst=None, thresh=1, maxval=255, type=cv.THRESH_BINARY_INV)[1]
        temp[i,...,0] = cv.normalize(temp[i, ..., 0], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        labels[i,...,0] = temp[i,...,0]
        images = images[..., ::-1] # flip from BGR to RGB (for display purposes)
    return images, labels, labels_display


def load_data_EndoVis15(data_path, dtype=np.float32):
    N = 120            # Number of images
    M = 3             # Number of labels
    DIM = (480, 640)  # Image dimensions

    images = np.empty((N, *DIM, 3), dtype=dtype)
    #labels = np.empty((N, *DIM, M), dtype=dtype)
    labels_display = np.empty((N, *DIM, 1), dtype=dtype)
    temp = np.empty((N, *DIM, 1), dtype=dtype)
    labels_temp = np.empty((N, *DIM), dtype=dtype)
    for l in range(1,4):
        for i in range(int(N/3)):
            image_path = os.path.join(data_path, 'Segmentation_Rigid_Training/Training/OP{}/Raw/img_{}_raw.png'.format(l, str(i + 1).zfill(2)))
            label_path = os.path.join(data_path, 'Segmentation_Rigid_Training/Training/OP{}/Masks/img_{}_class.png'.format(l, str(i + 1).zfill(2)))
            #print(image_path)
            #print(label_path)

            if l == 1:
                images[i] = cv.imread(image_path).astype(dtype)
                images[i] = cv.normalize(images[i], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
                labels_temp[i] = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
                labels_display[i,...,0] = labels_temp[i]
                labels_temp[i] = labels_temp[i] / 70

            elif l == 2:
                images[i+40] = cv.imread(image_path).astype(dtype)
                images[i+40] = cv.normalize(images[i+40], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
                labels_temp[i+40] = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
                labels_display[i+40,...,0] = labels_temp[i+40]
                labels_temp[i+40] = labels_temp[i+40] / 70

            elif l == 3:
                images[i+80] = cv.imread(image_path).astype(dtype)
                images[i+80] = cv.normalize(images[i+80], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
                labels_temp[i+80] = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
                labels_display[i+80,...,0] = labels_temp[i+80]
                labels_temp[i+80] = labels_temp[i+80] / 70
            '''
            for j in range(0,M):
                label_path = os.path.join(data_path, 'Segmentation_Rigid_Training/Training/OP{}/Masks/img_{}_class.png'.format(l, str(i + 1).zfill(2)))
                print(label_path)
                labels_temp[i] = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
                labels_temp[i] = labels_temp[i] / 70
                
                #k = np.where(label == 0)
                #labels[i,...,j][k] = 
                #labels_display[i, ..., 0] += labels[i, ..., j]
                labels[i,...,j+1] = cv.threshold(labels[i, ..., j + 1], dst=None, thresh=1, maxval=255, type=cv.THRESH_BINARY)[1]
                temp[i, ..., 0] += labels[i, ..., j+1]
                labels[i,...,j+1] = cv.normalize(labels[i, ..., j + 1], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
                
                
            for j in range(M-1):
                label_path = os.path.join(data_path, 'Annotated/Suturing ({})/data/00{}.png'.format(i + 1, j))
                im = cv.imread(label_path, cv.IMREAD_GRAYSCALE).astype(dtype)
                mask = cv.threshold(im, dst=None, thresh=1, maxval=255, type=cv.THRESH_BINARY)[1]
                k = np.where(mask == 255)
                labels_display[i][k] = (j + 1) * 30  # set pixel value here
            '''
        #temp[i,...,0] = cv.threshold(temp[i, ..., 0], dst=None, thresh=1, maxval=255, type=cv.THRESH_BINARY_INV)[1]
        #temp[i,...,0] = cv.normalize(temp[i, ..., 0], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        #labels[i,...,0] = temp[i,...,0]
        #images = images[..., ::-1] # flip from BGR to RGB (for display purposes)
    #labels_display = labels_temp
    #labels_display = labels_display.reshape((120, 480, 640, -1))
    labels = tf.keras.utils.to_categorical(labels_temp, num_classes=3, dtype='float32')
    print("labels: " + str(labels.shape))
    images = images[..., ::-1]  # flip from BGR to RGB (for display purposes)
    return images, labels, labels_display


def load_data_EndoVis17(data_path, dtype=np.float32):
    N = 225  # Number of images
    M = 4  # Number of labels
    DIM = (1024, 1280)  # Image dimensions

    images = np.empty((N, *DIM, 3), dtype=dtype)
    images_temp = np.empty((1080,1920, 3), dtype=dtype)
    labels = np.empty((N, *DIM, M), dtype=dtype)
    labels_display = np.empty((N, *DIM, 1), dtype=dtype)
    temp = np.empty((N, *DIM, 1), dtype=dtype)
    labels_temp = np.empty((1080,1920), dtype=dtype)
    labels_crop = np.empty((N, *DIM), dtype=dtype)
    #labels_temp_temp = np.empty(DIM, dtype=dtype)
    for i in range(N):
        print(str(i+1) + ' of 225')
        image_path = os.path.join(data_path, 'instrument_1_4_training/instrument_dataset_3/left_frames/frame{}.png'.format(str(i).zfill(3)))
        images_temp = cv.imread(image_path).astype(dtype)
        images[i] = images_temp[28:1052, 320:1600] #crop the black parts
        cv.imwrite('pictures_deepunet/image{}.png'.format(str(i).zfill(3)), images[i])
        images[i] = cv.normalize(images[i], dst=None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)

        label_path_left = os.path.join(data_path, 'instrument_1_4_training/instrument_dataset_3/ground_truth/Left_Large_Needle_Driver_labels/frame{}.png'.format(str(i).zfill(3)))
        label_path_right = os.path.join(data_path, 'instrument_1_4_training/instrument_dataset_3/ground_truth/Right_Large_Needle_Driver_labels/frame{}.png'.format(str(i).zfill(3)))
        labels_temp = cv.imread(label_path_left, cv.IMREAD_GRAYSCALE).astype(dtype) + cv.imread(label_path_right, cv.IMREAD_GRAYSCALE).astype(dtype)
        k1 = np.where(labels_temp[:, :] == 60)
        k2 = np.where(labels_temp[:, :] == 50)
        labels_temp[k1] = 30
        labels_temp[k2] = 30
        labels_crop[i] = labels_temp[28:1052, 320:1600]
        labels_display[i,...,0] = labels_crop[i]
        labels_crop[i] = labels_crop[i] / 10

    labels = tf.keras.utils.to_categorical(labels_crop, num_classes=4, dtype='float32')
    print("labels: " + str(labels.shape))
    images = images[..., ::-1]  # flip from BGR to RGB (for display purposes)

    return images, labels, labels_display

#categorical focal loss fromTh
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


def weighted_categorical_crossentropy(weights=[1]):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * -K.log(y_pred) * weights
        loss = K.sum(loss, -1)
        return loss

    return loss

def jaccard(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2, 3])
    union = K.sum(y_true, axis=[0, 1, 2, 3]) + K.sum(y_pred, axis=[0, 1, 2, 3]) - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

# https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def iou_coef_mean(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2]) - intersection
    iou_mean = K.mean((intersection + 1e-15) / (union + 1e-15), axis = 0)
    return iou_mean

def iou_coef0(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2]) - intersection
    iou = (intersection + 1e-15) / (union + 1e-15)
    return iou[0]

def iou_coef1(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2]) - intersection
    iou = (intersection + 1e-15) / (union + 1e-15)
    return iou[1]

def iou_coef2(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2]) - intersection
    iou = (intersection + 1e-15) / (union + 1e-15)
    return iou[2]

def iou_coef3(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2]) - intersection
    iou = (intersection + 1e-15) / (union + 1e-15)
    return iou[3]

def iou_coef4(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2]) - intersection
    iou = (intersection + 1e-15) / (union + 1e-15)
    return iou[4]


# https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def dice(y_true, y_pred):
    return (2 * K.sum(y_true * y_pred) + 1e-15) / (K.sum(y_true) + K.sum(y_pred) + 1e-15)

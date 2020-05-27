from keras import backend as K

def iou_coef(y_true, y_pred, smooth=K.epsilon()):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])#-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

import numpy as np
import tensorflow as tf

# create two inputs simulating a batch_size = 3
# shape (3,5)
y_true = np.array([[1,5,2,3,0.5], [1,5,2,3,0.5], [1,5,2,3,0.5]])
y_pred = np.array([[2,4,3,3,0.7], [2,4,3,3,0.7], [2,4,3,3,0.7]])

# call the first function
result = iou_coef(y_true, y_pred)

print(result)
print(type(result))
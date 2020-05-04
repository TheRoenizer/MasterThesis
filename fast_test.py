import sys
sys.path.insert(0,'/home/jandersenlocal/sshfs/jkha/keras_segmentation')


import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io
#from tensorflow.keras.models import load_model
from keras.models import load_model, model_from_json
import tensorflow as tf
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import os
import json

from keras_lookahead import Lookahead
from keras_radam import RAdam

from keras_custom_losses.dice_loss import dice_coef

def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen


# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

model = load_model('models/unet/binary_crossentropy/unet_lr_10E-03_drop_03_loss_365E-04.hdf5',custom_objects={'Lookahead': Lookahead,'RAdam': RAdam,'dice_coef': dice_coef})

im_size = (640,512)

try:
    p0 = im_size[0] % model.input_shape[0][1]
    p1 = im_size[1] % model.input_shape[0][2]
except:
    p0 = im_size[0] % model.input_shape[1]
    p1 = im_size[1] % model.input_shape[2]

im_size =  (im_size[0]+p0,im_size[1]+p1)
#im_size[1] += p1

weights = model.get_weights()
json_dict = json.loads(model.to_json())
#print json_dict
#print json_dict['config']['layers']
for i in range(len(json_dict['config']['layers'])):
    try:
        json_dict['config']['layers'][i]['config']['batch_input_shape'] = [None, im_size[0], im_size[1], 3]
        #print i
    except:
        pass
  
json_model = json.dumps(json_dict)
model = model_from_json(json_model)
model.set_weights(weights)
print(model.input_shape)


from tensorflow.keras.preprocessing import image

# Optional image to test model prediction.
img_path = 'mosiak/IM000069.JPG'

img = image.load_img(img_path, target_size=(im_size[0],im_size[1]))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

import time
times = []
for i in range(20):
    start_time = time.time()
    pred = model.predict(x)
    delta = (time.time() - start_time)
    times.append(delta)

mean_delta = np.array(times).mean()
fps = 1 / mean_delta

print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))

session = tf.keras.backend.get_session()

"""
for layer in model.layers:
    print(layer.name)
"""
input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

# Prints input and output nodes names, take notes of them.
print(input_names, output_names)
print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))

frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs])

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50)

graph_io.write_graph(trt_graph, "./model/",
                     "trt_graph.pb", as_text=False)


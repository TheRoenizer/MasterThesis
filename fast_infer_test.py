import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph('./model/trt_graph.pb')

input_names = [u"input_1_1"]
output_names = [u"conv2d_22_1/Sigmoid"]

input_names = [u'input_2_1']
output_names = [u'conv2d_48_1/Sigmoid']


# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')

# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))


#print [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]



input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"


print("input_tensor_name: {}\noutput_tensor_name: {}".format(input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)


from tensorflow.keras.preprocessing import image

# Optional image to test model prediction.
img_path = 'mosiak/IM000069.JPG'

img = image.load_img(img_path, target_size=image_size[:2])
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

feed_dict = {
    input_tensor_name: x
}
preds = tf_sess.run(output_tensor, feed_dict)

preds.shape

import time
times = []
for i in range(20):
    start_time = time.time()
    one_prediction = tf_sess.run(output_tensor, feed_dict)
    delta = (time.time() - start_time)
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1 / mean_delta
print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))

tf_sess.close()


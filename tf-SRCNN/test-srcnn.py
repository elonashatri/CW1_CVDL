import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import scipy
from skimage import measure
from PIL import Image
import imageio

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def preprocess(path, scale=3):
  """
  Preprocess single image file
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)

  print("Image size: {0}".format(image.size))
  print("Image shape: {0}".format(image.shape))

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  # shrinking the high res image by 3
  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  # scaling high res image by 3
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255

"""Define the model weights and biases
"""
# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

"""Define the model layers with three convolutional layers. Compute feature maps
"""
# conv1 layer with biases and relu : 64 filters with size 9 x 9
conv1 = tf.nn.relu(
    tf.nn.bias_add(
        tf.nn.conv2d(
            inputs,
            weights['w1'],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv1"
        ),
        biases['b1']
    )
)

## Non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1
conv2 = tf.nn.relu(
    tf.nn.bias_add(
        tf.nn.conv2d(
            conv1,
            weights['w2'],
            strides=[1, 1, 1, 1],
            padding="VALID",
        ),
        biases['b2']
    )
)

# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = tf.nn.bias_add(
    tf.nn.conv2d(
        conv2,
        weights['w3'],
        strides=[1, 1, 1, 1],
        padding="VALID",
    ),
    biases['b3']
)

"""Load the pre-trained model file
"""
model_path='/homes/es314/cv/tf-SRCNN/model/model.npy'
model = np.load(model_path, encoding='latin1', allow_pickle=True).item()

# MODEL WEIGHTS
# LAYER 1
"""Weights and bias of the first layer's first filter
"""
layer1_filter1 = model['w1'][:, :, :, 0]
print("layer 1 filter 1: {0}".format(layer1_filter1))
print("bias: {0}".format(model['b1'][9]))

"""Visualisation of first convolutional layer's first filter
"""
# Get rid of the last dimension to get 9 x 9
layer1_filter1 = np.squeeze(model['w1'][:, :, :, 0])
fig = plt.figure()
plt.imshow(layer1_filter1)
plt.savefig('layer1_filter1.png')
plt.close(fig)

## LAYER 2
"""Show 2nd layer 4th filter, 5th filter's bias, and number of input channels
"""
print("layer 2 filter 5: {0}".format(model['w2'][:,:,:,4]))
print("bias: {0}".format(model['b2'][5]))
print("layer 2 input channel number: {0}".format(model['w2'].shape[2]))
# display the filter
# plt.subplots(1, model['w2'].shape[2])
# for i in range(0, model['w2'].shape[2]):
#     plt.imshow(model['w2'][:, :, i, 4])
#     plt.show()
# # channel number of input


# # LAYER 3
def visualise_filters_layer3():
 fig = plt.figure(figsize=(10, 10))
 cols = 4
 rows = 8

 num_filters = model['w3'].shape[2]
 for i in range(0, num_filters):
     fig.add_subplot(rows, cols, i + 1)
     plt.imshow(model['w3'][:, :, i, 0])
 plt.savefig('layer3_filter{0}.png'.format(0))
 plt.close(fig)

print("layer 3 filter 1: {0}".format(model['w3'][:,:,:,0]))
# # 1st bias value
print("layer 3 bias 1: {0}".format(model['b3'][0]))
# Visualise layer 3, 1st filter feature map
visualise_filters_layer3()

"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
sess = tf.Session()

for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
blurred_image, groundtruth_image = preprocess('/homes/es314/cv/tf-SRCNN/image/w-01_p010.jpg')
# Save and show the ground truth image
scipy.misc.imsave('groundtruth.png', groundtruth_image)

"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
output_ = sess.run(conv3, feed_dict={inputs: input_})

scipy.misc.imsave('super_res.png', output_[0, :, :, 0])
scipy.misc.imsave('low_res.png', input_[0, :, :, 0])

''' Find difference in dimensions between groundtruth and super res output
    and take the center pixels of the groundtruth image to compare PSNR
'''
super_res = np.reshape(output_, (output_.shape[1], output_.shape[2]))
dim_diff = groundtruth_image.shape[0] - super_res.shape[0]
offset_start = int(dim_diff / 2) # offset from top/left edge to center
offset_end = 255 - offset_start # offset from bottom/right edge to center
high_res_psnr = measure.compare_psnr(
    groundtruth_image[offset_start:offset_end, offset_start:offset_end], super_res
)
print("HR-SRCNN vs groundtruth PSNR: {0}".format(high_res_psnr))
baseline_psnr = measure.compare_psnr(groundtruth_image, blurred_image)
print("baseline HR-BI vs groundtruth PSNR: {0}".format(baseline_psnr))

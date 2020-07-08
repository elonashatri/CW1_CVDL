import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import scipy
import pdb
import imageio
from skimage import measure
from PIL import Image


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

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
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

## ------ Add your code here: set the weight of three conv layers
# replace '0' with your hyper parameter numbers 
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
# finished
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





"""Define the model layers with three convolutional layers
"""
## ------ Add your code here: to compute feature maps of input low-resolution images
# replace 'None' with your layers: use the tf.nn.conv2d() and tf.nn.relu()
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
##------ Add your code here: to compute non-linear mapping
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


##------ Add your code here: compute the reconstruction of high-resolution image
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

##------ Add your code here: show the weights of model and try to visualisa
# variabiles (w1, w2, w3)

# Layer 1
# Printing weights and bias of the first layer's first filter

# layer 1 filter 1 represents layer 1 filter 1
first_layer_first_filter = model['w1'][:, :, :, 0]
print("Layer 1, Filter 1: {0}".format(first_layer_first_filter))
#show the tenth filter bias
print("Bias: {0}".format(model['b1'][9]))

#Visualising the first convolutional layer filter 

# First we squeeze to get rid of the last dimension and leave 9x9x1 (or 9x9) dimension
first_layer_first_filter = np.squeeze(first_layer_first_filter)
fig_1 = plt.figure()
plt.imshow(first_layer_first_filter)
plt.savefig('first_layer_first_filter.png')
plt.close(fig_1)




# Layer 2
# Printing weights and bias of the second layer's 5th filter, 6th filter bias and input channels number
 
second_layer_fifth_filter = model['w2'][:, :, :,4]
second_layer_sixth_bias = model['b2'][5]
second_layer_input_channel = model['w2'].shape[2]

      
print("Layer 2 filter 5: {0}".format(second_layer_fifth_filter))
print("Layer 2, filter 6, bias: {0}".format(second_layer_sixth_bias))
print("Second layer input channel number: {0}".format(second_layer_input_channel))

# Visualising the fourth convolutional layer filter

plt.subplots(1, second_layer_input_channel)
for i in range(0, second_layer_input_channel):
      plt.imshow(model['w2'][:, :, i,4])
      plt.show()
      ##plt.savefig('l4_f4.png')
                    
                    
# Layer 3
# Printing weights and bias of the third layer's th filter, 4th filter bias and input channels number
def visualise_third_layer_filters():
 fig = plt.figure(figsize=(10, 10))
 cols = 4
 rows = 8

 num_filters = model['w3'].shape[2]
 for i in range(0, num_filters):
     fig.add_subplot(rows, cols, i + 1)
     plt.imshow(model['w3'][:, :, i, 0])
 plt.savefig('Third layers filter {0}.png'.format(0))
 plt.close(fig)

 



print("Layer 3 filter 1: {0}".format(model['w3'][:,:,:,0]))
      
# # 1st bias value
print("Layer 3 bias 1: {0}".format(model['b3'][0]))
      
# Visualise layer 3, 1st filter feature map
visualise_third_layer_filters()

      
      


# Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file

# launch a sessionv
sess = tf.Session()

for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))

#Read the test image

blurred_image, ground_truth_img = preprocess('/homes/es314/cv/tf-SRCNN/image/butterfly_GT.bmp')
      
# To save and show the ground truth image
scipy.misc.imsave('groundtruth.png', ground_truth_img)

# Run the model and get the SR image

# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
ouput_ = sess.run(conv3, feed_dict={inputs: input_})



##------ Add your code here: save the blurred and SR images and compute the psnr
# hints: use the 'scipy.misc.imsave()'  and ' skimage.measure.compare_psnr()'
# Save low and high resolution images 
output_ = sess.run(conv3, feed_dict={inputs: input_})

scipy.misc.imsave('super_res.png', output_[0, :, :, 0])
scipy.misc.imsave('low_res.png', input_[0, :, :, 0])

# Compute PSNR for groundtruth and super resulution output, and then compare PSNR for the center pixel of the groundtruth image 
super_res = np.reshape(output_, (output_.shape[1], output_.shape[2]))
diff_dim = ground_truth_img.shape[0] - super_res.shape[0]
# from top-left to edge center
offset_start = int(diff_dim / 2) 
# from bottom-right to edge center
offset_end = 255 - offset_start
 
#compare psnr between groundtruth image and super resolution outputed image
high_res_psnr = measure.compare_psnr(ground_truth_img[offset_start:offset_end, offset_start:offset_end], super_res)
# copare psnr between the groundtruth image and the degraded (blurred image)
baseline_psnr = measure.compare_psnr(ground_truth_img, blurred_image)

print("High resolution - super resolution CNN VS groundtruth: {0}".format(high_res_psnr))
print("baseline blurred vs groundtruth PSNR: {0}".format(baseline_psnr))
      


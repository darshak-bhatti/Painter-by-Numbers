


'''
http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
http://stackoverflow.com/questions/33783672/how-can-i-visualize-the-weightsvariables-in-cnn-in-tensorflow
https://www.youtube.com/watch?v=uO3CMMT459w
https://www.youtube.com/watch?v=dYhrCUFN0eM
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels

Team Members :
John Gray
Phillip Hardy
Mihir Mirajkar
Piyush Choudhary
Darshak Harisinh Bhatti
Erick Draayer

'''
################################################################################################################################
#This file was created by Darshak Bhatti.
#Unity ID: dbhatti
################################################################################################################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import random
#from os import listdir
#from os.path import isfile, join
from mlxtend.preprocessing import one_hot




#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

print "Hello World"

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16        # There are 16 of these filters.

# Convolutional Layer 2 + Pool.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 16         # There are 16 of these filters.

# Convolutional Layer 3.
filter_size3 = 5          # Convolution filters are 5 x 5 pixels.
num_filters3 = 32         # There are 32 of these filters.

# Convolutional Layer 4.
filter_size4 = 5          # Convolution filters are 5 x 5 pixels.
num_filters4 = 32         

# Convolutional Layer 5 + Pool.
filter_size5 = 4          # Convolution filters are 4 x 4 pixels.
num_filters5 = 32         

# Convolutional Layer 6.
filter_size6 = 4          # Convolution filters are 5 x 5 pixels.
num_filters6 = 64         

# Convolutional Layer 7.
filter_size7 = 5          # Convolution filters are 5 x 5 pixels.
num_filters7 = 64         

# Convolutional Layer 8 + Pool.
filter_size8 = 5          # Convolution filters are 5 x 5 pixels.
num_filters8 = 64         

# Convolutional Layer 9.
filter_size9 = 5          # Convolution filters are 5 x 5 pixels.
num_filters9 = 128         

# Convolutional Layer 10.
filter_size10 = 5          # Convolution filters are 5 x 5 pixels.
num_filters10 = 128       

# Convolutional Layer 11 + Pool.
filter_size11 = 5          # Convolution filters are 5 x 5 pixels.
num_filters11 = 128        

# Convolutional Layer 12 + Pool.
filter_size12 = 5          # Convolution filters are 5 x 5 pixels.
num_filters12 = 256        

# Fully-connected layer.
fc_size = 16384             # Number of neurons in fully-connected layer.

#Images are 256 pixels in each dimension.
img_size = 256

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels
num_channels = 3

# Number of classes
num_classes = 1584

<<<<<<< HEAD:CNN/cnn.py
################################################################################################################################
#This method was created by Darshak Bhatti.
#Unity ID: dbhatti
################################################################################################################################

def plot_images(images, cls_true, cls_pred=None):
    if len(images) == 0:
        print("no images to show")
        return
    else:
        random_indices = random.sample(range(len(images)), min(len(images), 9))

    images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

################################################################################################################################
#This method was created by Darshak Bhatti.
#Unity ID: dbhatti
################################################################################################################################
=======
>>>>>>> 1a38084f079bfea8b9d91c1468af951629125291:cnn.py

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

################################################################################################################################
#This method was created by Darshak Bhatti.
#Unity ID: dbhatti
################################################################################################################################

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

################################################################################################################################
#This method was created by Darshak Bhatti.
#Unity ID: dbhatti
################################################################################################################################

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights
    weights = new_weights(shape=shape)

    # Create new biases
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
                         
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights

################################################################################################################################
#This method was created by Darshak Bhatti.
#Unity ID: dbhatti
################################################################################################################################

def flatten_layer(layer):

    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])
    
    return layer_flat, num_features

################################################################################################################################
#This method was created by Darshak Bhatti.
#Unity ID: dbhatti
################################################################################################################################

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

################################################################################################################################
#This method was created by Darshak Bhatti.
#Unity ID: dbhatti
################################################################################################################################

def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    #images_path = "/Users/darshak/PycharmProjects/TensorFlow/preprocessed/"
    images_path = "./train/"
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(',')
        filename = filename[1:-1]
        filenames.append(images_path + filename)
        label = label[1:-1]
        labels.append(int(label))
    return filenames, labels

################################################################################################################################
#This method was created by Darshak Bhatti.
#Unity ID: dbhatti
################################################################################################################################

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label

# Reads pathes of images together with their labels
#filename = "/Users/darshak/refined_art2.csv"
filename = "./refined_art2_train_f.csv"
image_list, label_list = read_labeled_image_list(filename)
images = tf.convert_to_tensor(image_list)
labels = tf.convert_to_tensor(label_list)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

image, label = read_images_from_disk(input_queue)

# Optional Preprocessing or Data Augmentation
# tf.image implements most of the standard image augmentation
#image = preprocess_image(image)
#label = preprocess_label(label)
image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(image, [256, 256])


# Optional Image and Label Batching
image_batch, label_batch = tf.train.batch([resized_image, label],
                                          batch_size=100)



#For testing
filename_test = "./refined_art2_test_f.csv"
image_list_test, label_list_test = read_labeled_image_list(filename_test)
images_test = tf.convert_to_tensor(image_list_test)
labels_test = tf.convert_to_tensor(label_list_test)

# Makes an input queue
input_queue_test = tf.train.slice_input_producer([images_test, labels_test], shuffle=True)

image_test, label_test = read_images_from_disk(input_queue_test)

# Optional Preprocessing or Data Augmentation
# tf.image implements most of the standard image augmentation
#image = preprocess_image(image)
#label = preprocess_label(label)
image_test = tf.cast(image_test, tf.float32)
resized_image_test = tf.image.resize_images(image_test, [256, 256])


# Optional Image and Label Batching
image_batch_test, label_batch_test = tf.train.batch([resized_image_test, label_test],
                                          batch_size=100)

                                          
#x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.placeholder(tf.float32, shape = [None, img_size, img_size, 3], name = "x_image")

#x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 1584], name='y_true') #one-hot


y_true_cls = tf.argmax(y_true, dimension=1) #one-hot to 1-D

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=False)

print "layer_conv1", layer_conv1


layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

print "layer_conv2", layer_conv2


layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size2,
                   num_filters=num_filters3,
                   use_pooling=False)

print "layer_conv3", layer_conv3

layer_conv4, weights_conv4 = \
    new_conv_layer(input=layer_conv3,
                   num_input_channels=num_filters3,
                   filter_size=filter_size2,
                   num_filters=num_filters4,
                   use_pooling=False)

print "layer_conv4", layer_conv4

layer_conv5, weights_conv5 = \
    new_conv_layer(input=layer_conv4,
                   num_input_channels=num_filters4,
                   filter_size=filter_size2,
                   num_filters=num_filters5,
                   use_pooling=True)

print "layer_conv5", layer_conv5

layer_conv6, weights_conv6 = \
    new_conv_layer(input=layer_conv5,
                   num_input_channels=num_filters5,
                   filter_size=filter_size2,
                   num_filters=num_filters6,
                   use_pooling=False)

print "layer_conv6", layer_conv6

layer_conv7, weights_conv7 = \
    new_conv_layer(input=layer_conv6,
                   num_input_channels=num_filters6,
                   filter_size=filter_size2,
                   num_filters=num_filters7,
                   use_pooling=False)

print "layer_conv7", layer_conv7

layer_conv8, weights_conv8 = \
    new_conv_layer(input=layer_conv7,
                   num_input_channels=num_filters7,
                   filter_size=filter_size2,
                   num_filters=num_filters8,
                   use_pooling=True)

print "layer_conv8", layer_conv8


layer_conv9, weights_conv9 = \
    new_conv_layer(input=layer_conv8,
                   num_input_channels=num_filters8,
                   filter_size=filter_size2,
                   num_filters=num_filters9,
                   use_pooling=False)

print "layer_conv9", layer_conv9

layer_conv10, weights_conv10 = \
    new_conv_layer(input=layer_conv9,
                   num_input_channels=num_filters9,
                   filter_size=filter_size2,
                   num_filters=num_filters10,
                   use_pooling=False)

print "layer_conv10", layer_conv10

layer_conv11, weights_conv11 = \
    new_conv_layer(input=layer_conv10,
                   num_input_channels=num_filters10,
                   filter_size=filter_size2,
                   num_filters=num_filters11,
                   use_pooling=True)

print "layer_conv11", layer_conv11


layer_conv121, weights_conv121 = \
    new_conv_layer(input=layer_conv11,
                   num_input_channels=num_filters11,
                   filter_size=filter_size2,
                   num_filters=num_filters12,
                   use_pooling=False)

print "layer_conv121", layer_conv121

layer_conv122, weights_conv122 = \
    new_conv_layer(input=layer_conv11,
                   num_input_channels=num_filters11,
                   filter_size=filter_size2,
                   num_filters=num_filters12,
                   use_pooling=False)

print "layer_conv122", layer_conv122


layer_conv12, weights_conv12 = \
    new_conv_layer(input=layer_conv122,
                   num_input_channels=num_filters12,
                   filter_size=filter_size2,
                   num_filters=num_filters12,
                   use_pooling=True)

print "layer_conv12", layer_conv12


layer_flat, num_features = flatten_layer(layer_conv12)

print "layer_flat", layer_flat
print "num_features", num_features

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=False)

print "layer_fc1", layer_fc1

layer_fc11 = new_fc_layer(input=layer_fc1,
                         num_inputs=16384,
                         num_outputs=2048,
                         use_relu=True)

print "layer_fc11", layer_fc11

layer_fc12 = new_fc_layer(input=layer_fc11,
                         num_inputs=2048,
                         num_outputs=1584,
                         use_relu=True)

print "layer_fc12", layer_fc12


layer_fc2 = new_fc_layer(input=layer_fc12,
                         num_inputs=1584,
                         num_outputs=num_classes,
                         use_relu=True)

print "layer_fc2", layer_fc2


y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, dimension=1) #The class-number is the index of the largest element.


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)


correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 100

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=session, coord=coord)



#Test the trained net
f = open('results', 'w')



for i in range(1,200000):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        #print " ..", i
        x_batch, label_array = session.run([image_batch, label_batch])
        #print "x_batch :: ", x_batch
        #print "y_true_batch :: ", y_true_batch
        #y_true_batch = np.zeros((100, 43))
        #y_true_batch[np.arange(100), label_array] = 1
        #print "LA ",label_array

        y_true_batch = one_hot(label_array, num_labels=1584, dtype='int')

        # print "x_batch :: ", x_batch
        #print "y_true_batch :: ", y_true_batch

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x_image: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        print "Running Optimizer ... ", i
        session.run(optimizer, feed_dict=feed_dict_train)


        # Print status every 100 iterations.
        if i % 100 == 0:
	    print "\n\nTrain : "
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

            print "\n\n Test : "

	    x_batch_test, label_array_test = session.run([image_batch_test, label_batch_test])

            y_true_batch_test = one_hot(label_array_test, num_labels=1584, dtype='int')
            feed_dict_test = {x_image: x_batch_test, y_true: y_true_batch_test}



            acct = session.run(accuracy, feed_dict=feed_dict_test)

            msg = "Iteration: {0:>6}, Test Accuracy: {1:>6.1%}"

            # Print
            print(msg.format(i + 1, acct))

	    f.write(str(acc) + ", " + str(acct) + "\n")


#Test the trained net
for j in range(0, 30):
    print " ..", j
    x_batch_test, label_array_test = session.run([image_batch_test, label_batch_test])

    y_true_batch_test = one_hot(label_array_test, num_labels=1584, dtype='int')
    feed_dict_test = {x_image: x_batch_test,
                       y_true: y_true_batch_test}


    #cls_pred[0:100] = session.run(y_pred_cls, feed_dict=feed_dict_test)

    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Message for printing.
    msg = "Iteration: {0:>6}, Test Accuracy: {1:>6.1%}"

    # Print
    print(msg.format(i + 1, acc))




    
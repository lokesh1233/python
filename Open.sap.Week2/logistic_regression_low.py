from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pylab

tf.reset_default_graph()
sess = tf.Session()

# Import the MNIST dataset. 
# It will be downloaded to '/tmp/data' if you don't already have a local copy.
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

# Uncomment these lines to understand the format of the dataset.

# 1. There are 55k, 5k, and 10k examples in train, validation, and test.
print('Train, validation, test: %d, %d, %d' % 
       (len(mnist.train.images), len(mnist.validation.images), len(mnist.test.images)))

# 2. The format of the labels is 'one-hot'.
# The fifth image happens to be a '6'.
# This is represented as '[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]'
print(mnist.train.labels[4])

# You can find the index of the label, like this:
print (np.argmax(mnist.train.labels[4]))

# 3. An image is a 'flattened' array of 28*28 = 784 pixels.
print (len(mnist.train.images[4]))

# 4. To display an image, first reshape it to 28x28.
pylab.imshow(mnist.train.images[4].reshape((28,28)), cmap=pylab.cm.gray_r)   
pylab.title('Label: %d' % np.argmax(mnist.train.labels[4]))

NUM_CLASSES = 10
NUM_PIXELS = 28 * 28
TRAIN_STEPS = 2000
BATCH_SIZE = 100
LEARNING_RATE = 0.5

# Define inputs
images = tf.placeholder(dtype=tf.float32, shape=[None, NUM_PIXELS])
labels = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])

# Define model
W = tf.Variable(tf.truncated_normal([NUM_PIXELS, NUM_CLASSES]))
b = tf.Variable(tf.zeros([NUM_CLASSES]))
y = tf.matmul(images, W) + b

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# Initialize variables after the model is defined
sess.run(tf.global_variables_initializer())

# Train the model
for i in range(TRAIN_STEPS):
    batch_images, batch_labels = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={images: batch_images, labels: batch_labels})

# Evaluate the trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Accuracy %f" % sess.run(accuracy, feed_dict={images: mnist.test.images, 
                                                    labels: mnist.test.labels}))

#As written, this code evaluates the accuracy of the trained model on the 
#entire testing set. Below is a function to predict the label for a single image.

prediction = tf.argmax(y,1)
def predict(i):
    image = mnist.test.images[i]
    actual_label = np.argmax(mnist.test.labels[i])
    predicted_label = sess.run(prediction, feed_dict={images: [image]})
    return predicted_label, actual_label

i = 0
predicted, actual = predict(i)
print ("Predicted: %d, actual: %d" % (predicted, actual))
pylab.imshow(mnist.test.images[i].reshape((28,28)), cmap=pylab.cm.gray_r)
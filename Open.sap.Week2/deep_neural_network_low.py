from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from logistic_regression_low import correct_prediction, train_step

tf.reset_default_graph()
sess = tf.Session()

# tip: if you run into problems with TensorBoard
# clear the contents of this directory, re-run this script
# then restart TensorBoard to see the result
LOGDIR = './graphs' 

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

# number of neurons in each hidden Layer
HIDDEN1_SIZE = 500
HIDDEN2_SIZE = 250

NUM_CLASSES = 10
NUM_PIXELS = 28 * 28

# experiment with the number of training steps to 
# see the effect
TRAIN_STEPS = 2000
BATCH_SIZE = 100

# we're using a different learning rate than the previous
# notebook, and a new optimizer
LEARNING_RATE = 0.001

# Define inputs
with tf.name_scope('input'):
    images = tf.placeholder(tf.float32, [None, NUM_PIXELS], name="pixels")
    labels = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="labels")

#function to create a fully connected layers
def fc_layer(input, size_out, name="fc", activation=None):
    with tf.name_scope(name):
        size_in = int(input.shape[1])
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev = 0.1), name = "weights")
        b = tf.Variable(tf.constant(0.1, shape = [size_out], name ="bias"))
        wx_plus_b = tf.matmul(input, w) + b
        if activation: return activation(wx_plus_b)
        return wx_plus_b
    
# this way we initialize variables has an affect on how quickly
# training converges. we may explore with different strategies later.
# w = tf.Variable(tf.truncated_normal(shape=[size_in, size_out], stddev=1.0 / math.sqrt(float(size_in))))


# define the model 

# First, we'll create two fully connected layers, with ReLU activations
fc1 = fc_layer(images, HIDDEN1_SIZE, "fc1", activation = tf.nn.relu)
fc2 = fc_layer(fc1, HIDDEN2_SIZE, "fc2", activation = tf.nn.relu)

# Next we'll apply Dropout to the second layer
# This can help prevent overfitting, and I've added it here
# for illustration. you can comment this out, if you like.
dropped = tf.nn.dropout(fc2, keep_prob=0.9)

# Finally, we'll calcualte logists. This will be
# the input to our softmax function, Notice we don't apply an activation at this layer.
# If you've commented out the dropout layer,
# switch the input here to 'fc2'.
y = fc_layer(dropped, NUM_CLASSES, name="output")

# Define loss and an optimizer
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels = labels))
    tf.summary.scalar('loss', loss)

with tf.name_scope("optimizer"):
    # whereas in the previous notebook we used a vanilla GradientDescentOptimizer
    # here, we're using adam. This is a single line of code change, and more
    # importantly, Tensorflow will still automatically analyze our graph
    # and determine how to adjust the variables to decrease the loss.
    tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# define evaluation
with tf.name_scope("evaluation"):
    #these three lines are identical to the previous notebook.
    correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# set up logging
# we'll use a second FileWriter to summarize accuracy on 
# the test set. This will let us display it nicely in TensorBoard.
train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'train'))
train_writer.add_graph(sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOGDIR, 'test'))
summry_op = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

for step in range(TRAIN_STEPS):
    batch_xs, batch_ys =mnist.train.next_batch(BATCH_SIZE)
    summary_result, _ =sess.run([summry_op,accuracy], feed_dict = {images:batch_xs,
                                                                 labels:batch_ys})
    train_writer.add_summary(summary_result, step)
    train_writer.add_run_metadata(tf.RunMetadata(), 'step%03d' % step)
    
    # calculate accuracy on the test set, every 100 steps.
    # we're using the entire test set here, so this will be a bit slow 
    if step % 100 == 0:
        summary_reslut, acc = sess.run([summry_op, accuracy], feed_dict={
            images:mnist.test.images,labels:mnist.test.labels})
        test_writer.add_summary(summary_result, step)
        test_writer.add_run_metadata(tf.RunMetadata(), 'step%03d' % step)
        print("test accuracy %f at step %d" % (acc, step))

print("Accuracy %f" % sess.run(accuracy, feed_dict={images:mnist.test.images,
                                                    labels:mnist.test.labels}))
train_writer.close()
test_writer.close()


# The next three imports help with compatability between
# Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pylab
import tensorflow as tf

# # A special command for IPython Notebooks that
# # intructs Matplotlib to display plots in the notebook
# %matplotlib inline

# This is a directory we'll use to store information
# about the graph to later visualize in TensorBoard.
# By default, it will be created in the same directory
# as this notebook. 

# Be sure to delete the contents of this directory before
# running the script.
LOGDIR = './graphs'

tf.reset_default_graph()
sess = tf.Session()

# This function will create a noisy dataset that's roughly linear, according to the equation *y = mx + b + noise*

def make_noisy_data(m=0.1, b=0.3, n=100):
    x = np.random.rand(n).astype(np.float32)
    noise = np.random.normal(scale=0.01, size=len(x))
    y = m*x + b + noise
    return x,y

# Create training and testing data.
x_train, y_train = make_noisy_data()
x_test, y_test = make_noisy_data()

# Plot our training and testing data 
pylab.plot(x_train, y_train, 'b.')
pylab.plot(x_test, y_test, 'g.')

# tf.name_scope is used to make a graph legible in the TensorBoard graph explorer
# shape=[None] means x_placeholder is a one dimensional array of any length. 
# name='x' gives TensorBoard a display name for this node.
with tf.name_scope('input'):
    x_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='x-input')
    y_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name='y-input')

print(x_placeholder)
print(y_placeholder)

# Define our model.
# Here, we'll use a linear model: y = mx + b
with tf.name_scope('model'):
    m = tf.Variable(tf.random_normal([1]),name='m')
    b = tf.Variable(tf.random_normal([1]),name='b')
    # This is the same as y = tf.add(tf.mul(m, x_placeholder), b), but looks nicer
    y = m * x_placeholder + b

print(m)
print(b)
print(y)

# Define a loss function (here, squared error) and an optimizer (here, gradient descent).
LEARNING_RATE = 0.5

with tf.name_scope('training'):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y - y_placeholder))
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train = optimizer.minimize(loss)

print("loss:", loss)
print("optimizer:", optimizer)
print("train_step:", train)

# setup tensorboard
# Write the graph
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)

# Attach summaries to Tensors (for TensorBoard visualization)
tf.summary.histogram('m', m)
tf.summary.histogram('b', b)
tf.summary.scalar('loss', loss)

# This op will calculate our summary data when run
summary_op = tf.summary.merge_all()

# 6)Initialize variables
sess.run(tf.global_variables_initializer())

# 7)Training
TRAIN_STEPS = 201

for step in range(TRAIN_STEPS):
    
    # Session will run two ops:
    # - summary_op prepares summary data we'll write to disk in a moment
    # - train will use the optimizer to adjust our variables to reduce loss
    summary_result, _ = sess.run([summary_op, train], feed_dict = {x_placeholder:x_train,
                                                                   y_placeholder:y_train})
    #write summary data to disk
    writer.add_summary(summary_result, step)
    #if step % 20 == 0:
    print(step, sess.run([m,b]))

#close the writer when we're finished using it
writer.close()

#See the trained values for m and b
print("m: %f, b: %f" % (sess.run(m),sess.run(b)))

#Use the trained model to make a prediction
# Remember that x_placeholder must be a vector, hence [2] not just 2.
# We expect the result to be (about): 2 * 0.1 + 0.3 + noise ~= 0.5
sess.run(y, feed_dict={x_placeholder: [2]})
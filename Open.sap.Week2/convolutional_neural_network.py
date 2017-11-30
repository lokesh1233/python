from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import tensorflow as tf

# We'll use Keras (included with TensorFlow) to import the data
# I figured I'd do all the preprocessing and reshaping here, 
# rather than in the model.
(x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

# Normalize the color values to 0-1
# (as imported, they're 0-255)
x_train /= 255
x_test /= 255

# The CNN we'll use later expects a color channel dimension
# Let's add this here
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Convert to one-hot.
y_train = tf.contrib.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.contrib.keras.utils.to_categorical(y_test, num_classes=10)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


#This function that defines our CNN.
def build_cnn(features, mode):
    
    image_batch = features['x']
    
    with tf.name_scope("conv1"):  
        conv1 = tf.layers.conv2d(inputs=image_batch, filters=32, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.relu)

    with tf.name_scope("pool1"):  
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    with tf.name_scope("conv2"):  
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                                 padding='same', activation=tf.nn.relu)

    with tf.name_scope("pool2"):  
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    with tf.name_scope("dense"):  
        # The 'images' are now 7x7 (28 / 2 / 2), and we have 64 channels per image
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

    with tf.name_scope("dropout"):  
        # Add dropout operation; 0.8 probability that a neuron will be kept
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.2, training = mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits


#To write a Custom Estimator we'll specify our own model function. Here, we'll use tf.layers to replicate the model from the third notebook.
def model_fn(features, labels, mode):
    
    logits = build_cnn(features, mode)
    
    # Generate Predictions
    classes = tf.argmax(logits, axis=1)
    predictions = {
        'classes': classes,
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return an EstimatorSpec for prediction
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
    # Compute the loss, per usual.
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)
        
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        # Configure the Training Op
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=1e-3,
            optimizer='Adam')

        # Return an EstimatorSpec for training
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                      loss=loss, train_op=train_op)    

    assert mode == tf.estimator.ModeKeys.EVAL
    
    # Configure the accuracy metric for evaluation
    metrics = {'accuracy': tf.metrics.accuracy(classes, tf.argmax(labels, axis=1))}
    
    return tf.estimator.EstimatorSpec(mode=mode, 
                                      predictions=predictions, 
                                      loss=loss,
                                      eval_metric_ops=metrics)
    
# Input functions, as before.
train_input = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train},
    y_train, 
    num_epochs=None, # repeat forever
    shuffle=True # 
)

test_input = tf.estimator.inputs.numpy_input_fn(
    {'x': x_test},
    y_test,
    num_epochs=1, # loop through the dataset once
    shuffle=False # don't shuffle the test data
)

estimator = tf.estimator.Estimator(model_fn=model_fn)

# If you are running on a machine without a GPU, this can take some time to train.
estimator.train(input_fn=train_input, steps=2000)

# Evaluate the estimator using our input function.
# We should see our accuracy metric below
# Tweaking with the params of the model, you can get >99% accuracy
evaluation = estimator.evaluate(input_fn=test_input)
print(evaluation)

# Here's how to print predictions on a few examples
MAX_TO_PRINT = 5

# This returns a generator object
predictions = estimator.predict(input_fn=test_input)
i = 0
for p in predictions:
    true_label = np.argmax(y_test[i])
    predicted_label = p['classes']
    print("Example %d. True: %d, Predicted: %s" % (i, true_label, predicted_label))
    i += 1
    if i == MAX_TO_PRINT: break
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

# Flatten 28x28 images to (784,)
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Convert to one-hot.
y_train = tf.contrib.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.contrib.keras.utils.to_categorical(y_test, num_classes=10)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Number of neurons in each hidden layer
HIDDEN1_SIZE = 500
HIDDEN2_SIZE = 250

# To write a Custom Estimator we'll specify our own model function. Here, 
# we'll use tf.layers to replicate the model from the third notebook.
def model_fn(features, labels, mode):
    
    # First we'll create 2 fully-connected layers, with ReLU activations.
    # Notice we're retrieving the 'x' feature (we'll provide this in the input function
    # in a moment).
    fc1 = tf.layers.dense(features['x'], HIDDEN1_SIZE, activation=tf.nn.relu, name="fc1")
    fc2 = tf.layers.dense(fc1, HIDDEN2_SIZE, activation=tf.nn.relu, name="fc2")
    
    # Add dropout operation; 0.9 probability that a neuron will be kept
    dropout = tf.layers.dropout(
        inputs=fc2, rate=0.1, training = mode == tf.estimator.ModeKeys.TRAIN, name="dropout")

    # Finally, we'll calculate logits. This will be
    # the input to our Softmax function. Notice we 
    # don't apply an activation at this layer.
    # If you've commented out the dropout layer,
    # switch the input here to 'fc2'.
    logits = tf.layers.dense(dropout, units=10, name="logits")
    
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

# At this point, our Estimator will work just like a canned one.
estimator = tf.estimator.Estimator(model_fn=model_fn)

# Train the estimator using our input function.
estimator.train(input_fn=train_input, steps=2000)

# Evaluate the estimator using our input function.
# We should see our accuracy metric below
evaluation = estimator.evaluate(input_fn=test_input)
print(evaluation)

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
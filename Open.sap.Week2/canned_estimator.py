from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# We'll use Keras (included with TensorFlow) to import the data
(x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

# Normalize the color values to 0-1
# (as imported, they're 0-255)
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

train_input = tf.estimator.inputs.numpy_input_fn({'x':x_train},
                                                  y_train, 
                                                  num_epochs=None,#repeat forever
                                                  shuffle = True)

test_input = tf.estimator.inputs.numpy_input_fn(
    {'x': x_test},
    y_test,
    num_epochs=1, # loop through the dataset once
    shuffle=False # don't shuffle the test data
)

# define the features for our model
# the names must match the input function
feature_spec = [tf.feature_column.numeric_column('x', shape = 784)]

estimator = tf.estimator.LinearClassifier(feature_spec, 
                                          n_classes=10,
                                          model_dir="./graphs/canned/linear")

# I've arbitrarily decided for 1000 steps
estimator.train(train_input, steps=1000)

# We should see about 90% accuracy here
evaluation = estimator.evaluate(input_fn = test_input)
print(evaluation)

MAX_TO_PRINT = 5
# this returns a generator object
predictions = estimator.predict(input_fn = test_input)
i=0
for p in predictions:
    true_label = y_test[i]
    predicted_labe = p['class_ids'][0]
    print("Example %d. True: %d, Prdicted: %d " % (i, true_label, predicted_labe))
    i += 1
    if i == MAX_TO_PRINT: break
    
# Here's how easy it is to switch the model to a fully connected DNN.
estimator = tf.estimator.DNNClassifier(
    hidden_units=[256], #we will arbitrarily use two layers
    feature_columns=feature_spec,
    n_classes=10,
    model_dir="./graphs/canned/deep")

# I've arbitrarily decided to train for 2000 steps
estimator.train(train_input, steps=2000)

# Expect accuracy around 97%
evaluation = estimator.evaluate(input_fn=test_input)
print(evaluation)
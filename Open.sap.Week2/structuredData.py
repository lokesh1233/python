from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pandas as pd
    
from IPython.display import Image

import tensorflow as tf
print('This code requires TensorFlow v1.3+')
print('You have:', tf.__version__)

#Image(filename='../images/facets1.jpg', width=500)

census_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
census_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
census_train_path = tf.contrib.keras.utils.get_file('census.train', census_train_url)
census_test_path = tf.contrib.keras.utils.get_file('census.test', census_test_url)

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
                 'hours-per-week', 'native-country', 'income']

# notes
# 1) we provide the header from above.
# 2) the test file has a line we want to disgard at the top, so we include the parameter 'skiprows=1' 

census_train = pd.read_csv(census_train_path, index_col=False, names=column_names)
census_test = pd.read_csv(census_test_path, skiprows=1, index_col=False, names=column_names)

# Drop any rows that have missing elements
# Of course there are others ways to handle missing data, but we'll
# take the simplest approach
census_train = census_train.dropna(axis=0, how="any")
census_test = census_test.dropna(axis=0, how="any")

# Separate the label we want to predict into its own object
# At the same time we will convert it into true/false to fix the formatting error 
census_train_label = census_train.pop('income').apply(lambda x: ">50K" in x)
census_test_label = census_test.pop('income').apply(lambda x: ">50K" in x)

# I find it useful to print out the shape of the data as I go, as a sanity check.
print('Training examples: %d' % census_train.shape[0])
print('Training labels: %d' % census_train_label.shape[0])
print()
print('Test Examples: %d' % census_test.shape[0])
print('Test labels: %d' % census_test_label.shape[0])

#Likewise, you could do a spot check of the testing examples and labels. 
census_train.head(5)
census_train_label.head(5)

# Estimators and input functions 
# input functions for training and testing data 
def create_train_input_fn():
    return tf.estimator.inputs.pandas_input_fn(
        x=census_train,
        y=census_train_label,
        batch_size=32,
        num_epochs=None, # Repeat forever
        shuffle=True)  # shuffle train data 

def create_test_input_fn():
    return tf.estimator.inputs.pandas_input_fn(
        x = census_test,
        y = census_test_label,
        num_epochs=1,  # just one epoch
        shuffle=False)  # Don't shuffle so we can compare to census_test_labels later

# A list of the feature columns we'll use to train the linear model
feature_columns = []
# To start, we'll use the raw, numeric value of age.
age = tf.feature_column.numeric_column('age')
feature_columns.append(age)

age_buckets = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('age'),
     boundaries = [31, 46, 60, 75, 90] # specify the ranges
     )
feature_columns.append(age_buckets)

# you can also evenly divide the data, if you prefer not to specify the ranges yourself.
# age_buckets = tf.feature_column.bucketized_column(
#     tf.feature_column.numeric_column('age'),
#       list(range(10))
#)

# Here's a categorial column
# We're specifying the possible values
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
                   "Assoc-voc", "7th-8th", "Doctorate", "Prof-school", "5th-6th",
                    "10th", "1st-4th", "Preschool", "12th"
                    ])
feature_columns.append(education)

# A categorical feature with a possibly large number of values
# and the vocabulary not specified in advance.
native_country = tf.feature_column.categorical_column_with_hash_bucket('native-country', 1000)
feature_columns.append(native_country)

age_cross_education = tf.feature_column.crossed_column(
    [age_buckets, education],
     hash_bucket_size = int(1e4) # Using a hash is handy here 
     )
feature_columns.append(age_cross_education)

# Train a Canned Linear estimator
train_input_fn = create_train_input_fn()
estimator = tf.estimator.LinearClassifier(feature_columns, model_dir='graphs/linear',n_classes=2)
estimator.train(train_input_fn, steps=1000)

# Evaluate
test_input_fn = create_test_input_fn()
estimator.evaluate(test_input_fn)

# Test
# The Estimator returns a generator object. this bit of code demonstrates how to retrieve predictions for individual examples
# reinitialize the input function 
test_input_fn = create_test_input_fn()
predictions = estimator.predict(test_input_fn)
i=0
for prediction in predictions:
    true_label = census_test_label[i]
    predicted_label = prediction['class_ids'][0] 
    # uncomment the following line to see probabilities for individual classes
    print(prediction)
    print('Example %d. Actual %d, predicted %d' % (i, true_label, predicted_label))
    i += 1
    if i == 5: break

# Train a Deep model 
# we'll provide vocabulary lists for features with just a few terms
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass',
    [' Self-emp-not-inc', ' Private', ' State-gov', ' Federal-gov', ' Local-gov',
     ' ?', ' Self-emp-inc', ' Without-pay', ' Never-worked'])

education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education',
    [' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th', ' Some-college', 
     ' Assoc-acdm', ' Assoc-voc', ' 7th-8th', ' Doctorate', ' Prof-scholl',
     ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital-status',
    [' Married-civ-spouse', ' Divorced', ' Married-spouse-absent',
     ' Never-married', ' Separated', ' Married-AF-spouse', ' Widowed'])
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    ' relationship',
    [' Husband', ' Not-in-family', ' Wife', ' Own-child', ' Unmarried',
     ' Other-relative'])

feature_columns = [
    # use indicator columns for low dimensional vocabularies
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(marital_status),
    #tf.feature_column.indicator_column(relationship),
    
    # use embedding for high dimensional vocabularies
    tf.feature_column.embedding_column(  # now using embedding!
        # params are hash buckets, embedding size
        tf.feature_column.categorical_column_with_hash_bucket('occupation', 100), 10),
    
    # numeric features
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('education-num'),
    tf.feature_column.numeric_column('capital-gain'),
    tf.feature_column.numeric_column('capital-loss'),
    tf.feature_column.numeric_column('hours-per-week'),             
]

estimator = tf.estimator.DNNClassifier(hidden_units=[256, 128, 64], 
                                       feature_columns=feature_columns,
                                       n_classes=2,
                                       model_dir='graphs/dnn')

train_input_fn = create_train_input_fn()
estimator.train(train_input_fn, steps=2000)

test_input_fn = create_test_input_fn()
estimator.evaluate(test_input_fn)

# thats a little better 
# reinitialize the input function
test_input_fn = create_test_input_fn()

predictions = estimator.predict(test_input_fn)
i = 0
for prediction in predictions:
    true_label = census_test_label[i]
    predicted_label = prediction['class_ids'][0]
    # Uncomment the following line to see probabilities for individual classes
    # print(prediction) 
    print("Example %d. Actual: %d, Predicted: %d" % (i, true_label, predicted_label))
    i += 1
    if i == 5: break
# create data/fashion folder, if it doesn't exist
import os
if not os.path.exists("data/fashion"):
    os.makedirs("data/fashion")
    
# Download the labels from the Fashion MNIST data
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz --directory-prefix=./data/fashion/
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz --directory-prefix=./data/fashion/
# Download the images from the Fashion MNIST data
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz --directory-prefix=./data/fashion/
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz --directory-prefix=./data/fashion/

from tensorflow.examples.tutorials.mnist import input_data
fash_data = input_data.read_data_sets('data/fashion')
# Train and test images
X_train = fash_data.train.images
X_test = fash_data.test.images


# Train and test labels
y_train = fash_data.train.labels.astype("int")
y_test = fash_data.test.labels.astype("int")

# view some input data from the dataset
from matplotlib import pyplot as plt
from random import randint
import numpy as np

def gen_image(arr, im_title):
    image_data = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.title(im_title)
    plt.imshow(image_data, cmap='gray', interpolation='nearest')
    return plt

fash_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
    }

# produces a new image for each run
im_index = randint(0, 100)
batch_X, batch_Y = fash_data.test.next_batch(100)
gen_image(batch_X[im_index], fash_labels[batch_Y[im_index]]).show()


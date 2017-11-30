import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt, style
from tensorflow.examples.tutorials.mnist import input_data
fash_data = input_data.read_data_sets('data/fashion')

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


num_inputs = 28*28  # Fashion MNIST
num_outputs = 10

num_hidden_1 = 300
num_hidden_2 = 100
num_hidden_3 = 100

# reset graphs and set seed to produce reproducible result across runs
def reset_graphs(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed()

reset_graphs(42)

X = tf.placeholder(tf.float32, shape=(None, num_inputs), name="X")
Y = tf.placeholder(tf.int64, shape=(None), name="Y")

# define a hidden layer method that crates a name scope in the TF graph
def hidden_layer(X, num_neurons, name, activation=None):
    # Use a name scope clearly separate the layers on TensorBoard
    with tf.name_scope(name):
        num_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(num_inputs)
        # Initialize the weights with truncated normal data and standard deviation from input data
        init = tf.truncated_normal((num_inputs, num_neurons), stddev=stddev)
        # weights and bias
        W = tf.Variable(init, name="weights")
        B = tf.Variable(tf.zeros([num_neurons]), name='bias')
        # Output
        Z = tf.matmul(X, W)+B
        # Support for returning values with and without non-linearity
        if activation is not None:
            Z = activation(Z)
        
        return Z

# Defining a one-layer neural network with no non-linearities
with tf.name_scope("onelayer_nn"):
    hidden_layer_1 = hidden_layer(X, num_neurons=num_hidden_1, name="hidden_layer_1", activation=None)
    logits = hidden_layer(hidden_layer_1, num_outputs, name="outputs")
    
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, Y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Define the initialization and saver for saving our model to local file system
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# Define the number of epochs and batch size
num_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(num_epochs):
        # train the network by feeding in batch_size examples for each iteration
        for iteration in range(fash_data.train.num_examples // batch_size):
            x_batch, y_batch = fash_data.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:x_batch,Y:y_batch})
            
        # Evaluate test data after every epoch
        acc_train = accuracy.eval(feed_dict={X:x_batch,Y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:fash_data.test.images,Y:fash_data.test.labels})
        # Log some results
        print("Epoch:", epoch, "Train accuracy", acc_train, "Test accuracy", acc_test)
        
    save_path = saver.save(sess, "./single_layer_model.ckpt")

# Evaluate the single-layer neural network
with tf.Session() as sess:
    saver.restore(sess, "./single_layer_model.ckpt")
    X_new_scalled = fash_data.test.images[:20]
    Z = logits.eval(feed_dict={X: X_new_scalled})
    y_pred = np.argmax(Z, axis=1)

for im_index in range(5):
    gen_image(fash_data.test.images[im_index],
              "Predicted: %s Actuval: %s" % (fash_labels[y_pred[im_index]],
                                             fash_labels[fash_data.test.labels[im_index]])).show()

# Defining a one layer neural network with ReLU non-linearity
with tf.name_scope("onelayer_nn_relu"):
    hidden_layer_1 = hidden_layer(X, num_neurons = num_hidden_1, name="hidden_layer_1", activation=tf.nn.relu)
    logits = hidden_layer(hidden_layer_1, num_outputs, name="outputs")
    
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, Y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Define the initialization and saver for saving our model to local file system
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Define the number of epochs and batch size
num_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(num_epochs):
        # Train the network by feeding in batch_size example for each iteration
        for iteration in range(fash_data.train.num_examples // batch_size):
            x_batch, y_batch = fash_data.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:x_batch,Y:y_batch})
        
        # Evaluate test data after every epoch
        acc_train = accuracy.eval(feed_dict={X:x_batch,Y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:fash_data.test.images,Y:fash_data.test.labels})
        
        # Log some results
        print("Epoch: ", epoch, "Train accuracy: ", acc_train, "Test accuracy", acc_test)
        
    save_path = saver.save(sess, "./single_layer_model_with_relu.ckpt")

# Evaluate the single layer network with relu
with tf.Session() as sess:
    saver.restore(sess, "./single_layer_model_with_relu.ckpt")
    X_new_scaled = fash_data.test.images[:20]
    Z = logits.eval(feed_dict = {X:X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
    
for im_index in range(5):
    gen_image(fash_data.test.images[im_index],
               "Predicted: %s Actual: %s" % (fash_labels[y_pred[im_index]], 
                                           fash_labels[fash_data.test.labels[im_index]])).show()

 
# defining a three layer neural network with ReLU non-linearity
with tf.name_scope("threelayer_nn_relu"):
    hidden_layer_1 = hidden_layer(X, num_neurons = num_hidden_1, name="hidden_layer_1", activation=tf.nn.relu)
    hidden_layer_2 = hidden_layer(hidden_layer_1, num_neurons = num_hidden_2, name="hidden_layer_2", activation=tf.nn.relu)
    hidden_layer_3 = hidden_layer(hidden_layer_2, num_neurons = num_hidden_3, name="hidden_layer_3", activation=tf.nn.relu)
    logits = hidden_layer(hidden_layer_3, num_outputs, name="outputs")
    
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, Y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Define the initialization and saver for saving our model to local file system
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Define the number of epochs and batch size
num_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(num_epochs):
        # Train the network by feeding in batch_size example for each iteration
        for iteration in range(fash_data.train.num_examples // batch_size):
            x_batch, y_batch = fash_data.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:x_batch,Y:y_batch})
        
        # Evaluate test data after every epoch
        acc_train = accuracy.eval(feed_dict={X:x_batch,Y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:fash_data.test.images,Y:fash_data.test.labels})
        
        # Log some results
        print("Epoch: ", epoch, "Train accuracy: ", acc_train, "Test accuracy", acc_test)
        
    save_path = saver.save(sess, "./three_layer_model_with_relu.ckpt")

# Evaluate the two layer network with relu
with tf.Session() as sess:
    saver.resotre(sess, "./three_layer_model_with_relu.ckpt")
    X_new_scaled = fash_data.test.images[:20]
    Z = logits.eval(feed_dict={X:X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
    
for im_index in range(5):
    gen_image(fash_data.test.images[im_index],
              "Predicted: %s Actuval: %s" % (fash_labels[y_pred[im_index]],
                                             fash_labels[fash_data.test.labels[im_index]])).show()


# Graph visualization
from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <strip>
        function load(){{
            document.getElementById("{id}").pbtxt = {data};
        }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
         <tf-graph-basic id="{id}"></tf-graph-basic>
         </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
        
# show the graph as output below 
show_graph(tf.get_default_graph())
import tensorflow as tf
print('your running tensorflow %s' % str(tf.__version__))
sess = tf.InteractiveSession()
a = tf.constant(23, name="const_a")
b = tf.constant(34, name="cons_b")

c = tf.add(a,b)
print(c.eval())

c = tf.subtract(a, b)
print(c.eval())

c = tf.multiply(a, b)
print(c.eval())

c = tf.divide(a, b)
print(c.eval())

print(tf.mod(a,b).eval())

x = tf.constant(2.,name="const_a")
y = tf.constant(10., name="const_b")

# lets calculate 2^10
print(tf.pow(x, y).eval())
# x < y
print(tf.less(x, y).eval())
# x <= y
print(tf.less_equal(x, y).eval())
# x > y
print(tf.greater(x, y))
# x >= y
print(tf.greater_equal(x, y).eval())
#conditional
print(tf.logical_and(True,False))

print(tf.logical_or(True,False))

print(tf.logical_xor(True,False))

print(tf.logical_not(True))

# matrix multiplication 3*1 and 1*3
mat_a = tf.constant([[1.,3.,5.]],name='mat_a')
mat_b = tf.constant([[2.],[6.],[2.]],name='mat_b')

# lets matrix multiply the two matrices
prod_op = tf.matmul(mat_a, mat_b)
print(prod_op)

# create a session object to run our matrix multiplication
sess = tf.Session()
# returns numpy array object
print(sess.run(prod_op))
# remember to close the session when done, releases the resources
# familiar 'with' block as follows 
with tf.Session() as sess:
    print(sess.run(prod_op))

# if you have multiple devices capable of computes as follows
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        print(sess.run(prod_op))

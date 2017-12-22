import tensorflow as tf

####### Constant ############
# Create TensorFlow object called hello_constant
# hello_constant is a 0-dimensional string tensor
hello_constant = tf.constant('Hello World!')
#tf.constant()   operation returns constatn tensor, because the value never changes
#A is a 0-d int32 tensor
A = tf.constant(1234)
#B is a 1-d int32 tensor
B = tf.constant([123, 456, 789])
#C is a 2-d int32 tensor
C = tf.constant([[123, 456, 789],[222,333,444]])

#creates a session instance, sess, using tf.Session
with tf.Session() as sess:
    # Run the tf.constant operation in the session
    # sess.run() function evaluates the tensor and returns the result
    output = sess.run(hello_constant)
    print(output)

####### Tensor folow input ############
x = tf.placeholder(tf.string)
with tf.Session() as sess:
    # use feed_dict parameter in tf.session.run() to set the placeholder tensor
    output = sess.run(x, feed_dict={x:'Hello World'})

# set more than one tensor using feed_dict
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run([x, y, z], feed_dict={x:'Test String', y:123, z:45.67})
    print(output)
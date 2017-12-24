import tensorflow as tf

# math
x = tf.add(5,2)#7
print (x)

y = tf.subtract(10,4)#6
m = tf.multiply(2,5)#10
d = tf.divide(tf.cast(tf.constant(6),tf.float64), tf.cast(tf.constant(3),tf.float64))

with tf.Session() as sess:
    output = sess.run([x,y,m,d])
    print (output)

# converting types
#subtract(x1, x2), x1,x2 need to be the same type
a = tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))
with tf.Session() as sess:
    output = sess.run(a)
    print(output)


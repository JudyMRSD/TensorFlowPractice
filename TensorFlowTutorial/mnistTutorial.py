#https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py
#MNIST for ML beginners

from __future__ import absolute_import
# https://www.python.org/dev/peps/pep-0238/
#The future division statement, spelled from __future__ import division, will change the / operator to mean true division throughout the module
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.example.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def main(_):
    #import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

    #Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss an optimizer
    y_ = tf.placeholder(tf.float32, [None,  10])

    # calculate cross entropy

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    # take mean here because this is a mini-batch, need to average over number of training examples
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.


    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits = y))
    #minimize cross entropy using the gradient descent algorithm with learning rate 0.5
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    #Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

    #test trained model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        #information for help
        parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                            help='Directory for storing input data')
        #Sometimes a script may only parse a few of the command-line arguments, passing the remaining arguments on to
        # another script or program.
        # In these cases, the parse_known_args() method can be useful.
        # It works much like parse_args() except that it does not produce an error when extra arguments are present.
        FLAGS, unparsed = parser.parse_known_args()
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


















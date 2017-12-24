#https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py
#MNIST for ML beginners

# input_data imports Functions for downloading and reading MNIST data.
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#an image of a handwritten digit and a corresponding label
# image x: flatten this array into a vector of 28x28 = 784 numbers. mnist.train.images
# label y: labels     mnist.train.labels


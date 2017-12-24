# First start to train a model
import tensorflow as tf
import math
# Remove previous Tensors and Operations
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

learning_rate = 0.001
n_input = 784 # MNIST data input (image shape 28*28)
n_classes = 10 # MNIST total classes( 0- 9 digits)
print("import data")
# Import MNIST data
mnist = input_data.read_data_sets('.', one_hot = True)
print("feature, label")
# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
print("weights, bias")
# Weights and bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))
print("Logits  xW + b")
# Logits  xW + b
output = tf.add(tf.matmul(features, weights), bias)
print("cost")
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
print("accuracy")
# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels,1))
# reduced_mean output type is the same as the input type, thus need cast there
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("start training")

# train the model and save weights

save_file = './train_model.ckpt'
batch_size = 128
n_epochs = 100
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Training cycle
    for epoch in range(n_epochs):
        # math.ceil():  return the smallest integer value greater than or equal to x.
        total_batch = math.ceil(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict = {features: batch_features, labels: batch_labels})
        #print status for every 10 epochs
        if epoch%10 == 0:
            valid_accuracy = sess.run(accuracy, feed_dict = {features: mnist.validation.images , labels:mnist.validation.labels})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))
    #save the model
    saver.save(sess, save_file)
    print ('Trained Model Saved')


# Load a Trained Model
# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)
    test_accuracy = sess.run(accuracy, feed_dict = {features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))






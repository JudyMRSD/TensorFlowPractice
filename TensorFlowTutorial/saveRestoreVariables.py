import tensorflow as tf


def saveVariables():
    # The file path to save the data
    save_file = './model.ckpt'
    # Two Tensor Variables: weights and bias
    # weights adn bias are set to random values using the tf.truncated_normal()
    weights = tf.Variable(tf.truncated_normal([2,3]))
    bias = tf.Variable(tf.truncated_normal([3]))
    # Class used to save and/or restore Tensor Variables
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize all the Variables
        sess.run(tf.global_varialbes_initializer())
        # Show the values of weights and bias
        print('Weights:')
        print(sess.run(weights))
        print('Bias:')
        print(sess.run(bias))
        # Save the model
        # the values are then saved to the save_file location "model.ckpt" using tf.train.Saver.save()
        saver.save(sess, save_file)

def loadVariables():
    # Remove the previous weights and bias
    tf.reset_default_graph()
    #two variables
    weights = tf.Variable(tf.truncated_normal([2,3]))
    bias = tf.Variable(tf.truncated_normal([3]))
    # Class used to save and/or restore Tensor Variables
    saver = tf.train.Saver()

    save_file = './model.ckpt'

    with tf.Session() as sess:
        # Load the weights and bias
        saver.restore(sess, save_file)
        # Show the values of weights and bias
        print('Weight:')
        print(sess.run(weights))
        print('Bias:')
        print(sess.run(bias))




saveVariables()
loadVariables()












import numpy as np

print("binomial")
print(np.random.binomial(5, 0.3, (10,10)))
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).T
#learning rate
alpha = 0.5
hidden_dim = 4
# probability that one node will be turned off
# https://iamtrask.github.io/2015/07/28/dropout/
# hidden layer: start as 50%
# input layer dropout shouldn't exceed 25%
dropout_percent = 0.2
do_dropout = True
# initialize weights
# np.random.random:  Return random floats in the half-open interval [0.0, 1.0).
# random numbers between -1 and 1 as weights
#2*(0 to 1) -1 gives (0 to 2) - 1, which is -1 to 1
# W_0:  3xhidden_dim, 3 is number of features in the input
W_0 = 2*np.random.random((3,hidden_dim))-1
print("W_0: ")
print(W_0)
# W_0:  hidden_dim x 1
W_1 = 2*np.random.random((hidden_dim, 1)) -1
print("W_1: ")
print(W_1)
# do 100 iterations of forward and backward propagations
for j in range(2):
    layer_1 = (1/(1+np.exp(-(np.dot(X,W_0)))))
    print ("layer_1")
    print (layer_1)
    # only do dropout during training , not testing
    # to perform dropout on a layer , you randomly set some of the layer's values to 0 during forward propagation
    if(do_dropout):

        layer_1 *= np.random.binomial([np.ones((len(X), hidden_dim))], 1 - dropout_percent)[0] * (1.0 / (1 - dropout_percent))
        print("new layer_1")
        print(layer_1)
    layer_2 = 1 / (1 + np.exp(-(np.dot(layer_1, W_1))))
    layer_2_delta = (layer_2 - y) * (layer_2 * (1 - layer_2))
    layer_1_delta = layer_2_delta.dot(W_1.T) * (layer_1 * (1 - layer_1))
    W_1 -= (alpha * layer_1.T.dot(layer_2_delta))
    W_0 -= (alpha * X.T.dot(layer_1_delta))
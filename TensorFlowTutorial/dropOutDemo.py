# dropout without using tensorflow
import numpy as np

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
truth = np.array([[0, 1, 1, 0]]).T

learningRate = 0.5
hidden_dim = 4


dropout_percent = 0.2
# success percent
sp = 1- dropout_percent
do_dropout = True
# initialize weights to [-1,1)         2*[0,1)-1 = [-1,1)
W_0 = 2*np.random.random((3,hidden_dim))-1
W_1 = 2*np.random.random((hidden_dim, 1)) -1

for j in range(2):
    layer_1 = (1/(1+np.exp(-(np.dot(X,W_0)))))
    # forward
    if(do_dropout):
        u_1 = np.random.binomial([np.ones((len(X), hidden_dim))], sp)[0] * (1.0 / sp)
        layer_1 *= u_1;
        print("new layer_1")
        print(layer_1)
    layer_2 = 1 / (1 + np.exp(-(np.dot(layer_1, W_1))))

    layer_2_delta = (layer_2 - truth) * (layer_2 * (1 - layer_2))
    layer_1_delta = layer_2_delta.dot(W_1.T) * (layer_1 * (1 - layer_1))
    #back prop
    W_1 -= (learningRate * layer_1.T.dot(layer_2_delta))
    W_0 -= (learningRate * X.T.dot(layer_1_delta))
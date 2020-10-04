import tensorflow as tf
import numpy as np

class DNN:
    def __init__(self, X, layer_size, Xmin, Xmax):
        self.X = X
        self.size = layer_size
        self.Xmin = Xmin
        self.Xmax = Xmax
    
    def hyper_initial(self):
        L = len(self.size)
        W = []
        b = []
        for l in range(1, L):
            in_dim = self.size[l-1]
            out_dim = self.size[l]
            std = np.sqrt(2/(in_dim + out_dim))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            W.append(weight)
            b.append(bias)

        return W, b

    def fnn(self, W, b):
        A = 2.0*(self.X - self.Xmin)/(self.Xmax - self.Xmin) - 1.0
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y

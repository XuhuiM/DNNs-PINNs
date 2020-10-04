import tensorflow as tf
import numpy as np

class DNN:
    def __init__(self, layer_size, Xmin, Xmax):
        self.size = layer_size
        self.Xmin = Xmin
        self.Xmax = Xmax
    
    def hyper_initial(self):
        L = len(self.size)
        Weights = []
        Biases = []
        for l in range(1, L):
            in_dim = self.size[l-1]
            out_dim = self.size[l]
            std = np.sqrt(2/(in_dim + out_dim))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            Weights.append(weight)
            Biases.append(bias)

        return Weights, Biases

    def fnn(self, X, W, b):
        A = 2.0*(X - self.Xmin)/(self.Xmax - self.Xmin) - 1.0
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y

    def pdenn(self, x, t, W, b, nu):
        u = self.fnn(tf.concat([x, t], 1), W, b)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u*u_x - nu*u_xx

        return f

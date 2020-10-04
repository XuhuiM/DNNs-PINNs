'''
PINNs for 2D Burgers equation
@Author: Xuhui Meng
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow.compat.v1 as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from dataset import Dataset
from net import DNN
from modeltrain import Train
from saveplot import SavePlot

np.random.seed(1234)
tf.set_random_seed(1234)

def main():
    t_range = [0, 1]
    x_range = [-1, 1]
    NX = 256
    NT = 101
    N_train = 15000
    N_bc = 80
    N_initial = 200

    data = Dataset(x_range, t_range, NX, NT, N_train, N_bc, N_initial)
    #inputdata
    X, X_initial, u_initial, X_bc_0, u_bc_0, X_bc_1, u_bc_1, Xmin, Xmax = data.build_data()

    #size of the DNN
    layers = [2] + 3*[20] + [1]

    x_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    t_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    x_initial_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    t_initial_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    u_initial_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    x_0_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    x_1_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    t_bc_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    
    u_0_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    u_1_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    
    #physics-infromed neural networks
    pinn = DNN(layers, Xmin, Xmax)
    W, b = pinn.hyper_initial()
    u_pred = pinn.fnn(tf.concat([x_train, t_train], 1), W, b)
    u_initial_pred = pinn.fnn(tf.concat([x_initial_train, t_initial_train], 1), W, b)
    u_0_pred = pinn.fnn(tf.concat([x_0_train, t_bc_train], 1), W, b)
    u_1_pred = pinn.fnn(tf.concat([x_1_train, t_bc_train], 1), W, b)
    nu = 0.01/np.pi
    f_pred = pinn.pdenn(x_train, t_train, W, b, nu)

    loss = tf.reduce_mean(tf.square(f_pred)) + \
           tf.reduce_mean(tf.square(u_initial_train - u_initial_pred)) + \
           tf.reduce_mean(tf.square(u_0_train - u_0_pred)) + \
           tf.reduce_mean(tf.square(u_1_train - u_1_pred))

    train_adam = tf.train.AdamOptimizer().minimize(loss)
    train_lbfgs = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                         method = "L-BFGS-B",
                                                         options = {'maxiter': 50000,
                                                                    'ftol': 1.0*np.finfo(float).eps
                                                                   }
                                                        )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    train_dict = {x_train: X[:, 0:1], t_train: X[:, 1:2], \
                  x_initial_train: X_initial[:, 0:1], t_initial_train: X_initial[:, 1:2], u_initial_train: u_initial,\
                  x_0_train: X_bc_0[:, 0:1], t_bc_train: X_bc_0[:, 1:2], u_0_train: u_bc_0, \
                  x_1_train: X_bc_1[:, 0:1], u_1_train: u_bc_1
                 }
    
    Model = Train(train_dict)
    start_time = time.perf_counter()
    Model.nntrain(sess, u_pred, loss, train_adam, train_lbfgs)
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))

    NX_test = 256
    NT_test = 101
    datasave = SavePlot(sess, x_range, t_range, NX_test, NT_test)
    datasave.saveplt(u_pred, x_train, t_train)

if __name__ == '__main__':
    main()

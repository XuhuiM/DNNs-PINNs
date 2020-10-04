import tensorflow as tf
import numpy as np
import time


class Train:
    def __init__(self, train_dict):
        self.train_dict = train_dict
        '''
        self.sess = sess
        self.u_pred = u_pred
        self.loss = loss
        self.train_adam = train_adam
        self.train_lbfgs = train_lbfgs
        '''
        self.step = 0

    def callback(self, loss_):
        self.step += 1
        if self.step%100 == 0:
            print('Loss: %.3e'%(loss_))

    '''
    def timer(fun):
        start_time = time.perf_counter()
        start_clock = time.process_time()
        fun()
        stop_time = time.perf_counter()
        stop_clock = time.process_time()
        print('Duration time via time() is %f'%((stop_time - start_time)))
        print('Duration time via clock() is %f'%((stop_clock - start_clock)))
    '''

    def nntrain(self, sess, u_pred, loss, train_adam, train_lbfgs):
        n = 0
        nmax = 50000
        loss_c = 1.0e-3
        loss_ = 1.0
        while n < nmax and loss_ > loss_c:
            n += 1
            u_, loss_, _ = sess.run([u_pred, loss, train_adam], feed_dict=self.train_dict)
            if n%100 == 0:
                print('Steps: %d, loss: %.3e'%(n, loss_))

        train_lbfgs.minimize(sess, feed_dict=self.train_dict, fetches=[loss], loss_callback=self.callback)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
savepath='./Output'
if not os.path.exists(savepath):
    os.makedirs(savepath)

class SavePlot:
    def __init__(self, sess, x_range, t_range, NX, NT):
        self.x_range = x_range
        self.t_range = t_range
        self.NX = NX
        self.NT = NT
        self.sess = sess

    def saveplt(self, u_pred, x_train, t_train):
        x_test = np.linspace(self.x_range[0], self.x_range[1], self.NX).reshape((-1, 1))
        t_test = np.linspace(self.t_range[0], self.t_range[1], self.NT).reshape((-1, 1))
        x_t, t_t = np.meshgrid(x_test, t_test)
        x_t = np.reshape(x_t, (-1, 1))
        t_t = np.reshape(t_t, (-1, 1))
        test_dict = {x_train: x_t, t_train: t_t}
        u_test = self.sess.run(u_pred, feed_dict=test_dict)
        u_test = np.reshape(u_test, (t_test.shape[0], x_test.shape[0]))
        u_test = np.transpose(u_test)
        np.savetxt('./Output/u_pred', u_test, fmt='%e')

        plt.imshow(u_test, cmap='rainbow', aspect='auto')
        plt.colorbar()
        plt.savefig('./Output/ucontour.png')
        plt.show()

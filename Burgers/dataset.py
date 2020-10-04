import tensorflow as tf
import numpy as np
from pyDOE import *

class Dataset:
    def __init__(self, x_range, t_range, NX_train, NT_train, N_train, N_bc, N_initial):
        self.x_range = x_range
        self.t_range = t_range
        self.NX = NX_train
        self.NT = NT_train
        self.N_train = N_train
        self.N_bc = N_bc
        self.N_init = N_initial

    def initial(self, x_in):
        u_init = -np.sin(np.pi*x_in)
        u_init = np.reshape(u_init, (-1, 1))

        return u_init

    def bc(self, t_in):
        u_bc0 = np.zeros((t_in.shape[0], 1))
        u_bc1 = np.zeros((t_in.shape[0], 1))

        return u_bc0, u_bc1

    def build_data(self):
        x0 = self.x_range[0]
        x1 = self.x_range[1]
        t0 = self.t_range[0]
        t1 = self.t_range[1]
        X = lhs(2, samples=self.N_train)
        X[:, 0:1] = (x1 - x0)*X[:, 0:1] + x0
        X[:, 1:2] = (t1 - t0)*X[:, 1:2] + t0
        X_input = X
        '''
        t_ = np.linspace(t0, t1, self.NT).reshape((-1, 1))
        x_ = np.linspace(x0, x1, self.NX).reshape((-1, 1))
        x, t = np.meshgrid(x_, t_)
        x = np.reshape(x, (-1, 1))
        t = np.reshape(t, (-1, 1))
        X = np.hstack((x, t))
        x_id = np.random.choice(self.NX*self.NT, self.N_train, replace=False)
        X_input = X[x_id]
        '''

        #initial/bcs
        x = X[:, 0:1]
        t = X[:, 1:2]
        t_0 = t.min(0)
        t_0 = np.reshape(t_0, (-1, 1))
        t_1 = t.max(0)
        t_1 = np.reshape(t_1, (-1, 1))
        x_0 = x.min(0)
        x_0 = np.reshape(x_0, (-1, 1))
        x_1 = x.max(0)
        x_1 = np.reshape(x_1, (-1, 1))

        Xmin = np.hstack((x_0, t_0))
        Xmax = np.hstack((x_1, t_1))
        
        x_ = (x1 - x0)*np.random.rand(self.N_init, 1) + x0
        x_initial, t_initial = np.meshgrid(x_, t_0)
        x_initial = np.reshape(x_initial, (-1, 1))
        t_initial = np.reshape(t_initial, (-1, 1))
        X_initial = np.hstack((x_initial, t_initial))
        u_initial = self.initial(x_initial)

        x_initial_id = np.random.choice(self.NX, self.N_init, replace=False)
        X_initial_input = X_initial #X_initial[x_initial_id]
        u_initial_input = u_initial#[x_initial_id]

        t_ = (t1 - t0)*np.random.rand(self.N_bc, 1) + t0
        x_bc_0, t_bc = np.meshgrid(x_0, t_)
        x_bc_0 = np.reshape(x_bc_0, (-1, 1))
        t_bc = np.reshape(t_bc, (-1, 1))
        X_bc_0 = np.hstack((x_bc_0, t_bc))
        x_bc_1, t_bc = np.meshgrid(x_1, t_)
        x_bc_1 = np.reshape(x_bc_1, (-1, 1))
        X_bc_1 = np.hstack((x_bc_1, t_bc))
        u_bc_0, u_bc_1 = self.bc(t_bc)

        x_bc_id = np.random.choice(self.NT, self.N_bc, replace=False)
        X_bc_0_input = X_bc_0#[x_bc_id]
        u_bc_0_input = u_bc_0#[x_bc_id]
        X_bc_1_input = X_bc_1#[x_bc_id]
        u_bc_1_input = u_bc_1#[x_bc_id]

        return X_input, X_initial_input, u_initial_input, \
               X_bc_0_input, u_bc_0_input, X_bc_1_input, \
               u_bc_1_input, Xmin, Xmax

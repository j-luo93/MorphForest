from __future__ import division, print_function

import numpy as np
from scipy.linalg import norm
import theano
from theano import tensor as T

class Adam(object):

    def __init__(self, lr=1e-1, beta1=0.9, beta2=0.999, stable=1e-8, eps=1.0, max_epoch=1000):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.stable = stable
        self.eps = eps
        self.max_epoch = max_epoch

    # given input (shared variable) and loss function (both symbolic)
    def run(self, params, loss):
        m = theano.shared(np.zeros(params.shape.eval()), borrow=True, name='m')
        v = theano.shared(np.zeros(params.shape.eval()), borrow=True, name='v')
        grad = T.grad(loss, params)
        norm_grad = grad.norm(2)
        m_t = self.beta1 * m + (1 - self.beta1) * grad
        v_t = self.beta2 * v + (1 - self.beta2) * T.pow(grad, 2)
        step = T.iscalar(name='step')
        update_rules = [(params, params - self.lr * (m_t / (1.0 - T.pow(self.beta1, step)) / (T.sqrt(v_t / (1.0 - T.pow(self.beta2, step))) + self.stable))), (m, m_t), (v, v_t)]
        train_epoch = theano.function([step], [loss, norm_grad], updates=update_rules)

        for epoch in xrange(self.max_epoch):
            loss, grad = train_epoch(epoch + 1)
            norm_l2 = norm(grad)
            print("epoch = %d\t loss = %f\t norm = %f" %(epoch + 1, loss, norm_l2), end='')
            print()
            if norm_l2 < self.eps: break

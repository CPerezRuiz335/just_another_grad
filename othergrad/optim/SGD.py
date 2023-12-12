import numpy as np

from othergrad.tensor import Tensor
from othergrad.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(
            self, 
            params, 
            lr=1e-3, 
            momentum=0.,
            weight_decay=0.,
            dampening=0., 
            nesterov=False, 
            maximize=False
        ):
        assert not nesterov or (momentum != 0  and dampening == 0), \
            "Nesterov momentum requires a momentum and zero dampening"
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.dampening, self.nesterov, self.maximize = dampening, nesterov, maximize
        self.b = [np.zeros(t.shape) for t in self.params]

    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    def step(self):
        for t, b in zip(self.params, self.b):
            g = t.grad.copy()

            if self.weight_decay != 0:
                g += self.weight_decay * t.data

            if self.momentum != 0:
                if self.ite > 1:
                    b[:] = self.momentum * b + (1-self.dampening) * g
                else:
                    b[:] = g 

                if self.nesterov:
                    g += self.momentum * b 
                else:
                    g[:] = b 

            if self.maximize:
                t.data += self.lr * g 
            else:                
                t.data -= self.lr * g

        self.ite += 1

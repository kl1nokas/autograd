from optimizer import Optimizer
import numpy as np

class SGD_Momentum(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.9, weighted_decay=0.001):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weighted_decay = weighted_decay
        
        self.velocities = [0.0] * len(self.params)

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                grad = param.grad
                
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param.data
                
                self.sum_squares[i] += grad * grad
                
                
                adjusted_lr = self.lr / (np.sqrt(self.sum_squares[i]) + self.eps)
                param.data -= adjusted_lr * grad


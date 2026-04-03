import numpy as np
from engine import Tensor
import random

class Module:
    def zero_grad(self):
        for p in self.parametrs(): 
            p.grad = np.zeros_like(p.data)

    def parametrs(self):
        return []


class Linear(Module):
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nout, nin))
        self.b = Tensor(np.zeros((nout, )))

    def __call__(self, x):
        return self.W * x + self.b

    def parametrs(self):
        return [self.W, self.b]


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Linear(sz[i], sz[i+1])  
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)  
            if i != len(self.layers) - 1:
                x = x.relu()  
        return x

    def parametrs(self):
        return [p for l in self.layers for p in l.parametrs()]

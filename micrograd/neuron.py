from engine import Tensor
import numpy
import random

class Module:

    def zero_grad(self):
        for p in self.parametrs:
            p.grad = 0

    def parametrs(self):
        return []
    
class Neuron():

    def __int__(self, nin, nonlin=True):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for i in zip(self.w, x)), self.b)
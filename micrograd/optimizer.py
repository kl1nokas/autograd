from abc import ABC, abstractmethod
import numpy as np
from engine import Tensor

class Optimizer(ABC):
    def __init__(self, params, lr=0.001):
        
        """
        Args:
        params: return list of tensors
        lr - leaning_rate 
        """

        self.params = list(params)
        self.lr = lr
        self._validate_params()

    def validate_params(self): 
        for param in self.params:
            if not isinstance(param, Tensor):
                raise TypeError(f"Expected Tensor, got {type(param)}")
            
    def step(self):
        for param in self.params:
            if param.grad == None:
               continue

            param.data -= self.lr * param.grad

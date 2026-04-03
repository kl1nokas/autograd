from optimizer import Optimizer
import numpy as np
import math

class Adam(Optimizer):
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is not None:
                grad = param.grad
                grad_data = grad.data if hasattr(grad, 'data') else grad
                
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_data
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad_data ** 2)
                
                m_hat = self.m[i] / (1 - math.pow(self.beta1, self.t))
                v_hat = self.v[i] / (1 - math.pow(self.beta2, self.t))
                
                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

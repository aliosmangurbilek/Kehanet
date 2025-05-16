# kehanet/optimizers/adam.py
"""
Adam Optimizer
==============

Adaptive Moment Estimation (Adam) optimizer implementation.
Diederik P. Kingma and Jimmy Ba, 2014.
https://arxiv.org/abs/1412.6980
"""

from typing import List
import numpy as np
from core.tensor import Tensor


class Adam:
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8
    ):
        """
        Adam optimizer constructor.
        
        Args:
            parameters: List of parameters to optimize
            lr: Learning rate (default: 0.001)
            betas: Coefficients for computing running averages of gradient and its square
                  (default: (0.9, 0.999))
            eps: Term added to denominator for numerical stability (default: 1e-8)
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        # Initialize moment estimates and step counter
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0
    
    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
                
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        """Clears the gradients of all optimized tensors."""
        for p in self.parameters:
            p.grad = None

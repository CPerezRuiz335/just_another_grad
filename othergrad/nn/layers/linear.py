from othergrad.tensor import Tensor
from othergrad.nn.module import Module
from math import sqrt
import numpy as np


class Linear(Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,  
        bias: bool = True
    ):
        super().__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features  
        k = 1 / sqrt(in_features)

        self.w = Tensor(
            np.random.uniform(-k, k, size=(out_features, in_features)), 
            requires_grad=True
        )
        
        if self.bias:
            self.b = Tensor(
                np.random.uniform(-k, k, size=out_features),
                requires_grad=True
            )

    def __call__(self, x: Tensor) -> Tensor:    
        if self.bias: 
            return x @ self.w.T + self.b 
        else: 
            return x @ self.w.T    

    def __str__(self):
        return (
            "Layer("
            + (f"in_features={self.in_features}, " if self.in_features else '')
            +  f"out_features={self.out_features}, "
            +  f"bias={self.bias})"
        )

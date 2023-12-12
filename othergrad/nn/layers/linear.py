from othergrad.tensor import Tensor, Function
from othergrad.nn.module import Module
from math import sqrt
from typing import Optional
import numpy as np

def overload(fun):
    def wrapper(self, *args, **kwargs):
        if len(args) == 1:
            return fun(self, in_features=None, out_features=args[0], **kwargs)
        if 'out_features' in kwargs and 'in_features' not in kwargs:
            return fun(self, in_features=None, **kwargs)
        return fun(self, *args, **kwargs)
    return wrapper

class Linear(Module):
    @overload
    def __init__(
        self, 
        in_features: Optional[int], 
        out_features: int,  
        bias: bool = True
    ):
        super().__init__()
        self.w: Optional[Tensor] = None # uninitialized
        self.b: Optional[Tensor] = None # uninitialized

        self.bias = bias
        self.out_features = out_features
        self.in_features = in_features

        self.__init_tensors(in_features) if in_features is not None else ...

    def __init_tensors(self, in_features: int):
        if self.in_features is None:
            self.in_features = in_features  

        k = 1 / sqrt(in_features)
        self.w = Tensor(
            np.random.uniform(-k, k, size=(self.out_features, in_features)), 
            requires_grad=True
        )
        
        if self.bias:
            self.b = Tensor(
            np.random.uniform(-k, k, size=self.out_features),
            requires_grad=True
        )

    def __call__(self, x: Tensor) -> Tensor:    
        self.__init_tensors(x.shape[-1]) if self.w is None else ...

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


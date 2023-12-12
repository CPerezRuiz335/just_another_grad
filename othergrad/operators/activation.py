from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Optional, Union
from othergrad.tensor import Function


class ReLU(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1) -> NDArray:
    	# Ex: 3 TODO
        self.save_for_backward(t1)
        return np.maximum(t1.data, 0) 

    def backward(self, partial: NDArray):
    	# Ex: 3 TODO
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial * (p.data > 0).astype(int)

class Softmax(Function):
    def __init__(self):
        super().__init__()
    
    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        tmp = np.exp(t1.data - np.max(t1.data, axis=1, keepdims=True))
        self.s = tmp / tmp.sum(axis=1, keepdims=True)
        return self.s 

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            row, col = p.shape
            tmp = (
                self.s[..., None] 
                * (np.eye(col) - np.expand_dims(self.s, -2))
            ) @ partial[..., None]
            p.grad += tmp.squeeze()
            
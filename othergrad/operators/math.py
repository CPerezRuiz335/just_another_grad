from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Optional, Union
from othergrad.tensor import Function


def collapse(partial: NDArray, parent_shape: Tuple[int, ...]):
    # Ex: 4 TODO
    # return partial
    # add ones to extra partial dimensions 
    axes = [1,]*(partial.ndim-len(parent_shape)) + list(parent_shape)
    # check which ones were expanded with respect to partial.shape
    reduce_axis = []
    for i, (ax1, ax2) in enumerate(zip(partial.shape, axes)):
        if ax1 > ax2: 
            reduce_axis.append(i)
    # reshape to keep parent shape 
    return np.sum(partial, axis=tuple(reduce_axis)).reshape(parent_shape)

class Matmul(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1, t2) -> NDArray:
        # Ex: 2 TODO
        self.save_for_backward(t1, t2)
        return t1.data.dot(t2.data)

    def backward(self, partial: NDArray):
        # Ex: 2 TODO
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += partial.dot(p2.data.T)

        if p2.requires_grad:
            p2.grad += p1.data.T.dot(partial)

class Add(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1, t2) -> NDArray:
        self.save_for_backward(t1, t2)
        return t1.data + t2.data

    def backward(self, partial: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += collapse(partial, p1.shape)  

        if p2.requires_grad:
            p2.grad += collapse(partial, p2.shape) 

class Sub(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1, t2) -> NDArray:
        self.save_for_backward(t1, t2)
        return t1.data - t2.data

    def backward(self, partial: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += collapse(partial, p1.shape)    

        if p2.requires_grad:
            p2.grad -= collapse(partial, p2.shape)  

class Mul(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1, t2) -> NDArray:
        self.save_for_backward(t1, t2)
        return t1.data * t2.data

    def backward(self, partial: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += collapse(partial * p2.data, p1.shape) 

        if p2.requires_grad:
            p2.grad += collapse(partial * p1.data, p2.shape) 

class Log(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.log(t1.data)

    def backward(self, partial: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += partial * np.reciprocal(p1.data)

class Exp(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.exp(t1.data)

    def backward(self, partial: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += partial * np.exp(p1.data)
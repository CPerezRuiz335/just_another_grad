from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional 

import numpy as np 
from numpy.typing import NDArray

class Function(ABC):
    def __init__(self):
        self.parents = []
        self._name = type(self).__name__

    def save_for_backward(self, *tensors):
        assert all(isinstance(t, Tensor) for t in tensors), \
        "parents must not contain other types than Tensor"
        self.parents.extend(tensors)

    @abstractmethod
    def forward(self, *tensors, **kwargs) -> NDArray:
        raise NotImplementedError(f"forward not implemented for {type(self)}")
    
    @abstractmethod
    def backward(self, partial: NDArray):
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    def __str__(self): 
        return self._name 

import othergrad.operators as ops

class Tensor:
    __array_ufunc__ = None # tell numpy to trust Tensor to make __r****__ methods
    __slots__ = 'data', 'grad', 'fn', 'requires_grad', 'name'

    def __init__(
        self, 
        data, 
        requires_grad: bool = False, 
        fn: Optional[Function] = None, 
        name: str = ''
    ):
        self.data = np.array(data, dtype=np.float32)
        self.fn = fn
        self.requires_grad = requires_grad
        self.name = name
        if requires_grad:  
            self.grad = np.zeros(self.data.shape, dtype=self.data.dtype) 
        else:
            self.grad = None
    
    # ***** backprop *****
    def backward(self, retain_graph=False):
        topo = []
        visited = set([self])    
        def build_topo(tensor: Tensor):
            if tensor.fn:
                for t in tensor.fn.parents:
                    if not t.requires_grad:
                        continue
                    if t not in visited:
                        visited.add(t)
                        build_topo(t)
                topo.append(tensor)

        build_topo(self)

        # chain rule 
        self.grad = np.ones_like(self.data) # dL/dL = 1
        for tensor in reversed(topo):
            tensor.fn.backward(tensor.grad)
            if not retain_graph: 
                tensor.fn = None 

        del topo, visited # outsmart garbage collector

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            'tensor: ' 
            + np.array2string(self.data, prefix='tensor: ', precision=4) 
            + (f" fn: {self.fn}" if self.fn else '') 
            + (f", name: {self.name}" if self.name else '')
        )

    @classmethod
    def comm(cls, function: Function, *tensors) -> Tensor:
        operands = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        data = function.forward(*operands)
        # NOTE: if no leaf tensor requires_grad, neither intermediate ones  
        requires_grad = any(t.requires_grad for t in operands)
        return cls(data, requires_grad=requires_grad, fn=function)

    def __add__(self, x): return Tensor.comm(ops.Add(), self, x)
    def __radd__(self, x): return Tensor.comm(ops.Add(), x, self)

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Optional, Union
from othergrad.tensor import Function


class Transpose(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1) -> NDArray:
    	# Ex: 1 TODO
        self.save_for_backward(t1)
        return t1.data.T

    def backward(self, partial: NDArray):
    	# Ex: 1 TODO
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial.T
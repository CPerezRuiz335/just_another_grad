from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Any

from othergrad.tensor import Tensor


class Module(ABC):
    def __init__(self):
        self.training = True

    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)
        instance.__odict__ = OrderedDict()
        return instance 

    def __setattr__(self, key, value):
        if key != '__odict__': 
            if isinstance(value, Tensor) or isinstance(value, Module):
                self.__odict__[key] = value
        object.__setattr__(self, key, value)

    def __getattr__(self, attr: Any):
        try:
            out_module = self.__odict__[attr]
        except AttributeError:
            raise AttributeError(f"{attr} is not a subModule")
        return out_module

    def train(self):
        self.training = True 
        for subModule in self.__odict__.values():
            if isinstance(subModule, Module):
                subModule.train()

    def eval(self):
        self.training = False
        for subModule in self.__odict__.values():
            if isinstance(subModule, Module):
                subModule.eval()

    def parameters(self) -> List[Tensor]:
        out = []
        for x in self.__odict__.values(): 
            if isinstance(x, Tensor): 
                out.append(x) 
            elif isinstance(x, Module): 
                out.extend(x.parameters())
        return out

    def __repr__(self):
        return self.__str__()

    @abstractmethod 
    def __call__(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(f"__call__ not implemented for class {type(self)}")

    def __str__(self):
        return (
            f"{type(self).__name__}\n{' '*4}" 
            + f"\n{' '*4}".join([
                f"{name}: {mod}" 
                for name, mod 
                in self.__odict__.items() 
                if isinstance(mod, Module)
            ])
        )

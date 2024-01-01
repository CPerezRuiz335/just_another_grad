from othergrad.tensor import Tensor
from othergrad.nn.module import Module
import numpy as np
from numpy.typing import NDArray


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, pred: Tensor, target: NDArray | Tensor) -> Tensor:
        if not isinstance(target, Tensor):
            target = Tensor(target)

        one_hot = np.zeros_like(pred.data)
        one_hot[np.arange(target.size), target.data.astype(np.int8)] = 1

        return (-one_hot*pred.log_softmax()).mean(axis=0).sum()

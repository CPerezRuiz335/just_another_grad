import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import numpy as np
import torch # type: ignore
from othergrad.tensor import Tensor

def test():
    a = Tensor([[1.5, -3.4, 3.0, 1.2,  0,-100.0],
                [-1,  5.0, 6.0, 1.2,  30, -30.0]], requires_grad=True)
    c = a.log_softmax()
    d = c.relu()
    z = d.sum()
    z.backward()
    amg, zmg = a, z

    a = torch.Tensor([[1.5, -3.4, 3.0, 1.2,  0,-100.0],
                      [-1,  5.0, 6.0, 1.2,  30, -30.0]])
    a.requires_grad = True
    c = torch.nn.functional.log_softmax(a, dim=1)
    d = torch.nn.functional.relu(c)
    
    z = d.sum()
    z.backward()
    apt, zpt = a, z

    tol = 1e-5

    # forward pass went well
    assert abs(zmg.data - zpt.data.item()) < tol
    # backward pass went well
    assert np.all(abs(amg.grad.flatten() - apt.grad.detach().numpy().flatten()) < tol) 

if __name__ == "__main__":
    test()
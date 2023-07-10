import torch
import norse

import nir
from nirtorch.from_nir import load, GraphExecutor

import pytest


def _torch_model_map(m: nir.NIRNode, device: str = "cpu") -> torch.nn.Module:
    if isinstance(m, nir.Linear):
        l = torch.nn.Linear(*m.weights.shape[-2:])
        l.weight.data = torch.tensor(m.weights, device=device)
        l.bias.data = torch.tensor(m.bias, device=device)
        return l
    if isinstance(m, norse.torch.LIFBoxCell):
        return nir.LIF(
            m.p.tau_mem_inv, m.p.v_leak, torch.ones_like(m.p.tau_mem_int), m.p.v_th
        )
    elif isinstance(m, nir.Input) or isinstance(m, nir.Output):
        return None
    else:
        raise NotImplementedError(f"Unsupported module {m}")


def test_extract_empty():
    g = nir.NIR([], [])
    with pytest.raises(ValueError):
        m = load(g, _torch_model_map)


def test_extract_lin():
    x = torch.randn(1, 1)
    lin = nir.Linear(x, torch.randn(1, 1))
    torchlin = torch.nn.Linear(1, 1)
    torchlin.weight.data = torch.tensor(lin.weights)
    torchlin.bias.data = torch.tensor(lin.bias)
    y = torchlin(torchlin(x))
    g = nir.NIR([lin, lin], [(0, 1)])
    m = load(g, _torch_model_map)
    assert isinstance(m.get_input_nodes()[0].elem, torch.nn.Linear)
    assert torch.allclose(m.get_input_nodes()[0].elem.weight, lin.weights)
    assert torch.allclose(m.get_input_nodes()[0].elem.bias, lin.bias)
    assert torch.allclose(m(x), y)

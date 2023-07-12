import nir
import norse
import pytest
import torch

from nirtorch.from_nir import load


def _torch_model_map(m: nir.NIRNode, device: str = "cpu") -> torch.nn.Module:
    if isinstance(m, nir.Affine):
        lin = torch.nn.Linear(*m.weight.shape[-2:])
        lin.weight.data = torch.tensor(m.weight, device=device)
        lin.bias.data = torch.tensor(m.bias, device=device)
        return lin
    if isinstance(m, norse.torch.LIFBoxCell):
        return nir.LIF(
            m.p.tau_mem_inv, m.p.v_leak, torch.ones_like(m.p.tau_mem_int), m.p.v_th
        )
    elif isinstance(m, nir.Input) or isinstance(m, nir.Output):
        return None
    else:
        raise NotImplementedError(f"Unsupported module {m}")


def test_extract_empty():
    g = nir.NIRGraph([], [])
    with pytest.raises(ValueError):
        _ = load(g, _torch_model_map)


def test_extract_lin():
    x = torch.randn(1, 1)
    lin = nir.Affine(x, torch.randn(1, 1))
    torchlin = torch.nn.Linear(1, 1)
    torchlin.weight.data = torch.tensor(lin.weight)
    torchlin.bias.data = torch.tensor(lin.bias)
    y = torchlin(torchlin(x))
    g = nir.NIRGraph([lin, lin], [(0, 1)])
    m = load(g, _torch_model_map)
    assert isinstance(m.execution_order[0].elem, torch.nn.Linear)
    assert torch.allclose(m.execution_order[0].elem.weight, lin.weight)
    assert torch.allclose(m.execution_order[0].elem.bias, lin.bias)
    assert torch.allclose(m(x), y)

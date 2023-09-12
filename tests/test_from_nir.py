import nir
import pytest
import torch

from nirtorch.from_nir import load


def _torch_model_map(m: nir.NIRNode, device: str = "cpu") -> torch.nn.Module:
    if isinstance(m, nir.Affine):
        lin = torch.nn.Linear(*m.weight.shape[-2:])
        lin.weight.data = torch.nn.Parameter(m.weight.to(device))
        lin.bias.data = torch.nn.Parameter(m.bias.to(device))
        return lin
    elif isinstance(m, nir.Input) or isinstance(m, nir.Output):
        return None
    else:
        raise NotImplementedError(f"Unsupported module {m}")


def test_extract_empty():
    g = nir.NIRGraph({}, [])
    with pytest.raises(ValueError):
        _ = load(g, _torch_model_map)


def test_extract_lin():
    x = torch.randn(1, 1)
    lin = nir.Affine(x, torch.randn(1, 1))
    torchlin = torch.nn.Linear(1, 1)
    torchlin.weight.data = torch.nn.Parameter(lin.weight)
    torchlin.bias.data = torch.nn.Parameter(lin.bias)
    y = torchlin(torchlin(x))
    g = nir.NIRGraph({"a": lin, "b": lin}, [("a", "b")])
    m = load(g, _torch_model_map)
    assert isinstance(m.execution_order[0].elem, torch.nn.Linear)
    assert torch.allclose(m.execution_order[0].elem.weight, lin.weight)
    assert torch.allclose(m.execution_order[0].elem.bias, lin.bias)
    assert torch.allclose(m(x), y)

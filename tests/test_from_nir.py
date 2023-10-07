import nir
import numpy as np
import pytest
import torch

from nirtorch.from_nir import load


def _torch_model_map(m: nir.NIRNode, device: str = "cpu") -> torch.nn.Module:
    if isinstance(m, nir.Affine):
        lin = torch.nn.Linear(*m.weight.shape[-2:])
        lin.weight.data = torch.nn.Parameter(torch.tensor(m.weight).to(device))
        lin.bias.data = torch.nn.Parameter(torch.tensor(m.bias).to(device))
        return lin
    elif isinstance(m, nir.Linear):
        lin = torch.nn.Linear(*m.weight.shape[-2:], bias=False)
        lin.weight.data = torch.nn.Parameter(torch.tensor(m.weight).to(device))
        return lin
    elif isinstance(m, nir.Input) or isinstance(m, nir.Output):
        return None
    else:
        raise NotImplementedError(f"Unsupported module {m}")


def test_extract_empty():
    g = nir.NIRGraph({}, [])
    with pytest.raises(ValueError):
        _ = load(g, _torch_model_map)


def test_extract_illegal_name():
    graph = nir.NIRGraph({"a.b": nir.Input(np.ones(1))}, [])
    torch_graph = load(graph, _torch_model_map)
    assert "a_b" in torch_graph._modules


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


def test_extrac_recurrent():
    w = np.random.randn(1, 1)
    g = nir.NIRGraph(
        nodes={"in": nir.Input(np.ones(1)), "a": nir.Linear(w), "b": nir.Linear(w)},
        edges=[("in", "a"), ("a", "b"), ("b", "a")],
    )
    l1 = torch.nn.Linear(1, 1, bias=False)
    l1.weight.data = torch.tensor(w)
    l2 = torch.nn.Linear(1, 1, bias=False)
    l2.weight.data = torch.tensor(w)
    m = load(g, _torch_model_map)
    data = torch.randn(1, 1, dtype=torch.float64)
    torch.allclose(m(data), l2(l1(data)))

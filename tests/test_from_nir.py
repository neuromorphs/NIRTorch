import warnings

import nir
import numpy as np
import pytest
import torch

from nirtorch.from_nir import load


def _torch_model_map(m: nir.NIRNode, device: str = "cpu") -> torch.nn.Module:
    if isinstance(m, nir.Affine):
        lin = torch.nn.Linear(*m.weight.shape[-2:])
        lin.weight.data = torch.nn.Parameter(torch.tensor(m.weight).to(device).float())
        lin.bias.data = torch.nn.Parameter(torch.tensor(m.bias).to(device).float())
        return lin
    elif isinstance(m, nir.Linear):
        lin = torch.nn.Linear(*m.weight.shape[-2:], bias=False)
        lin.weight.data = torch.nn.Parameter(torch.tensor(m.weight).to(device).float())
        return lin
    elif isinstance(m, nir.Input) or isinstance(m, nir.Output):
        return torch.nn.Identity()


def _recurrent_model_map(m: nir.NIRNode, device: str = "cpu") -> torch.nn.Module:
    class MyCubaLIF(torch.nn.Module):
        def __init__(self, lif, lin):
            super().__init__()
            self.lif = lif
            self.lin = lin

        def forward(self, x, state=None):
            if state is None:
                state = torch.zeros_like(x)
            z = self.lif(x + state)
            return self.lin(z), z

    try:
        mapped_model = _torch_model_map(m, device)
    except NotImplementedError:
        mapped_model = None
    if mapped_model is None:
        if isinstance(m, nir.CubaLIF):
            return torch.nn.Identity()
        elif isinstance(m, nir.NIRGraph):
            return MyCubaLIF(
                _recurrent_model_map(m.nodes["lif"], device),
                _recurrent_model_map(m.nodes["lin"], device),
            )
        else:
            raise NotImplementedError(f"Unsupported module {m}")
    else:
        return mapped_model


def test_extract_empty():
    g = nir.NIRGraph({}, [])
    with pytest.raises(ValueError):
        _ = load(g, _torch_model_map)


def test_extract_illegal_name():
    graph = nir.NIRGraph(
        {"a.b": nir.Input(np.ones(1)), "a.c": nir.Linear(np.array([[1.0]]))},
        [("a.b", "a.c")],
    )
    torch_graph = load(graph, _torch_model_map)
    assert "a_c" in torch_graph._modules


def test_extract_lin():
    x = torch.randn(1, 1)
    lin = nir.Affine(x, torch.randn(1, 1))
    torchlin = torch.nn.Linear(1, 1)
    torchlin.weight.data = torch.nn.Parameter(lin.weight)
    torchlin.bias.data = torch.nn.Parameter(lin.bias)
    y = torchlin(torchlin(x))
    g = nir.NIRGraph(
        {"i": nir.Input(np.ones((1))), "a": lin, "b": lin}, [("i", "a"), ("a", "b")]
    )
    m = load(g, _torch_model_map)
    assert isinstance(m.execution_order[1].elem, torch.nn.Linear)
    assert torch.allclose(m.execution_order[1].elem.weight, lin.weight)
    assert torch.allclose(m.execution_order[1].elem.bias, lin.bias)
    assert isinstance(m.execution_order[2].elem, torch.nn.Linear)
    assert torch.allclose(m.execution_order[2].elem.weight, lin.weight)
    assert torch.allclose(m.execution_order[2].elem.bias, lin.bias)
    assert torch.allclose(m(x)[0], y)


def test_execute_stateful():
    class StatefulModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, state=None):
            if state is None:
                state = 1
            return x + state, state

    def _map_stateful(node):
        if isinstance(node, nir.Flatten):
            return StatefulModel()

    g = nir.NIRGraph(
        nodes={
            "i": nir.Input(np.array([1, 1, 1])),
            "li": nir.Flatten(np.array([1, 1, 1])),
            "li2": nir.Flatten(np.array([1, 1]), start_dim=0),
            "o": nir.Output(np.array([1])),
        },
        edges=[("i", "li"), ("li", "li2"), ("li2", "o")],
    )  # Mock node

    # TODO: Revisit and validate assertions
    m = load(g, _map_stateful)
    out = m(torch.ones(10, 10, 10))
    assert isinstance(out, tuple)
    out, state = out
    assert torch.allclose(out, torch.ones(10) * 3)
    assert state.state["li"] == (1,)
    assert state.state["li"] == (1,)

    # Test that the model can avoid returning state
    m = load(g, _map_stateful, return_state=False)
    assert not isinstance(m(torch.ones(10)), tuple)


def test_execute_recurrent():
    w = np.ones((1, 1))
    g = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.ones(1)),
            "a": nir.Linear(w),
            "b": nir.Linear(w),
            "out": nir.Output(np.ones(1)),
        },
        edges=[("in", "a"), ("a", "b"), ("b", "a"), ("a", "out")],
    )
    m = load(g, _torch_model_map)
    data = torch.ones(1, 1)

    # Same execution without reusing state should yield the same result
    y1 = m(data)
    y2 = m(data)
    assert torch.allclose(y1[0], y2[0])
    out, s = m(*m(data))
    assert torch.allclose(out, torch.tensor(2.0))


def test_execute_graph_in_graph():
    w = np.ones((1, 1)) * 2
    g1 = nir.NIRGraph.from_list(nir.Linear(w))
    g2 = nir.NIRGraph.from_list(g1, nir.Linear(w))
    m = load(g2, _torch_model_map)
    data = torch.ones(1, 1)
    out, state = m(data)
    assert torch.allclose(out, torch.tensor(4.0))
    assert torch.allclose(
        state.state["nirgraph"][0].cache["linear"], torch.tensor([2.0])
    )


def test_import_braille():
    g = nir.read("tests/braille.nir")
    m = load(g, _recurrent_model_map)
    assert m(torch.empty(1, 12))[0].shape == (1, 7)


def test_deprecation_warning():
    g = nir.read("tests/braille.nir")
    with warnings.catch_warnings(record=True) as warn:
        load(g, _recurrent_model_map)
        assert len(warn) == 1
        assert isinstance(warn[0].message, DeprecationWarning)

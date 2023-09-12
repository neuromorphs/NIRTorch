import nir
import numpy as np
import torch
import torch.nn as nn

from nirtorch.to_nir import extract_nir_graph


def test_extract_single():
    m = nn.Linear(1, 1)
    g = extract_nir_graph(
        m,
        lambda x: nir.Affine(x.weight.detach().numpy(), x.bias.detach().numpy()),
        torch.rand(1, 1),
    )
    assert set(g.edges) == {("input", "model"), ("model", "output")}
    assert isinstance(g.nodes["input"], nir.Input)
    assert np.allclose(g.nodes["input"].input_type["input"], np.array([1, 1]))
    assert isinstance(g.nodes["model"], nir.Affine)
    assert np.allclose(g.nodes["model"].weight, m.weight.detach().numpy())
    assert np.allclose(g.nodes["model"].bias, m.bias.detach().numpy())
    assert isinstance(g.nodes["output"], nir.Output)


def test_extract_nir_edges():
    mymodel = nn.Sequential(
        nn.Conv2d(2, 10, 3),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(10, 4, 3),
        nn.ReLU(),
        nn.AvgPool2d(2),
    )
    sample_data = torch.rand(1, 2, 16, 16)

    def dummy_model_map(module: nn.Module) -> nir.NIRNode:
        return nir.NIRNode()

    nir_graph = extract_nir_graph(mymodel, dummy_model_map, sample_data, "mysequential")

    assert len(nir_graph.nodes) == 8
    print(nir_graph.edges)
    assert set(nir_graph.edges) == {
        ("input", "0"),
        ("0", "1"),
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "5"),
        ("5", "output"),
    }


class BranchedModel(nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = torch.randn(a, b)
        self.b = torch.randn(b, c)

    def forward(self, x):
        return x @ self.a @ self.b


def test_extract_multiple_explicit():
    model = nn.Sequential(BranchedModel(1, 2, 3), nn.Linear(3, 4))

    def extractor(module: nn.Module):
        if isinstance(module, BranchedModel):
            return nir.NIRGraph(
                nodes={
                    "0": nir.Input(np.array(module.a.shape[0])),
                    "1": nir.Linear(module.a),
                    "2": nir.Linear(module.b),
                    "3": nir.Output(np.array(module.b.shape[0])),
                },
                edges=[("0", "1"), ("0", "2"), ("1", "2"), ("1", "3"), ("2", "3")],
            )
        else:
            return nir.Affine(module.weight, module.bias)

    g = extract_nir_graph(model, extractor, torch.rand(1))
    print([type(n) for n in g.nodes])
    assert len(g.nodes) == 7
    assert len(g.edges) == 8  # in + 5 + 1 + out


def test_extract_recursive():
    class RecursiveModel(torch.nn.Module):
        def forward(self, x, s=None):
            if s is None:
                s = torch.zeros_like(x)
            return x + 1, s + x

    model = RecursiveModel()

    def extractor(m):
        if isinstance(m, RecursiveModel):
            return nir.Delay(np.array([1]))

    g = extract_nir_graph(model, extractor, torch.rand(1))
    assert set(g.edges) == {
        ("input", "model"),
        ("model", "output"),
        # ("model", "model") TODO: Detect and add recursive connections
    }


# def test_extract_stateful():
#     model = norse.SequentialState(norse.LIFBoxCell(), nn.Linear(3, 1))

#     def extract(module: torch.nn.Module):
#         if isinstance(module, norse.LIFBoxCell):
#             return nir.NIR(
#                 nodes=[
#                     nir.LIF(
#                         tau=module.p.tau_mem_inv.detach(),
#                         v_threshold=module.p.v_th.detach(),
#                         v_leak=module.p.v_leak.detach(),
#                         r=torch.ones_like(module.p.v_leak.detach()),
#                     )
#                 ]
#             )
#         elif isinstance(module, torch.nn.Linear):
#             return nir.NIR(nodes=[nir.Linear(module.weight, module.bias)])

#     graph = extract_nir_graph(model, extract, torch.rand(1, 3))
#     assert len(graph.nodes) == 4
#     assert len(graph.edges) == 3
#     assert graph.edges[0] == (0, 1)

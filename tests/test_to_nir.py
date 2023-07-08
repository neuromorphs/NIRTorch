import torch
import torch.nn as nn

import norse.torch as norse

import nir
from nirtorch.to_nir import extract_nir_graph


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
    print(nir_graph)

    assert len(nir_graph.nodes) == 6
    assert len(nir_graph.edges) == 5


class BranchedModel(nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = torch.randn(a, b)
        self.b = torch.randn(b, c)

    def forward(self, x):
        return x @ self.a @ self.b


def test_extract_multiple():
    model = nn.Sequential(BranchedModel(1, 2, 3), nn.Linear(3, 4))

    def extractor(module: nn.Module):
        if isinstance(module, BranchedModel):
            return [
                nir.Linear(module.a, torch.zeros(2)),
                nir.Linear(module.b, torch.zeros(3)),
            ]
        else:
            return nir.Linear(module.weight, module.bias)

    g = extract_nir_graph(model, extractor, torch.rand(1, 1))
    assert len(g.nodes) == 3
    assert len(g.edges) == 2


def test_extract_stateful():
    model = norse.SequentialState(norse.LIFBoxCell(), nn.Linear(3, 1))

    def extract(module: torch.nn.Module):
        if isinstance(module, norse.LIFBoxCell):
            return nir.LIF(
                tau=module.p.tau_mem_inv,
                v_th=module.p.v_th,
                v_leak=module.p.v_leak,
                r=torch.ones_like(module.p.v_leak),
            )
        elif isinstance(module, torch.nn.Linear):
            return nir.Linear(module.weight, module.bias)

    graph = extract_nir_graph(model, extract, torch.rand(1, 3))
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0] == (0, 1)

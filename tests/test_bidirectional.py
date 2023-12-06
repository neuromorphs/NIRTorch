import nir
import numpy as np
import torch

import nirtorch

use_snntorch = False
# use_snntorch = True


if use_snntorch:
    import snntorch as snn


def _nir_to_snntorch_module(node: nir.NIRNode) -> torch.nn.Module:
    if isinstance(node, (nir.Linear, nir.Affine)):
        return torch.nn.Linear(*node.weight.shape)

    elif isinstance(node, (nir.LIF, nir.CubaLIF)):
        return snn.Leaky(0.9, init_hidden=True)

    else:
        return None


def _nir_to_pytorch_module(node: nir.NIRNode) -> torch.nn.Module:
    if isinstance(node, (nir.Linear, nir.Affine)):
        return torch.nn.Linear(*node.weight.shape)

    elif isinstance(node, (nir.LIF, nir.CubaLIF)):
        return torch.nn.Linear(1, 1)

    else:
        return None


if use_snntorch:
    _nir_to_torch_module = _nir_to_snntorch_module
else:
    _nir_to_torch_module = _nir_to_pytorch_module


def _create_torch_model() -> torch.nn.Module:
    if use_snntorch:
        return torch.nn.Sequential(
            torch.nn.Linear(1, 1), snn.Leaky(0.9, init_hidden=True)
        )
    else:
        return torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Identity())


def _torch_to_nir(module: torch.nn.Module) -> nir.NIRNode:
    if isinstance(module, torch.nn.Linear):
        return nir.Linear(np.array(module.weight.data))

    else:
        return None


def _lif_nir_graph(from_file=True):
    if from_file:
        return nir.read("tests/lif_norse.nir")
    else:
        return nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([1])}),
                "0": nir.Affine(weight=np.array([[1.0]]), bias=np.array([0.0])),
                "1": nir.LIF(
                    tau=np.array([0.1]),
                    r=np.array([1.0]),
                    v_leak=np.array([0.0]),
                    v_threshold=np.array([0.1]),
                ),
                "output": nir.Output(output_type={"output": np.array([1])}),
            },
            edges=[("input", "0"), ("0", "1"), ("1", "output")],
        )


def test_nir_to_torch_to_nir(from_file=True):
    graph = _lif_nir_graph(from_file=from_file)
    assert graph is not None
    module = nirtorch.load(graph, _nir_to_torch_module)
    assert module is not None
    graph2 = nirtorch.extract_nir_graph(module, _torch_to_nir, torch.zeros(1, 1))
    edges1 = sorted(graph.edges)
    edges2 = sorted(graph2.edges)
    for e1, e2 in zip(edges1, edges2):
        assert e1 == e2


# if __name__ == '__main__':
#     test_nir_to_torch_to_nir(from_file=False)

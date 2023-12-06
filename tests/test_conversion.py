import nir
import torch
import torch.nn as nn

import nirtorch


def _torch_convert(module: nn.Module) -> nir.NIRNode:
    if isinstance(module, nn.Conv1d):
        return nir.Conv1d(None, module.weight, 1, 1, 1, 1, module.bias)
    elif isinstance(module, nn.Linear):
        return nir.Affine(module.weight, module.bias)


def test_extract_pytorch():
    model = torch.nn.Sequential(
        torch.nn.Conv1d(1, 2, 3),
        torch.nn.Linear(8, 1),
        torch.nn.Conv1d(2, 1, 1),
        torch.nn.Linear(1, 2),
    )
    batch_size = 2
    nir_graph = nirtorch.extract_nir_graph(
        model, _torch_convert, torch.rand(batch_size, 1, 10)
    )

    torch.testing.assert_close(nir_graph.nodes["0"].weight, model[0].weight)
    torch.testing.assert_close(nir_graph.nodes["0"].bias, model[0].bias)
    torch.testing.assert_close(nir_graph.nodes["3"].weight, model[3].weight)

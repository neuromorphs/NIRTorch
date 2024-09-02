import torch

from nirtorch.graph import TorchGraph
from nirtorch.graph_executor import GraphExecutor


def test_execute_stateful_then_stateless():
    class StatefulModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, state=None):
            if state is None:
                state = 0
            return x + state, x

    s = StatefulModel()
    lin = torch.nn.Linear(1, 1, bias=False)
    lin.weight.data = torch.tensor([[1]]).float()
    graph = TorchGraph.from_torch_modules({"0": s, "1": lin}, [("0", "1")])

    # With state
    executor = GraphExecutor(graph, True)
    data = torch.ones(1, 1)
    out, state = executor(data)
    assert torch.allclose(out, torch.ones(1))
    out, _ = executor(data, state)
    assert torch.allclose(out, torch.ones(1) * 2)

    # Without state
    executor = GraphExecutor(graph, False)
    data = torch.ones(1, 1)
    out = executor(data)
    assert torch.allclose(out, torch.ones(1))

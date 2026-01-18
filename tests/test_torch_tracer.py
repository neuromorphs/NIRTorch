import pytest

import nir
import numpy as np
import torch

from nirtorch.torch_tracer import torch_to_nir


def _filter_edges(graph, t1, t2):
    return [
        e
        for e in graph.edges
        if graph.nodes[e[0]].__class__ == t1 and graph.nodes[e[1]].__class__ == t2
    ]


def _filter_nodes(graph, t):
    return {k: v for k, v in graph.nodes.items() if v.__class__ == t}


def test_trace_unknown_leaf():
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x

    model = MyModule()
    with pytest.raises(ValueError):
        torch_to_nir(model, {})


def test_trace_default_linear():
    model = torch.nn.Linear(1, 1)
    graph = torch_to_nir(model, {})
    assert graph.__class__ == nir.Affine


def test_trace_default_linear_no_bias():
    model = torch.nn.Linear(1, 1, bias=False)
    graph = torch_to_nir(model, {})
    assert graph.__class__ == nir.Linear


def test_trace_mapped_module():
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x

    def map_my_module(module):
        return nir.Linear(np.array([[1]]))

    model = MyModule()
    graph = torch_to_nir(model, {MyModule: map_my_module})
    assert graph.__class__ == nir.Linear


def test_trace_mapped_module_stateless():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)

        def forward(self, x, state):
            return self.lin(x) + state

    model = MyModule()
    graph = torch_to_nir(model, {})
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert len(_filter_edges(graph, nir.Input, nir.Affine)) == 1
    assert len(_filter_edges(graph, nir.Affine, nir.Output)) == 1
    input_node_name, input_node = list(graph.inputs.items())[0]
    assert input_node.input_type["input"] == np.array([3])
    assert graph.input_type[input_node_name] == np.array([3])


def test_trace_mapped_module_stateful():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 2)
            self.state = torch.tensor([1.0])

        def forward(self, x):
            return self.lin(x) + self.state

    model = MyModule()
    graph = torch_to_nir(model, {})
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert len(_filter_edges(graph, nir.Input, nir.Affine)) == 1
    assert len(_filter_edges(graph, nir.Affine, nir.Output)) == 1
    input_node_name, input_node = list(graph.inputs.items())[0]
    assert input_node.input_type["input"] == np.array([3])
    assert graph.input_type[input_node_name] == np.array([3])


def test_trace_addition():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.lin(x) + self.lin(x)

    model = MyModule()
    graph = torch_to_nir(model, {})
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 4
    assert len(_filter_edges(graph, nir.Input, nir.Affine)) == 2
    assert len(_filter_edges(graph, nir.Affine, nir.Output)) == 2


def test_trace_sequential():
    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Linear(2, 1))
    graph = torch_to_nir(model, {})
    assert graph.__class__ == nir.NIRGraph
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3
    list(graph.nodes.items())
    assert len(_filter_nodes(graph, nir.Input)) == 1, "We require one input node"
    assert len(_filter_nodes(graph, nir.Output)) == 1, "We require one output node"
    assert len(_filter_nodes(graph, nir.Affine)) == 2, "We require two affine nodes"

    assert len(_filter_edges(graph, nir.Input, nir.Affine)) == 1
    assert len(_filter_edges(graph, nir.Affine, nir.Affine)) == 1
    assert len(_filter_edges(graph, nir.Affine, nir.Output)) == 1


def test_trace_submodule():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = MyModule()
    graph = torch_to_nir(model, {})
    assert graph.__class__ == nir.NIRGraph
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    list(graph.nodes.items())
    assert len(_filter_nodes(graph, nir.Input)) == 1, "We require one input node"
    assert len(_filter_nodes(graph, nir.Output)) == 1, "We require one output node"
    assert len(_filter_nodes(graph, nir.Affine)) == 1, "We require one affine node"

    assert len(_filter_edges(graph, nir.Input, nir.Affine)) == 1
    assert len(_filter_edges(graph, nir.Affine, nir.Output)) == 1


def test_trace_nested_submodule():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    class MyModule2(torch.nn.Module):
        def __init__(self):
            super(MyModule2, self).__init__()
            self.module = MyModule()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(self.module(x))

    model = MyModule2()
    graph = torch_to_nir(model, {})
    assert graph.__class__ == nir.NIRGraph
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3

    assert len(_filter_nodes(graph, nir.Input)) == 1, "We require one input node"
    assert len(_filter_nodes(graph, nir.Output)) == 1, "We require one output node"
    assert len(_filter_nodes(graph, nir.Affine)) == 2, "We require two affine nodes"

    assert len(_filter_edges(graph, nir.Input, nir.Affine)) == 1
    assert len(_filter_edges(graph, nir.Affine, nir.Affine)) == 1
    assert len(_filter_edges(graph, nir.Affine, nir.Output)) == 1


def test_recursive_stateful():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.state = torch.tensor([1.0])

        def forward(self, x):
            out = self.linear(x + self.state)
            self.state = out
            return out

    model = MyModule()
    graph = torch_to_nir(model, {})
    assert graph.__class__ == nir.NIRGraph


def test_stateful_module_with_tuple_return():
    """Test tracing a module that returns (output, state) tuples."""

    class StatefulCell(torch.nn.Module):
        """A simple stateful module that returns (output, state) tuple."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x, state=None):
            if state is None:
                state = torch.zeros(2)
            output = self.linear(x) + state
            new_state = output.detach()
            return output, new_state

    def map_stateful_cell(module):
        return nir.Affine(
            module.linear.weight.detach().numpy(),
            module.linear.bias.detach().numpy(),
        )

    model = StatefulCell()
    graph = torch_to_nir(
        model,
        {StatefulCell: map_stateful_cell},
        stateful_modules={StatefulCell},
    )
    assert graph.__class__ == nir.Affine


def test_sequential_with_stateful_modules():
    """Test tracing a sequential model containing stateful modules."""

    class StatefulCell(torch.nn.Module):
        """A simple stateful module that returns (output, state) tuple."""

        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features)

        def forward(self, x, state=None):
            if state is None:
                state = torch.zeros(self.linear.out_features)
            output = self.linear(x) + state
            new_state = output.detach()
            return output, new_state

    class SequentialStateful(torch.nn.Module):
        """A sequential module that handles stateful children."""

        def __init__(self, *modules):
            super().__init__()
            self.layers = torch.nn.ModuleList(modules)
            self.stateful = [isinstance(m, StatefulCell) for m in modules]

        def forward(self, x, state=None):
            if state is None:
                state = [None] * len(self.layers)
            output_states = []
            for i, layer in enumerate(self.layers):
                if self.stateful[i]:
                    x, s = layer(x, state[i])
                    output_states.append(s)
                else:
                    x = layer(x)
                    output_states.append(None)
            return x, output_states

    def map_stateful_cell(module):
        return nir.Affine(
            module.linear.weight.detach().numpy(),
            module.linear.bias.detach().numpy(),
        )

    model = SequentialStateful(
        StatefulCell(3, 4),
        torch.nn.Linear(4, 2),
    )

    graph = torch_to_nir(
        model,
        {StatefulCell: map_stateful_cell},
        stateful_modules={StatefulCell},
        concrete_args={"state": None},
    )

    assert graph.__class__ == nir.NIRGraph
    assert len(graph.nodes) == 4  # input, StatefulCell, Linear, output
    assert len(graph.edges) == 3
    assert len(_filter_nodes(graph, nir.Input)) == 1
    assert len(_filter_nodes(graph, nir.Output)) == 1
    assert len(_filter_nodes(graph, nir.Affine)) == 2
    assert len(_filter_edges(graph, nir.Input, nir.Affine)) == 1
    assert len(_filter_edges(graph, nir.Affine, nir.Affine)) == 1
    assert len(_filter_edges(graph, nir.Affine, nir.Output)) == 1


def test_concrete_args_state_none():
    """Test that concrete_args={'state': None} allows tracing with state parameter."""

    class ModuleWithState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x, state=None):
            # When state is None (concrete), this creates a real list, not a Proxy
            if state is None:
                state = [0.0]
            return self.linear(x)

    model = ModuleWithState()
    graph = torch_to_nir(
        model,
        {},
        concrete_args={"state": None},
    )
    assert graph.__class__ == nir.NIRGraph
    assert len(_filter_nodes(graph, nir.Affine)) == 1

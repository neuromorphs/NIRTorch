import typing

import nir
import numpy as np
import torch
import pytest

from nirtorch import interpreter


class LeakyNode(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, x, state=None):
        if state is None:
            state = self.v_leak
        new_value = state + self.tau * (self.r * x - self.v_leak)
        return new_value, new_value


def _map_li(li: nir.LI) -> torch.nn.Module:
    return LeakyNode(
        tau=torch.from_numpy(li.tau),
        r=torch.from_numpy(li.r),
        v_leak=torch.from_numpy(li.v_leak),
    )


def test_map_fails_on_unknown():
    w = np.ones((2, 2))
    linear = nir.Linear(w)
    graph = nir.NIRGraph.from_list(linear)
    with pytest.raises(ValueError):
        interpreter.to_torch(graph, {}, {})


def test_map_linear_node():
    w = np.random.random((2, 3)).astype(np.float32)
    linear = nir.Linear(w)
    module = interpreter._map_nir_node_to_torch(linear, interpreter.DEFAULT_MAP)
    assert torch.allclose(module.weight, torch.from_numpy(w))
    out = module(torch.ones(3))
    assert out.shape == (2,)


def test_map_conv1d_node():
    w = np.random.random((2, 3, 4)).astype(np.float32)
    b = np.random.random((2,)).astype(np.float32)
    nir_conv = nir.Conv1d(
        input_shape=10, weight=w, stride=2, padding=3, dilation=4, groups=1, bias=b
    )
    torch_conv = interpreter._map_nir_node_to_torch(nir_conv, interpreter.DEFAULT_MAP)
    assert torch.allclose(torch.from_numpy(nir_conv.weight), torch_conv.weight)
    assert torch.allclose(torch.from_numpy(nir_conv.bias), torch_conv.bias)
    assert nir_conv.stride == torch_conv.stride[0]
    assert nir_conv.padding == torch_conv.padding[0]
    assert nir_conv.dilation == torch_conv.dilation[0]
    assert nir_conv.groups == torch_conv.groups
    assert torch_conv(torch.ones(1, 3, 10)).shape == (1, 2, 2)


def test_map_conv2d_node():
    w = np.random.random((2, 3, 4, 5)).astype(np.float32)
    b = np.random.random((2,)).astype(np.float32)
    nir_conv = nir.Conv2d(
        input_shape=(10, 11),
        weight=w,
        stride=(2, 3),
        padding=(1, 2),
        dilation=(1, 2),
        groups=1,
        bias=b,
    )
    torch_conv = interpreter._map_nir_node_to_torch(nir_conv, interpreter.DEFAULT_MAP)
    assert torch.allclose(torch.from_numpy(nir_conv.weight), torch_conv.weight)
    assert torch.allclose(torch.from_numpy(nir_conv.bias), torch_conv.bias)
    assert nir_conv.stride == torch_conv.stride
    assert nir_conv.padding == torch_conv.padding
    assert nir_conv.dilation == torch_conv.dilation
    assert nir_conv.groups == torch_conv.groups
    assert torch_conv(torch.ones(1, 3, 10, 11)).shape == (1, 2, 5, 3)


def test_map_leaky_stateful_graph_single_module():
    # Test that the graph can handle a single stateful module
    tau = np.random.random(1)
    r = np.random.random(1)
    v_leak = np.random.random(1)
    li = nir.LI(tau, r, v_leak)
    li_module = _map_li(li)
    module = interpreter.to_torch(nir.NIRGraph.from_list(li), {nir.LI: _map_li})
    data = torch.rand(1)
    output = module(data)
    assert isinstance(output, typing.Tuple)
    module_output = li_module(data)
    assert torch.allclose(output[0], module_output[0])
    # Application a second time should yield a different, stateful response
    output = module(output[0], output[1])
    assert isinstance(output, typing.Tuple)
    module_output = li_module(*module_output)
    assert torch.allclose(output[0], module_output[0])


def test_map_leaky_stateful_graph_sequential_modules():
    # Test that the graph can handle two chained stateful modules internally
    tau = np.random.random(1)
    r = np.random.random(1)
    v_leak = np.random.random(1)
    li = nir.LI(tau, r, v_leak)
    li_module = _map_li(li)
    module = interpreter.to_torch(nir.NIRGraph.from_list(li, li), {nir.LI: _map_li})
    data = torch.rand(1)
    output = module(data)
    assert isinstance(output, typing.Tuple)
    module_result = li_module(li_module(data)[0])[0]
    assert torch.allclose(output[0], module_result)


def test_map_linear_graph_default():
    w = np.random.random((2, 3)).astype(np.float32)
    linear = nir.Linear(w)
    graph = nir.NIRGraph.from_list(linear)
    module = interpreter.to_torch(graph, {})
    assert torch.allclose(module.linear.weight, torch.from_numpy(w))
    out = module(torch.ones(3))
    assert isinstance(out, typing.Tuple)
    assert out[0].shape == (2,)


def test_map_subgraph_default():
    w = np.random.random((2, 3)).astype(np.float32)
    linear = nir.Linear(w)
    subgraph = nir.NIRGraph.from_list(linear)
    graph = nir.NIRGraph.from_list(subgraph)
    module = interpreter.to_torch(graph, {})
    assert torch.allclose(module.nirgraph.linear.weight, torch.from_numpy(w))
    out = module(torch.ones(3))
    assert isinstance(out, typing.Tuple)
    assert out[0].shape == (2,)


def test_map_subgraph_with_state():
    tau = np.random.random(1)
    r = np.random.random(1)
    v_leak = np.random.random(1)
    li = nir.LI(tau, r, v_leak)
    li_module = _map_li(li)
    subgraph = nir.NIRGraph.from_list(li)
    graph = nir.NIRGraph.from_list(subgraph)
    module = interpreter.to_torch(graph, {nir.LI: _map_li})
    data = torch.ones(1)
    out = module(data)
    assert isinstance(out, typing.Tuple)
    assert out[0].shape == (1,)
    module_result = li_module(data)[0]
    assert torch.allclose(out[0], module_result)


def test_map_nested_subgraph_default():
    w = np.random.random((2, 3)).astype(np.float32)
    linear = nir.Linear(w)
    subgraph = nir.NIRGraph.from_list(linear)
    subgraph2 = nir.NIRGraph.from_list(subgraph)
    graph = nir.NIRGraph.from_list(subgraph2)
    module = interpreter.to_torch(graph, {})
    assert torch.allclose(module.nirgraph.nirgraph.linear.weight, torch.from_numpy(w))
    out = module(torch.ones(3))
    assert isinstance(out, typing.Tuple)
    assert out[0].shape == (2,)

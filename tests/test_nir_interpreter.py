import typing
import warnings

import nir
import numpy as np
import torch
import pytest

from nirtorch import nir_interpreter


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


def _map_affine_node(m: nir.Affine):
    lin = torch.nn.Linear(*m.weight.shape[-2:])
    lin.weight.data = torch.nn.Parameter(torch.tensor(m.weight).float())
    lin.bias.data = torch.nn.Parameter(torch.tensor(m.bias).float())
    return lin


def _map_linear_node(m: nir.Affine):
    lin = torch.nn.Linear(*m.weight.shape[-2:], bias=False)
    lin.weight.data = torch.nn.Parameter(torch.tensor(m.weight).float())
    return lin


def _map_identity(m: nir.NIRNode):
    return torch.nn.Identity()


_torch_node_map = {
    nir.Affine: _map_affine_node,
    nir.Linear: _map_linear_node,
    nir.CubaLIF: _map_identity,
    nir.LIF: _map_identity,
    nir.LI: _map_li,
}


def test_map_fails_on_unknown():
    w = np.ones((2, 2))
    linear = nir.Linear(w)
    graph = nir.NIRGraph.from_list(linear)
    with pytest.raises(ValueError):
        nir_interpreter.nir_to_torch(graph, {}, {})


def test_map_nodes_with_periods_in_name():
    w = np.ones((2, 2))
    linear = nir.Linear(w)
    graph = nir.NIRGraph(
        nodes={
            "some.name": linear,
            "i": nir.Input(np.array([2])),
            "o": nir.Output(np.array([2])),
        },
        edges=[("i", "some.name"), ("some.name", "o")],
    )
    mapped = nir_interpreter.nir_to_torch(graph, {})
    named_children = list(mapped.named_children())
    assert len(named_children) == 1
    assert named_children[0][0] == "some_name"


def test_map_avg_pool_2d():
    pool = nir.AvgPool2d(2, 1, (1, 2))
    torch_pool = nir_interpreter.nir_to_torch(pool, {})
    assert isinstance(torch_pool, torch.nn.AvgPool2d)
    assert torch_pool.kernel_size == 2
    assert torch_pool.stride == 1
    assert torch_pool.padding == (1, 2)


def test_map_conv1d_node():
    w = np.random.random((2, 3, 4)).astype(np.float32)
    b = np.random.random((2,)).astype(np.float32)
    nir_conv = nir.Conv1d(
        input_shape=10, weight=w, stride=2, padding=3, dilation=4, groups=1, bias=b
    )
    torch_conv = nir_interpreter._map_nir_node_to_torch(
        nir_conv, nir_interpreter.DEFAULT_MAP
    )
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
    torch_conv = nir_interpreter._map_nir_node_to_torch(
        nir_conv, nir_interpreter.DEFAULT_MAP
    )
    assert torch.allclose(torch.from_numpy(nir_conv.weight), torch_conv.weight)
    assert torch.allclose(torch.from_numpy(nir_conv.bias), torch_conv.bias)
    assert nir_conv.stride == torch_conv.stride
    assert nir_conv.padding == torch_conv.padding
    assert nir_conv.dilation == torch_conv.dilation
    assert nir_conv.groups == torch_conv.groups
    assert torch_conv(torch.ones(1, 3, 10, 11)).shape == (1, 2, 5, 3)


def test_map_if_node():
    # Generating code with the "if" keyword can be sensitive
    # This tests ensures that it works properly
    v_th = np.random.random(1)
    r = np.random.random(1)
    v_reset = np.zeros(1)
    nir_node = nir.IF(r=r, v_threshold=v_th, v_reset=v_reset)

    class MyIF(torch.nn.Module):
        def __init__(self, r, v_th):
            super().__init__()
            self.r = r
            self.v_th = v_th

        def forward(self, x, state):
            return x * self.r - self.v_th, state

    node_map = {
        nir.IF: lambda node: MyIF(
            torch.from_numpy(node.r), torch.from_numpy(node.v_threshold)
        )
    }
    torch_if = nir_interpreter._map_nir_node_to_torch(nir_node, node_map)
    data = torch.rand(1)
    assert torch.allclose(data * r - v_th, torch_if(data, None)[0])

    torch_if = nir_interpreter.nir_to_torch(nir.NIRGraph.from_list(nir_node), node_map)
    assert isinstance(torch_if.get_submodule("nir_node_if"), MyIF)


def test_map_flatten():
    flatten = nir.Flatten({"input": np.array([1, 2, 10, 10])}, 2, 3)
    torch_flatten = nir_interpreter.nir_to_torch(flatten, {})
    assert isinstance(torch_flatten, torch.nn.Flatten)
    assert torch_flatten.start_dim == 2
    assert torch_flatten.end_dim == 3


def test_map_linear_node():
    w = np.random.random((2, 3)).astype(np.float32)
    linear = nir.Linear(w)
    module = nir_interpreter._map_nir_node_to_torch(linear, nir_interpreter.DEFAULT_MAP)
    assert torch.allclose(module.weight, torch.from_numpy(w))
    out = module(torch.ones(3))
    assert out.shape == (2,)


def test_map_single_node():
    w = np.random.random((2, 2))
    node = nir.Linear(w)
    torch_linear = nir_interpreter.nir_to_torch(node, {})
    assert torch.allclose(torch_linear.weight, torch.from_numpy(w))


def test_map_sum_pool_2d():
    pool = nir.SumPool2d(2, 1, 0)
    torch_pool = nir_interpreter.nir_to_torch(pool, {})
    assert isinstance(torch_pool, torch.nn.LPPool2d)
    assert torch_pool.norm_type == 1
    assert torch_pool.kernel_size == 2
    assert torch_pool.stride == 1


def test_map_leaky_stateful_graph_single_module():
    # Test that the graph can handle a single stateful module
    tau = np.random.random(1)
    r = np.random.random(1)
    v_leak = np.random.random(1)
    li = nir.LI(tau, r, v_leak)
    li_module = _map_li(li)
    module = nir_interpreter.nir_to_torch(nir.NIRGraph.from_list(li), {nir.LI: _map_li})
    data = torch.rand(1)
    output = module(data)
    assert isinstance(output, typing.Tuple)
    module_output = li_module(data)
    assert torch.allclose(output[0], module_output[0])
    assert torch.allclose(output[1]["li"], module_output[1])
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
    module = nir_interpreter.nir_to_torch(
        nir.NIRGraph.from_list(li, li), {nir.LI: _map_li}
    )
    data = torch.rand(1)
    output = module(data)
    assert isinstance(output, typing.Tuple)
    module_result = li_module(li_module(data)[0])[0]
    assert torch.allclose(output[0], module_result)


def test_map_linear_graph_default():
    w = np.random.random((2, 3)).astype(np.float32)
    linear = nir.Linear(w)
    graph = nir.NIRGraph.from_list(linear)
    module = nir_interpreter.nir_to_torch(graph, {})
    assert torch.allclose(module.linear.weight, torch.from_numpy(w))
    out = module(torch.ones(3))
    assert isinstance(out, typing.Tuple)
    assert out[0].shape == (2,)


def test_map_sequential_graph():
    l1 = nir.Linear(np.random.random((2, 2)))
    l2 = nir.Linear(np.random.random((2, 2)))
    l3 = nir.Linear(np.random.random((2, 2)))
    l4 = nir.Linear(np.random.random((2, 2)))
    graph = nir.NIRGraph.from_list(l1, l2, l3, l4)
    module = nir_interpreter.nir_to_torch(graph, {})
    data = torch.rand(2)
    expected = torch.from_numpy(
        l4.weight @ (l3.weight @ (l2.weight @ (l1.weight @ data.numpy())))
    ).float()
    assert len(list(module.children())) == 4
    assert torch.allclose(module(data)[0], expected)


def test_map_subgraph_default():
    w = np.random.random((2, 3)).astype(np.float32)
    linear = nir.Linear(w)
    subgraph = nir.NIRGraph.from_list(linear)
    graph = nir.NIRGraph.from_list(subgraph)
    module = nir_interpreter.nir_to_torch(graph, {})
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
    module = nir_interpreter.nir_to_torch(graph, {nir.LI: _map_li})
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
    module = nir_interpreter.nir_to_torch(graph, {})
    assert torch.allclose(module.nirgraph.nirgraph.linear.weight, torch.from_numpy(w))
    out = module(torch.ones(3))
    assert isinstance(out, typing.Tuple)
    assert out[0].shape == (2,)


def test_can_overwrite_default_map():
    w = np.random.random((2, 3)).astype(np.float32)
    linear = nir.Linear(w)
    graph = nir.NIRGraph.from_list(linear)
    state = {"called": False}

    def mock_linear_map(node: nir.Linear):
        state["called"] = True
        return torch.nn.Linear(2, 3)

    nir_interpreter.nir_to_torch(graph, {nir.Linear: mock_linear_map})
    assert state["called"]


def test_map_out_of_order():
    w = np.random.random((3, 2)).astype(np.float32)
    nodes = {
        "linear": nir.Linear(w),
        "input": nir.Input(np.array([2])),
        "output": nir.Output(np.array([3])),
    }
    edges = [("input", "linear"), ("linear", "output")]
    graph = nir.NIRGraph(nodes, edges)
    module = nir_interpreter.nir_to_torch(graph, {nir.Linear: _map_linear_node})
    data = torch.rand(2)
    assert torch.allclose(module(data)[0], torch.from_numpy(w) @ data)


def test_map_recursive_graph():
    w = np.random.random((2, 2)).astype(np.float32)
    nodes = {
        "input": nir.Input(np.array([2])),
        "linear": nir.Linear(w),
        "output": nir.Output(np.array([2])),
    }
    edges = [("linear", "linear"), ("input", "linear"), ("linear", "output")]
    graph = nir.NIRGraph(nodes, edges)
    module = nir_interpreter.nir_to_torch(graph, {nir.Linear: _map_linear_node})
    data = torch.rand(2)
    expected = data @ torch.from_numpy(w).T
    actual, state = module(data)
    assert torch.allclose(actual, expected)
    assert "linear_prev_output" in state
    actual2, _ = module(data, state)
    assert torch.allclose(actual2, (expected + data) @ torch.from_numpy(w).T)


def test_map_recursive_graph_two_nodes():
    w = np.random.random((2, 2)).astype(np.float32)
    nodes = {
        "input": nir.Input(np.array([2])),
        "linear1": nir.Linear(w),
        "linear2": nir.Linear(w),
        "output": nir.Output(np.array([2])),
    }
    edges = [
        ("linear1", "linear2"),
        ("linear2", "linear1"),
        ("input", "linear1"),
        ("linear2", "output"),
    ]
    graph = nir.NIRGraph(nodes, edges)
    module = nir_interpreter.nir_to_torch(graph, {nir.Linear: _map_linear_node})
    lin_module = torch.nn.Linear(2, 2, bias=False)
    lin_module.weight.data = torch.from_numpy(w)
    data = torch.rand(2)
    expected = lin_module(lin_module(data))
    actual, state = module(data)
    assert torch.allclose(actual, expected)
    assert torch.allclose(state["linear2_prev_output"], expected)
    actual2, _ = module(data, state)
    assert torch.allclose(actual2, lin_module(lin_module((expected + data))))


def test_find_recursive_inputs_multiple():
    node = "A"
    edges = set([("A", "B"), ("B", "C"), ("B", "A"), ("C", "A")])
    actual = nir_interpreter._find_recursive_inputs(node, edges)
    assert actual == set(["B", "C"])


def test_map_recursive_graph_multiple_nodes():
    w = np.random.random((2, 2)).astype(np.float32)
    nodes = {
        "input": nir.Input(np.array([2])),
        "linear1": nir.Linear(w),
        "linear2": nir.Linear(w),
        "linear3": nir.Linear(w),
        "output": nir.Output(np.array([2])),
    }
    edges = [
        ("linear1", "linear2"),
        ("linear2", "linear1"),
        ("linear2", "linear3"),
        ("linear3", "linear3"),
        ("linear3", "linear1"),
        ("input", "linear1"),
        ("linear2", "output"),
    ]
    """
    In -> 1 -> 2 -> Out
          ^---/ \---> 3 --\
          ^----------/ ^--/   
    """
    graph = nir.NIRGraph(nodes, edges)
    module = nir_interpreter.nir_to_torch(graph, {nir.Linear: _map_linear_node})
    lin_module = torch.nn.Linear(2, 2, bias=False)
    lin_module.weight.data = torch.from_numpy(w)
    data = torch.rand(2)
    expected = lin_module(lin_module(data))
    actual, state = module(data)
    assert torch.allclose(actual, expected)
    old_lin2_state = state["linear2_prev_output"]
    old_lin3_state = state["linear3_prev_output"]
    # Test that the states for lin2 and lin3 are set
    assert torch.allclose(old_lin2_state, expected)
    assert torch.allclose(old_lin3_state, lin_module(expected))
    actual2, _ = module(data, state)
    # Note: The state dictionary has now been updated!
    # Linear 3 uses its own old state and the new input from linear2
    assert torch.allclose(
        state["linear3_prev_output"],
        lin_module(old_lin3_state + state["linear2_prev_output"]),
    )
    # Test the output from linear 2 = linearity(lin3_state + linearity(lin3_state + data))
    lin_output = lin_module(old_lin3_state + lin_module(old_lin3_state + data))
    assert torch.allclose(actual2, lin_output)


##########################################
#### Integration tests
##########################################


def test_from_nir_fx():
    w = np.random.random((1, 2))
    g = nir.NIRGraph.from_list(nir.Linear(w))
    with warnings.catch_warnings(record=True) as warn:
        m = nir_interpreter.nir_to_torch(g, _torch_node_map)
        assert len(warn) == 0

    assert m(torch.empty(2))[0].shape == (1,)


def test_import_lif_new_api():
    g = nir.read("tests/lif_norse.nir")
    m = nir_interpreter.nir_to_torch(g, _torch_node_map)
    assert m(torch.empty(1))[0].shape == (1,)


def test_import_lif_new_api_string():
    m = nir_interpreter.nir_to_torch("tests/lif_norse.nir", _torch_node_map)
    assert m(torch.empty(1))[0].shape == (1,)

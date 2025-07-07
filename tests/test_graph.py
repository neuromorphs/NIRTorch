import nir
import pytest
import torch
import torch.nn as nn

# from norse.torch import LIBoxCell, LIFCell, SequentialState
from sinabs.layers import Merge

from nirtorch import extract_nir_graph
from nirtorch.graph import TorchGraph, named_modules_map, GraphTracer


class TupleModule(torch.nn.Module):
    def forward(self, data):
        return (data, data)


# Branched model
class SinabsBranchedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.relu2_1 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.add_mod = Merge()
        self.relu3 = nn.ReLU()

    def forward(self, data):
        out1 = self.relu1(data)
        out2_1 = self.relu2_1(out1)
        out2_2 = self.relu2_2(out1)
        out3 = self.add_mod(out2_1, out2_2)
        out4 = self.relu3(out3)
        return out4


"""
class NorseStatefulModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.lif = SequentialState(LIFCell())

    def forward(self, data):
        out1 = self.relu1(data)
        out2, _ = self.lif(out1)
        return out2
"""

input_type = (2, 28, 28)
batch_size = 1

data = torch.ones((batch_size, *input_type))

my_branched_model = SinabsBranchedModel()


class DeepModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block1 = SinabsBranchedModel()
        self.block2 = SinabsBranchedModel()

    def forward(self, data):
        out = self.block1(data)
        out2 = self.block2(out)
        return out2


mydeepmodel = DeepModel()


def test_named_modules_map():
    from nirtorch.graph import named_modules_map

    mod_map = named_modules_map(my_branched_model)
    print(mod_map)
    for k, v in mod_map.items():
        assert isinstance(k, nn.Module)
        assert isinstance(v, str)


def test_module_forward_wrapper():
    mymodel = SinabsBranchedModel()

    orig_call = nn.Module.__call__

    from nirtorch.graph import TorchGraph, module_forward_wrapper, named_modules_map

    output_types = {}
    model_graph = TorchGraph(named_modules_map(mymodel), ["block1"])
    new_call = module_forward_wrapper(model_graph, output_types)

    # Override call to the new wrapped call
    nn.Module.__call__ = new_call

    with torch.no_grad():
        _ = mymodel(data)

    # Restore normal behavior
    nn.Module.__call__ = orig_call

    print(model_graph)
    assert (
        len(model_graph.node_list) == 1 + 5 + 5 + 1
    )  # 1 top module + 5 submodules + 5 tensors + 1 output tensor
    assert len(output_types) == 6  # 1 top module + 5 submodules


def test_graph_trace_one():
    model = nn.ReLU()
    module_map = named_modules_map(model)
    sample_data = torch.rand((1,))
    with GraphTracer(module_map) as tracer, torch.no_grad():
        _ = model(sample_data)
    assert len(tracer.graph.node_list) == 1
    assert tracer.graph.node_list[0].elem == model


def test_graph_trace_many():
    from nirtorch.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(my_branched_model)) as tracer, torch.no_grad():
        _ = my_branched_model(data)

    assert len(tracer.graph.node_list) == 5  # 5 submodules
    assert tracer.graph.get_edges() == [
        ("relu1", "relu2_1"),
        ("relu1", "relu2_2"),
        ("relu2_1", "add_mod"),
        ("relu2_2", "add_mod"),
        ("add_mod", "relu3"),
    ]


def test_leaf_only_graph():
    from nirtorch.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(mydeepmodel)) as tracer, torch.no_grad():
        _ = mydeepmodel(data)

    print(tracer.graph)

    # Get graph with just the leaf nodes
    leaf_graph = tracer.graph.leaf_only()
    print(leaf_graph)
    assert (
        len(leaf_graph.node_list) == len(tracer.graph.node_list) - 2
    )  # No more top modules


def test_ignore_submodules_of():
    from nirtorch.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(mydeepmodel)) as tracer, torch.no_grad():
        _ = mydeepmodel(data)

    top_overview_graph = tracer.graph.ignore_submodules_of(
        [SinabsBranchedModel]
    ).leaf_only()
    assert len(top_overview_graph.node_list) == 2


def test_snn_branched():
    from sinabs.layers import ChannelShift, IAFSqueeze, SumPool2d
    from torch.nn import Conv2d

    from nirtorch.graph import extract_torch_graph

    class MySNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = Conv2d(2, 8, 3, bias=False)
            self.iaf1 = IAFSqueeze(batch_size=1)
            self.pool1 = SumPool2d(2)
            self.conv2_1 = Conv2d(8, 16, 3, stride=1, padding=1, bias=False)
            self.iaf2_1 = IAFSqueeze(batch_size=1)
            self.pool2_1 = SumPool2d(2)
            self.conv2_2 = Conv2d(8, 16, 5, stride=1, padding=2, bias=False)
            self.iaf2_2 = IAFSqueeze(batch_size=1)
            self.pool2_2 = SumPool2d(2)
            self.channel_shift = ChannelShift(channel_shift=16)
            self.merge = Merge()
            self.conv3 = Conv2d(32, 10, 3, stride=3, bias=False)
            self.iaf3 = IAFSqueeze(batch_size=1)

        def forward(self, spikes):
            out = self.conv1(spikes)
            out = self.iaf1(out)
            out = self.pool1(out)

            out1 = self.conv2_1(out)
            out1 = self.iaf2_1(out1)
            out1 = self.pool2_1(out1)

            out2 = self.conv2_2(out)
            out2 = self.iaf2_2(out2)
            out2 = self.pool2_2(out2)

            out = self.merge(out1, self.channel_shift(out2))
            out = self.conv3(out)
            out = self.iaf3(out)
            return out

    my_snn = MySNN()
    graph = extract_torch_graph(
        my_snn, sample_data=torch.rand((100, 2, 14, 14)), model_name=None
    )

    assert len(graph.node_list) == 13


"""
@pytest.mark.skip(reason="Norse currently missing CuBaLIF v_reset __init__ parameter")
def test_input_output():
    from norse.torch import to_nir as norse_to_nir

    g = norse_to_nir(NorseStatefulModel(), data)
    assert len(g.nodes) == 4  # in -> relu -> lif -> out
    assert len(g.edges) == 3
"""


def test_output_type_when_single_node():
    g = extract_nir_graph(
        torch.nn.ReLU(),
        lambda x: nir.Threshold(torch.tensor(0.1)),
        sample_data=torch.rand((1,)),
    )
    assert "input" in g.nodes
    assert "output" in g.nodes
    g.nodes["output"].output_type["output"] == torch.Size([1])


def test_sequential_flatten():
    d = torch.empty(3, 4)
    g = extract_nir_graph(torch.nn.Flatten(1), lambda x: nir.Flatten(d.shape, 1), d)
    assert tuple(g.nodes["input"].input_type["input"]) == (3, 4)

    d = torch.empty(2, 3, 4)
    g = extract_nir_graph(
        torch.nn.Flatten(1), lambda x: nir.Flatten(d.shape, 1), d, ignore_dims=[0]
    )
    assert tuple(g.nodes["input"].input_type["input"]) == (3, 4)


"""
@pytest.mark.skip(reason="Not supported yet")
def test_captures_recurrence_automatically():
    class Recmodel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.r = LIBoxCell()
            self.l = torch.nn.Linear(1, 1)

        def forward(self, x, state=None):
            if state is None:
                state = [None, torch.zeros_like(x)]
            s1, s2 = state  # s1 = LIF, s2 = recurrence
            out, s1 = self.r(x + s2, s1)
            s2 = self.l(out)
            return out, [s1, s2]

    from norse.torch.utils.export_nir import _extract_norse_module

    m = Recmodel()
    data = torch.randn((1, 1))
    actual, actual_state = m(*m(data))
    d = extract_nir_graph(m, _extract_norse_module, data)
    assert d.nodes.keys() == {"input", "l", "r", "output"}
    assert set(d.edges) == {("input", "r"), ("r", "l"), ("l", "output"), ("r", "r")}
"""


@pytest.mark.skip(reason="Subgraphs are currently flattened")
def test_captures_recurrence_manually():
    def export_affine_rec_gru(module):
        if isinstance(module, torch.nn.Linear):
            return nir.Affine(module.weight, module.bias)
        elif isinstance(module, torch.nn.ReLU):
            return nir.NIRGraph(nodes={"i": nir.I(torch.randn(1))}, edges=[("i", "i")])
        else:
            raise ValueError(f"Unsupported module {module}")

    m = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())
    data = torch.randn((1, 1))

    d = extract_nir_graph(m, export_affine_rec_gru, data)
    assert d.nodes.keys() == {"input", "0", "1.i", "output"}
    assert set(d.edges) == {("input", "0"), ("0", "1"), ("1.i", "1.i"), ("1", "output")}
    # TODO: Pass recursively
    # assert d.nodes["1"].nodes.keys() == {"i"}
    # assert d.nodes["1"].edges == [("1.i", "1.i")]


def test_graph_captures_input():
    model = nn.Sequential(
        nn.Conv2d(2, 8, 3),
        nn.AvgPool2d(2),
        nn.ReLU(),
    )

    modules = {module: name for name, module in list(model.named_modules())[1:]}
    g = TorchGraph(modules, ["0"])
    assert len(g.inputs) == 1
    assert g.inputs[0] == "0"

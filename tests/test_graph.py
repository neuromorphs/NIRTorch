import pytest
import torch
import torch.nn as nn
from norse.torch import LIFCell, SequentialState
from sinabs.layers import Merge


class TupleModule(torch.nn.Module):
    def forward(self, data):
        return (data, data)


def test_sequential_graph_extract():
    from nirtorch.graph import extract_torch_graph

    model = nn.Sequential(
        nn.Conv2d(2, 8, 3),
        nn.AvgPool2d(2),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3),
        nn.AvgPool2d(2),
        nn.ReLU(),
        nn.Conv2d(16, 32, 2),
        nn.AvgPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128, 10),
    )

    sample_data = torch.rand(1, 2, 32, 32)
    graph = extract_torch_graph(
        model=model, sample_data=sample_data, model_name=None
    ).ignore_tensors()
    print(graph)

    assert len(graph.node_list) == 11


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


class NorseStatefulModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.lif = SequentialState(LIFCell())

    def forward(self, data):
        out1 = self.relu1(data)
        out2, _ = self.lif(out1)
        return out2


input_shape = (2, 28, 28)
batch_size = 1

data = torch.ones((batch_size, *input_shape))

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

    from nirtorch.graph import Graph, module_forward_wrapper, named_modules_map

    model_graph = Graph(named_modules_map(mymodel))
    new_call = module_forward_wrapper(model_graph)

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


def test_graph_tracer():
    from nirtorch.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(my_branched_model)) as tracer, torch.no_grad():
        _ = my_branched_model(data)

    print(tracer.graph)
    assert (
        len(tracer.graph.node_list) == 1 + 5 + 5 + 1
    )  # 1 top module + 5 submodules + 5 tensors + 1 output tensor


def test_leaf_only_graph():
    from nirtorch.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(mydeepmodel)) as tracer, torch.no_grad():
        _ = mydeepmodel(data)

    print(tracer.graph)

    # Get graph with just the leaf nodes
    leaf_graph = tracer.graph.leaf_only()
    print(leaf_graph)
    assert (
        len(leaf_graph.node_list) == len(tracer.graph.node_list) - 3
    )  # No more top modules


def test_ignore_submodules_of():
    from nirtorch.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(mydeepmodel)) as tracer, torch.no_grad():
        _ = mydeepmodel(data)

    top_overview_graph = tracer.graph.ignore_submodules_of(
        [SinabsBranchedModel]
    ).leaf_only()
    print(top_overview_graph)
    assert len(top_overview_graph.node_list) == 2 + 2 + 1


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

    print(graph)
    assert len(graph.node_list) == 27  # 2*13 + 1


def test_snn_stateful():
    from nirtorch.graph import extract_torch_graph

    model = NorseStatefulModel()
    graph = extract_torch_graph(model, sample_data=torch.rand((1, 2, 3, 4)))
    assert len(graph.node_list) == 7  # 2 + 1 nested + 4 tensors


def test_ignore_tensors():
    from nirtorch.graph import extract_torch_graph

    graph = extract_torch_graph(my_branched_model, sample_data=data)
    mod_only_graph = graph.ignore_tensors()
    assert len(mod_only_graph.node_list) == 6


def test_root_has_no_source():
    from nirtorch.graph import extract_torch_graph

    graph = extract_torch_graph(my_branched_model, sample_data=data)
    graph = graph.ignore_tensors()
    assert (
        len(graph.find_source_nodes_of(graph.find_node(my_branched_model.relu1))) == 0
    )


def test_get_root():
    from nirtorch.graph import extract_torch_graph

    graph = extract_torch_graph(my_branched_model, sample_data=data, model_name=None)
    graph = graph.ignore_tensors()
    root_nodes = graph.get_root()
    assert len(root_nodes) == 1
    assert root_nodes[0].elem == my_branched_model.relu1


def test_ignore_nodes_parent_model():
    from nirtorch.graph import extract_torch_graph

    graph = extract_torch_graph(
        my_branched_model, sample_data=data, model_name="ShouldDisappear"
    )

    new_graph = graph.ignore_nodes(SinabsBranchedModel)
    print(new_graph)

    with pytest.raises(ValueError):
        new_graph.find_node(my_branched_model)

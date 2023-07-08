from typing import Callable, Optional

import torch.nn as nn
import nir
from .graph import Graph, Node


class ExtractedModel(nn.Module):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph = graph
        root_nodes = graph.get_root()
        print(root_nodes)
        assert len(root_nodes) == 1, "Currently, we only support one input node"
        self.root_node = root_nodes[0]

    def apply_recursive(self, m: Node, *args):
        y = m.elem(*args)
        if len(m.outgoing_nodes) == 0:
            return y
        output = []
        for child in m.outgoing_nodes():
            if isinstance(y, tuple) or isinstance(y, list):
                output.append(self.apply_recursive(child, *y))
            else:
                output.append(self.apply_recursive(child, y))
        return output
        

    def forward(self, x):
        return self.root_node(x)


def _convert_number_to_legal_variable_name(num: int) -> str:
    return f"mod_{num}"


def _mod_nir_to_graph(nir_graph: nir.NIR) -> Graph:
    module_names = {
        module: _convert_number_to_legal_variable_name(idx)
        for idx, module in enumerate(nir_graph.nodes)
    }
    graph = Graph(module_names=module_names)
    for src, dst in nir_graph.edges:
        graph.add_edge(nir_graph.nodes[src], nir_graph.nodes[dst])
    return graph


def _switch_models_with_map(
    nir_graph: nir.NIR, model_map: Callable[[nir.NIRNode], Optional[nn.Module]]
) -> nir.NIR:
    nodes = [model_map(node) for node in nir_graph.nodes]
    nodes = [x for x in nodes if x is not None]
    return nir.NIR(nodes, nir_graph.edges)


def load(nir_graph: nir.NIR, model_map: Callable[[nir.NIRNode], Optional[nn.Module]]) -> nn.Module:
    """Load a NIR object and convert it to a torch module using the given model map

    Args:
        nir_graph (nir.NIR): NIR object
        model_map (Callable[[nn.NIR], nn.Module]): A method that returns the a torch 
            module that corresponds to each NIR node.

    Returns:
        nn.Module: The generated torch module
    """
    # Map modules to the target modules using th emodel map
    nir_module_graph = _switch_models_with_map(nir_graph, model_map)
    # Build a nirtorch.Graph based on the nir_graph
    graph = _mod_nir_to_graph(nir_module_graph)
    # Build and return ExtractedModule
    model = ExtractedModel(graph)
    return model

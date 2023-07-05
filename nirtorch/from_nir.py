import torch.nn as nn
import nir
from typing import Callable
from .graph import Graph


class ExtractedModel(nn.Module):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph = graph

    def instantiate_modules(self, graph: Graph):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


def _convert_number_to_legal_variable_name(num: int) -> str:
    return f"mod_{num}"


def _mod_nir_to_graph(nir_graph: nir.NIR) -> Graph:
    module_names = {
        module: _convert_number_to_legal_variable_name(idx)
        for idx, module in enumerate(nir_graph.nodes)
    }
    graph = Graph(module_names=module_names)
    for src, dst in nir_graph.edges:
        graph.add_edge(src, dst)
    return graph


def _switch_models_with_map(
    nir_graph: nir.NIR, model_map: Callable[[nn.Module], nn.Module]
) -> nir.NIR:
    nodes = [model_map(node) for node in nir_graph.nodes]
    return nir.NIR(nodes, nir_graph.edges)


def load(nir_graph: nir.NIR, model_map: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """Load a NIR object and convert it to a torch module using the given model map

    Args:
        nir_graph (nir.NIR): NIR object
        model_map (Callable[[nn.Module], nn.Module]): A method that returns the a torch 
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

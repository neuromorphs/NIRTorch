import torch
import torch.nn as nn
import nir
from typing import Callable, List, Optional, Tuple
from .graph import Graph, Node


def execution_order_up_to_node(
    node: Node, graph: Graph, execution_order: List[Node]
) -> List[Node]:
    """Recursive function to evaluate execution order until a given node

    Args:
        node (Node): Execution order for the node of interest
        graph (Graph): Graph object describing the network
        execution_order (List[Node]): The current known execution order.

    Returns:
        List[Node]: Execution order
    """
    if len(execution_order) == list(graph.node_list):
        # All nodes are executed
        return execution_order
    for parent in graph.find_source_nodes_of(node):
        if parent not in execution_order:
            execution_order = execution_order_up_to_node(parent, graph, execution_order)
    # Finally since all parents are known and executed
    return execution_order + [node]


class GraphExecutor(nn.Module):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph = graph
        self.instantiate_modules()
        root_nodes = graph.get_root()
        if len(root_nodes) == 0:
            raise ValueError("No root node found in the graph")
        self.root_node = root_nodes[0]

    def instantiate_modules(self):
        for mod, name in self.graph.module_names.items():
            self.add_module(name, mod)

    def forward_recursive(self, node: Node, *args):
        y = node.elem(*args)
        if len(node.outgoing_nodes) == 0:
            return y
        output = []
        for child in node.outgoing_nodes:
            if isinstance(y, tuple):
                output.append(self.forward_recursive(child, *y))
            else:
                output.append(self.forward_recursive(child, y))
        return output

    def forward(self, data: torch.Tensor):
        # Note: We assume single input/output for now
        return self.forward_recursive(self.root_node, data)


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


def load(nir_graph: nir.NIR, model_map: Callable[[nir.NIR], nn.Module]) -> nn.Module:
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
    # Build and return a graph executor module
    return GraphExecutor(graph)

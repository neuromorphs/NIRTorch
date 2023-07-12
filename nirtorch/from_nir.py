from typing import Callable, List

import nir
import torch
import torch.nn as nn

from .graph import Graph, Node


def execution_order_up_to_node(
    node: Node, graph: Graph, execution_order: List[Node]
) -> List[Node]:
    """Recursive function to evaluate execution order until a given node.

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
        self.execution_order = self.get_execution_order()
        if len(self.execution_order) == 0:
            raise ValueError("Graph is empty")

    def get_execution_order(self) -> List[Node]:
        """Evaluate the execution order and instantiate that as a list."""
        execution_order = []
        # Then loop over all nodes and check that they are added to the execution order.
        for node in self.graph.node_list:
            if node not in execution_order:
                execution_order = execution_order_up_to_node(
                    node, self.graph, execution_order
                )
        return execution_order

    def instantiate_modules(self):
        for mod, name in self.graph.module_names.items():
            self.add_module(name, mod)

    def get_input_nodes(self) -> List[Node]:
        # NOTE: This is a hack. Should use the input nodes from NIR graph
        return self.graph.get_root()

    def forward(self, data: torch.Tensor):
        outs = {}
        # NOTE: This logic is not yet consistent for models with multiple input nodes
        for node in self.execution_order:
            input_nodes = self.graph.find_source_nodes_of(node)
            if len(input_nodes) == 0:
                # This is the root node
                outs[node.name] = node.elem(data)
            else:
                # Intermediate nodes
                input_data = (outs[node.name] for node in input_nodes)
                outs[node.name] = node.elem(*input_data)
        return outs[node.name]


def _convert_number_to_legal_variable_name(num: int) -> str:
    return f"mod_{num}"


def _mod_nir_to_graph(nir_graph: nir.NIRNode) -> Graph:
    module_names = {
        module: _convert_number_to_legal_variable_name(idx)
        for idx, module in enumerate(nir_graph.nodes)
    }
    graph = Graph(module_names=module_names)
    for src, dst in nir_graph.edges:
        graph.add_edge(nir_graph.nodes[src], nir_graph.nodes[dst])
    return graph


def _switch_models_with_map(
    nir_graph: nir.NIRNode, model_map: Callable[[nn.Module], nn.Module]
) -> nir.NIRNode:
    nodes = [model_map(node) for node in nir_graph.nodes]
    return nir.NIRGraph(nodes, nir_graph.edges)


def load(
    nir_graph: nir.NIRNode, model_map: Callable[[nir.NIRNode], nn.Module]
) -> nn.Module:
    """Load a NIR object and convert it to a torch module using the given model map.

    Args:
        nir_graph (nir.NIRNode): NIR object
        model_map (Callable[[nn.NIRNode], nn.Module]): A method that returns the a torch
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
    # Build and return a graph executor module
    return GraphExecutor(graph)

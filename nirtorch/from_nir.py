import dataclasses
import inspect
from typing import Callable, Dict, List, Optional, Any, Union

import nir
import torch
import torch.nn as nn

from .graph import Graph, Node
from .utils import sanitize_name


def execution_order_up_to_node(
    node: Node,
    graph: Graph,
    execution_order: List[Node],
    visited: Optional[Dict[Node, bool]] = None,
) -> List[Node]:
    """Recursive function to evaluate execution order until a given node.

    Args:
        node (Node): Execution order for the node of interest
        graph (Graph): Graph object describing the network
        execution_order (List[Node]): The current known execution order.

    Returns:
        List[Node]: Execution order
    """
    if visited is None:
        visited = {n: False for n in graph.node_list}
    is_recursive = False
    if len(execution_order) == list(graph.node_list):
        # All nodes are executed
        return execution_order
    for parent in graph.find_source_nodes_of(node):
        if parent not in execution_order and not visited[parent]:
            visited[parent] = True
            execution_order = execution_order_up_to_node(
                parent, graph, execution_order, visited
            )
        if node in parent.outgoing_nodes:
            is_recursive = True
    # Ensure we're not re-adding a recursive node
    if is_recursive and node in execution_order:
        return execution_order
    else:  # Finally since all parents are known and executed
        return execution_order + [node]


@dataclasses.dataclass
class GraphExecutorState:
    """State for the GraphExecutor that keeps track of both the state of hidden
    units and caches the output of previous modules, for use in (future) recurrent
    computations."""

    state: Dict[str, Any] = dataclasses.field(default_factory=dict)
    cache: Dict[str, Any] = dataclasses.field(default_factory=dict)


class GraphExecutor(nn.Module):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph = graph
        self.stateful_modules = {}
        self.instantiate_modules()
        self.execution_order = self.get_execution_order()
        if len(self.execution_order) == 0:
            raise ValueError("Graph is empty")

    def _is_module_stateful(self, module: torch.nn.Module) -> bool:
        signature = inspect.signature(module.forward)
        arguments = len(signature.parameters)
        return arguments > 1

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
            if mod is not None:
                self.add_module(sanitize_name(name), mod)
                self.stateful_modules[sanitize_name(name)] = self._is_module_stateful(
                    mod
                )

    def get_input_nodes(self) -> List[Node]:
        # NOTE: This is a hack. Should use the input nodes from NIR graph
        return self.graph.get_root()

    def _apply_module(
        self,
        node: Node,
        input_nodes: List[Node],
        old_state: GraphExecutorState,
        new_state: GraphExecutorState,
        data: Optional[torch.Tensor] = None,
    ):
        """Applies a module and keeps track of its state.
        TODO: Use pytree to recursively construct the state
        """
        inputs = []
        # Append state if needed
        if node.name in self.stateful_modules and node.name in old_state.state:
            inputs.extend(old_state.state[node.name])

        # Sum recurrence if needed
        summed_inputs = [] if data is None else [data]
        for input_node in input_nodes:
            if (
                input_node.name not in new_state.cache
                and input_node.name in old_state.cache
            ):
                summed_inputs.append(old_state.cache[input_node.name])
            elif input_node.name in new_state.cache:
                summed_inputs.append(new_state.cache[input_node.name])

        if len(input_nodes) == 0 and data is not None:
            # single input, no need to sum (fix issue #16)
            inputs.insert(0, data)
        elif len(input_nodes) == 1:
            # single input, no need to sum (fix issue #16)
            inputs.insert(0, summed_inputs[0])
        else:
            # multiple inputs, sum them
            inputs.insert(0, torch.stack(summed_inputs).sum(0))

        out = node.elem(*inputs)
        # If the module is stateful, we know the output is (at least) a tuple
        if self.stateful_modules[node.name]:
            new_state.state[node.name] = out[1:]  # Store the new state
            out = out[0]
        return out, new_state

    def forward(
        self, data: torch.Tensor, old_state: Optional[GraphExecutorState] = None
    ):
        if old_state is None:
            old_state = GraphExecutorState()
        new_state = GraphExecutorState()
        first_node = True
        # NOTE: This logic is not yet consistent for models with multiple input nodes
        for node in self.execution_order:
            input_nodes = self.graph.find_source_nodes_of(node)
            if node.elem is None:
                continue
            out, new_state = self._apply_module(
                node, input_nodes, new_state, old_state, data if first_node else None
            )
            new_state.cache[node.name] = out
            first_node = False

        # If the output node is a dummy nir.Output node, use the second-to-last node
        if node.name not in new_state.cache:
            node = self.execution_order[-2]
        return new_state.cache[node.name], new_state


def _mod_nir_to_graph(nir_graph: nir.NIRNode) -> Graph:
    module_names = {module: name for name, module in nir_graph.nodes.items()}
    graph = Graph(module_names=module_names)
    for src, dst in nir_graph.edges:
        graph.add_edge(nir_graph.nodes[src], nir_graph.nodes[dst])
    return graph


def _switch_models_with_map(
    nir_graph: nir.NIRNode, model_map: Callable[[nn.Module], nn.Module]
) -> nir.NIRNode:
    nodes = {name: model_map(node) for name, node in nir_graph.nodes.items()}
    return nir.NIRGraph(nodes, nir_graph.edges)


def load(
    nir_graph: Union[nir.NIRNode, str], model_map: Callable[[nir.NIRNode], nn.Module]
) -> nn.Module:
    """Load a NIR graph and convert it to a torch module using the given model map.

    Args:
        nir_graph (Union[nir.NIRNode, str]): The NIR object to load, or a string representing
            the path to the NIR object.
        model_map (Callable[[nn.NIRNode], nn.Module]): A method that returns the a torch
            module that corresponds to each NIR node.

    Returns:
        nn.Module: The generated torch module
    """
    if isinstance(nir_graph, str):
        nir_graph = nir.read(nir_graph)
    # Map modules to the target modules using th emodel map
    nir_module_graph = _switch_models_with_map(nir_graph, model_map)
    # Build a nirtorch.Graph based on the nir_graph
    graph = _mod_nir_to_graph(nir_module_graph)
    # Build and return a graph executor module
    return GraphExecutor(graph)

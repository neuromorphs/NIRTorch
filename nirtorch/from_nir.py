import dataclasses
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import nir
import torch
import torch.nn as nn

from .graph import Graph, Node
from .graph_utils import trace_execution
from .utils import sanitize_name


@dataclasses.dataclass
class GraphExecutorState:
    """State for the GraphExecutor that keeps track of both the state of hidden units
    and caches the output of previous modules, for use in (future) recurrent
    computations."""

    state: Dict[str, Any] = dataclasses.field(default_factory=dict)
    cache: Dict[str, Any] = dataclasses.field(default_factory=dict)


class GraphExecutor(nn.Module):
    """Executes the NIR graph in PyTorch.

    By default the graph executor is stateful, since there may be recurrence or
    stateful modules in the graph. Specifically, that means accepting and returning a
    state object (`GraphExecutorState`). If that is not desired,
    set `return_state=False` in the constructor.

    Arguments:
        graph (Graph): The graph to execute
        return_state (bool, optional): Whether to return the state object.
            Defaults to True.

    Raises:
        ValueError: If there are no edges in the graph
    """

    def __init__(self, graph: Graph, return_state: bool = True) -> None:
        super().__init__()
        self.graph = graph
        self.stateful_modules = set()
        self.return_state = return_state
        self.instantiate_modules()
        self.execution_order = self.get_execution_order()
        if len(self.execution_order) == 0:
            raise ValueError("Graph is empty")

    def _is_module_stateful(self, module: torch.nn.Module) -> bool:
        signature = inspect.signature(module.forward)
        arguments = len(signature.parameters)
        # HACK for snntorch modules
        if "snntorch" in str(module.__class__):
            if module.__class__.__name__ in [
                "Synaptic",
                "RSynaptic",
                "Leaky",
                "RLeaky",
            ]:
                return not module.init_hidden
        return "state" in signature.parameters and arguments > 1

    def get_execution_order(self) -> List[Node]:
        """Evaluate the execution order and instantiate that as a list."""
        # TODO: Adapt this for graphs with multiple inputs
        inputs = self.graph.inputs
        if len(inputs) != 1:
            raise ValueError(
                f"Currently, only one input is supported, but {len(inputs)} was given"
            )
        return trace_execution(inputs[0], lambda n: n.outgoing_nodes.keys())

    def instantiate_modules(self):
        for mod, name in self.graph.module_names.items():
            if mod is not None:
                self.add_module(sanitize_name(name), mod)
                if self._is_module_stateful(mod):
                    self.stateful_modules.add(sanitize_name(name))

    def get_input_nodes(self) -> List[Node]:
        # NOTE: This is a hack. Should use the input nodes from NIR graph
        return self.graph.get_root()

    def _apply_module(
        self,
        node: Node,
        input_nodes: List[Node],
        new_state: GraphExecutorState,
        old_state: GraphExecutorState,
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

        if len(summed_inputs) == 0:
            raise ValueError("No inputs found for node {}".format(node.name))
        elif len(summed_inputs) == 1:
            inputs.insert(0, summed_inputs[0])
        elif len(summed_inputs) > 1:
            inputs.insert(0, torch.stack(summed_inputs).sum(0))

        out = node.elem(*inputs)
        # If the module is stateful, we know the output is (at least) a tuple
        # HACK to make it work for snnTorch
        is_rsynaptic = "snntorch._neurons.rsynaptic.RSynaptic" in str(
            node.elem.__class__
        )
        if is_rsynaptic and not node.elem.init_hidden:
            assert "lif" in node.name, "this shouldnt happen.."
            new_state.state[node.name] = out  # snnTorch requires output inside state
            out = out[0]
        elif node.name in self.stateful_modules:
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
                node,
                input_nodes,
                new_state=new_state,
                old_state=old_state,
                data=data if first_node else None,
            )
            new_state.cache[node.name] = out
            first_node = False

        # If the output node is a dummy nir.Output node, use the second-to-last node
        if node.name not in new_state.cache:
            node = self.execution_order[-2]
        if self.return_state:
            return new_state.cache[node.name], new_state
        else:
            return new_state.cache[node.name]


def _mod_nir_to_graph(
    torch_graph: nir.NIRGraph, nir_nodes: Dict[str, nir.NIRNode]
) -> Graph:
    module_names = {module: name for name, module in torch_graph.nodes.items()}
    inputs = [name for name, node in nir_nodes.items() if isinstance(node, nir.Input)]
    graph = Graph(module_names=module_names, inputs=inputs)
    for src, dst in torch_graph.edges:
        # Allow edges to refer to subgraph inputs and outputs
        if src not in torch_graph.nodes and f"{src}.output" in torch_graph.nodes:
            src = f"{src}.output"
        if dst not in torch_graph.nodes and f"{dst}.input" in torch_graph.nodes:
            dst = f"{dst}.input"
        graph.add_edge(torch_graph.nodes[src], torch_graph.nodes[dst])
    return graph


def _switch_default_models(nir_graph: nir.NIRNode) -> Optional[torch.nn.Module]:
    if isinstance(nir_graph, nir.Input) or isinstance(nir_graph, nir.Output):
        return torch.nn.Identity()


def _switch_models_with_map(
    nir_graph: nir.NIRNode, model_map: Callable[[nn.Module], nn.Module]
) -> nir.NIRNode:
    nodes = {}
    for name, node in nir_graph.nodes.items():
        mapped_module = model_map(node)
        if mapped_module is None:
            mapped_module = _switch_default_models(node)
        nodes[name] = mapped_module
    # nodes = {name: model_map(node) for name, node in nir_graph.nodes.items()}
    return nir.NIRGraph(nodes, nir_graph.edges)


def load(
    nir_graph: Union[nir.NIRNode, str],
    model_map: Callable[[nir.NIRNode], nn.Module],
    return_state: bool = True,
) -> nn.Module:
    """Load a NIR graph and convert it to a torch module using the given model map.

    Because the graph can contain recurrence and stateful modules, the execution accepts
    a secondary state argument and returns a tuple of [output, state], instead of just
    the output as follows

    >>> executor = nirtorch.load(nir_graph, model_map)
    >>> old_state = None
    >>> output, state = executor(input, old_state) # Notice second argument and output
    >>> output, state = executor(input, state) # This can go on for many (time)steps

    If you do not wish to operate with state, set `return_state=False`.

    Args:
        nir_graph (Union[nir.NIRNode, str]): The NIR object to load, or a string
            representing the path to the NIR object.
        model_map (Callable[[nn.NIRNode], nn.Module]): A method that returns the a torch
            module that corresponds to each NIR node.
        return_state (bool): If True, the execution of the loaded graph will return a
            tuple of [output, state], where state is a GraphExecutorState object.
            If False, only the NIR graph output will be returned. Note that state is
            required for recurrence to work in the graphs.

    Returns:
        nn.Module: The generated torch module
    """
    if isinstance(nir_graph, str):
        nir_graph = nir.read(nir_graph)
    # Map modules to the target modules using th emodel map
    nir_module_graph = _switch_models_with_map(nir_graph, model_map)
    # Build a nirtorch.Graph based on the nir_graph
    graph = _mod_nir_to_graph(nir_module_graph, nir_nodes=nir_graph.nodes)
    # Build and return a graph executor module
    return GraphExecutor(graph, return_state=return_state)

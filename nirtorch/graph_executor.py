import dataclasses
import inspect
import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from .graph import TorchGraph, Node
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

    def __init__(self, graph: TorchGraph, return_state: bool = True) -> None:
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
        # Note that GraphExecutor is *also* a stateful module
        return "state" in signature.parameters and arguments > 1

    def get_execution_order(self) -> List[Node]:
        """Evaluate the execution order and instantiate that as a list."""
        inputs = set()
        name_dict = {name: node for node, name in self.graph.module_names.items()}
        for input_node in self.graph.inputs:
            if input_node.name in name_dict:
                inputs.add(input_node)
            else:
                logging.warning(
                    f"Input node {input_node.name} not found in module names. Skipping."
                )
        assert len(inputs) > 0, "No input nodes found, we require at least one."
        if len(inputs) > 1:
            logging.warning(
                "Multiple input nodes found. Using the first node as the input."
            )
        return trace_execution(next(iter(inputs)), lambda n: n.outgoing_nodes.keys())

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
        elif len(summed_inputs) == 1:  # Prepend the input
            inputs.insert(0, summed_inputs[0])
        elif len(summed_inputs) > 1:  # Prepend the sum of the inputs
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

    def forward(self, data: torch.Tensor, state: Optional[GraphExecutorState] = None):
        if state is None:
            state = GraphExecutorState()
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
                old_state=state,
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

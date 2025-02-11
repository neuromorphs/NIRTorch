import warnings
from typing import Callable, Optional, Union

import nir
import torch
import torch.nn as nn

from .graph import TorchGraph
from .graph_executor import GraphExecutor
from .interpreter import to_torch, NodeMapType
from .utils import sanitize_name


def _switch_default_models(nir_graph: nir.NIRNode) -> Optional[torch.nn.Module]:
    node = None
    if isinstance(nir_graph, nir.Input):
        node = torch.nn.Identity()
        node.node_type = "input"
    elif isinstance(nir_graph, nir.Output):
        node = torch.nn.Identity()
        node.node_type = "output"
    return node


def _map_graph_to_torch(
    nir_graph: nir.NIRNode,
    model_map: Callable[[nn.Module], nn.Module],
    return_state: bool = True,
) -> nir.NIRNode:
    nodes = {}
    for name, node in nir_graph.nodes.items():
        mapped_module = model_map(node)
        if mapped_module is None:
            if isinstance(node, nir.NIRGraph):  # Recurse
                mapped_module = _map_graph_to_torch(node, model_map, return_state)
            else:  # Map identity nodes (input, output, etc.)
                mapped_module = _switch_default_models(node)
        nodes[sanitize_name(name)] = mapped_module
    # Ensure that we use legal PyTorch names for the edges
    sanitized_edges = [
        (sanitize_name(src), sanitize_name(dst)) for src, dst in nir_graph.edges
    ]
    # Reconstruct NIR graph with torch modules
    recon_graph = nir.NIRGraph(nodes, sanitized_edges)
    # Build a TorchGraph for tracing and executing
    trace_graph = TorchGraph.from_torch_modules(recon_graph.nodes, recon_graph.edges)
    # Build and return a graph executor module
    return GraphExecutor(trace_graph, return_state=return_state)


def load(
    nir_graph: Union[nir.NIRNode, str],
    model_map: Union[Callable[[nir.NIRNode], nn.Module], NodeMapType],
    return_state: bool = True,
) -> nn.Module:
    """Load a NIR graph and convert it to a torch module using the given model map.

    Because the graph can contain recurrence and stateful modules.

    >>> map = {nir.LI: lambda li: MyLIModule(li.tau, li.r, li.v_leak)}
    >>> executor = nirtorch.load(nir_graph, model_map)
    >>> output, state = executor(input) # Note the state return value
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

    # TODO: Deprecate the old model map type
    if isinstance(model_map, Callable):
        warnings.warn(
            "You are using an old call to nirtorch.load. "
            "Please refer to https://github.com/neuromorphs/NIRTorch/pull/28 for instructions",
            DeprecationWarning,
        )
        # Convert the NIR graph to a torch graph
        return _map_graph_to_torch(nir_graph, model_map, return_state=return_state)
    else:
        if not return_state:
            warnings.warn(
                "return_state is deprecated and will be removed in future releases",
                DeprecationWarning,
            )
        return to_torch(nir_graph, node_map=model_map)

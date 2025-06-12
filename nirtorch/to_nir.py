import logging
from typing import Any, Callable, Optional, Sequence
import warnings

import nir
import numpy as np
import torch.nn as nn

from .graph import extract_torch_graph


def extract_nir_graph(
    model: nn.Module,
    model_map: Callable[[nn.Module], nir.NIRNode],
    sample_data: Any,
    model_name: Optional[str] = "model",
    ignore_submodules_of=None,
    model_fwd_args=[],
    ignore_dims: Optional[Sequence[int]] = None,
) -> nir.NIRNode:
    """
    DEPRECATED: Use `nirtorch.torch_to_nir` instead.

    Given a `model`, generate an NIR representation using the specified `model_map`.

    Assumptions and known issues:
        - Cannot deal with layers like torch.nn.Identity(), since the input tensor and
          output tensor will be the same object, and therefore lead to cyclic
          connections.

    Args:
        model (nn.Module): The model of interest
        model_map (Callable[[nn.Module], nir.NIRNode]): A method that converts a given
            module type to an NIRNode type
        sample_data (Any): Sample input data to be used for model extraction
        model_name (Optional[str], optional): The name of the top level module.
            Defaults to "model".
        ignore_submodules_of (Optional[Sequence[nn.Module]]): If specified,
            the corresponding module's children will not be traversed for graph.
        ignore_dims (Optional[Sequence[int]]): Dimensions of data to be ignored for
            type/shape inference. Typically the dimensions that you will want to ignore
            are for batch and time.
    Returns:
        nir.NIR: Returns the generated NIR graph representation.
    """
    warnings.warn(
        "nirtorch.extract_nir_graph is being deprecated in favour of nirtorch.torch_to_nir. "
        "Please refer to https://neuroir.org/docs/dev_pytorch.html for detailed instructions",
        DeprecationWarning,
    )

    if len(list(model.children())):
        # If the model has submodules, ignore the top level module
        model_name = None

    # Extract a torch graph given the model
    torch_graph = extract_torch_graph(
        model, sample_data=sample_data, model_name=model_name, model_args=model_fwd_args
    )

    if ignore_submodules_of is not None:
        torch_graph = torch_graph.ignore_submodules_of(ignore_submodules_of)

    # Convert the nodes and get indices
    nir_edges = []
    input_shape = np.array(sample_data.shape)
    if ignore_dims:
        nir_nodes = {"input": nir.Input(np.delete(input_shape, ignore_dims))}
    else:
        nir_nodes = {"input": nir.Input(input_shape)}
    nir_edges = []

    subgraph_keys = []
    subgraph_input_nodekeys = []
    subgraph_output_nodekeys = []
    # Get all the NIR nodes
    for indx, node in enumerate(torch_graph.node_list):
        # Convert the node type to NIR subgraph
        mapped_node = model_map(node.elem)

        if isinstance(mapped_node, nir.NIRGraph):
            subgraph_keys.append(node.name)
            for k, v in mapped_node.nodes.items():
                # For now, we add nodes in subgraphs to the top-level node list
                # TODO: support deeper nesting -> parse graphs recursively
                assert not isinstance(v, nir.NIRGraph), "cannot handle sub-sub-graphs"

                subgraph_node_key = f"{node.name}.{k}"

                # keep track of subgraph input and outputs (to remove later)
                if isinstance(v, nir.Input):
                    subgraph_input_nodekeys.append(subgraph_node_key)
                elif isinstance(v, nir.Output):
                    subgraph_output_nodekeys.append(subgraph_node_key)

                if isinstance(v, nir.NIRNode):
                    nir_nodes[subgraph_node_key] = v
                else:
                    nir_nodes[v.name] = v  # would this ever happen??
            # Add edges from graph
            for x, y in mapped_node.edges:
                nir_edges.append((f"{node.name}.{x}", f"{node.name}.{y}"))
        else:
            nir_nodes[node.name] = mapped_node

        # Add edges from input, if first element
        # TODO: Replace with mapping to input(s)/output(s) of subgraph
        if indx == 0:  # TODO:
            keys = list(nir_nodes.keys())
            for k1, k2 in zip(keys[:-1], keys[1:]):
                nir_edges.append((k1, k2))

    # Get all the edges
    for node in torch_graph.node_list:
        for destination, shape in node.outgoing_nodes.items():
            nir_edges.append((node.name, destination.name))

        if len(node.outgoing_nodes) == 0:
            out_name = "output"
            # Try to find shape of input to the Output node
            if ignore_dims:
                out_shape = np.delete(
                    torch_graph.module_output_types[node.elem], ignore_dims
                )
            else:
                out_shape = torch_graph.module_output_types[node.elem]
            output_node = nir.Output(out_shape)
            nir_nodes[out_name] = output_node
            nir_edges.append((node.name, out_name))

    # Remove duplicate edges
    nir_edges = list(set(nir_edges))

    # change edges to subgraph to point to either input or output of subgraph
    for idx in range(len(nir_edges)):
        if nir_edges[idx][0] in subgraph_keys:
            nir_edges[idx] = (f"{nir_edges[idx][0]}.output", nir_edges[idx][1])
        if nir_edges[idx][1] in subgraph_keys:
            nir_edges[idx] = (nir_edges[idx][0], f"{nir_edges[idx][1]}.input")

    # remove subgraph input and output nodes (& redirect edges)
    for rm_nodekey in subgraph_input_nodekeys + subgraph_output_nodekeys:
        in_keys = [e[0] for e in nir_edges if e[1] == rm_nodekey]
        out_keys = [e[1] for e in nir_edges if e[0] == rm_nodekey]
        # connect all incoming to all outgoing nodes
        for in_key in in_keys:
            for out_key in out_keys:
                nir_edges.append((in_key, out_key))
        # remove the original edges
        for in_key in in_keys:
            nir_edges.remove((in_key, rm_nodekey))
        for out_key in out_keys:
            nir_edges.remove((rm_nodekey, out_key))
        # remove the node
        nir_nodes.pop(rm_nodekey)

    # HACK: remove self-connections (this is a bug in the extraction of an RNN graph)
    for edge in nir_edges:
        if edge[0] == edge[1]:
            logging.warn(f"removing self-connection {edge}")
            nir_edges.remove(edge)

    return nir.NIRGraph(nir_nodes, nir_edges, type_check=False)

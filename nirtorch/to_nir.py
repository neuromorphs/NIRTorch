from typing import Any, Callable, Optional

import nir
import numpy as np
import torch.nn as nn

from .graph import extract_torch_graph


def extract_nir_graph(
    model: nn.Module,
    model_map: Callable[[nn.Module], nir.NIRNode],
    sample_data: Any,
    model_name: Optional[str] = "model",
) -> nir.NIRNode:
    """Given a `model`, generate an NIR representation using the specified `model_map`.

    Args:
        model (nn.Module): The model of interest
        model_map (Callable[[nn.Module], nir.NIRNode]): A method that converts a given
            module type to an NIRNode type
        sample_data (Any): Sample input data to be used for model extraction
        model_name (Optional[str], optional): The name of the top level module.
            Defaults to "model".

    Returns:
        nir.NIR: Returns the generated NIR graph representation.
    """

    if len(list(model.children())):
        # If the model has submodules, ignore the top level module
        model_name = None

    # Extract a torch graph given the model
    torch_graph = extract_torch_graph(
        model, sample_data=sample_data, model_name=model_name
    ).ignore_tensors()

    # Get the root node
    root_nodes = torch_graph.get_root()
    if len(root_nodes) != 1:
        raise ValueError(
            f"Currently, only one input is supported, but {len(root_nodes)} was given"
        )

    # Convert the nodes and get indices
    nir_edges = []
    nir_nodes = [nir.Input(np.array(sample_data.shape))]
    indices = {}

    # Get all the NIR nodes
    for indx, node in enumerate(torch_graph.node_list):
        # Convert the node type to NIR subgraph
        mapped_node = model_map(node.elem)

        if isinstance(mapped_node, nir.NIRGraph):
            current_node_index = len(nir_nodes)
            nir_nodes.extend(mapped_node.nodes)
            indices[node] = indx + len(mapped_node.nodes)
            # Add edges from graph
            for x, y in mapped_node.edges:
                nir_edges.append((x + current_node_index, y + current_node_index))
        else:
            nir_nodes.append(mapped_node)
            indices[node] = indx + 1

        # Add edges from input, if first element
        # TODO: Replace with mapping to input(s)/output(s) of subgraph
        if indx == 0:  # TODO:
            for sub_index in range(1, len(nir_nodes)):
                nir_edges.append((0, sub_index))

    outputs = []

    # Get all the edges
    for node in torch_graph.node_list:
        for destination in node.outgoing_nodes:
            nir_edges.append((indices[node], indices[destination]))
        if len(node.outgoing_nodes) == 0:
            output_node = nir.Output(None)
            outputs.append(output_node)
            nir_nodes.append(output_node)
            nir_edges.append((indices[node], len(nir_nodes) - 1))

    return nir.NIRGraph(nir_nodes, nir_edges)

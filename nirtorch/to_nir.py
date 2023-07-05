import torch.nn as nn
from typing import Any, Optional, Callable, List, Union
import nir
from .graph import extract_torch_graph


def extract_nir_graph(
    model: nn.Module,
    model_map: Callable[[nn.Module], Union[nir.NIRNode, List[nir.NIRNode]]],
    sample_data: Any,
    model_name: Optional[str] = "model",
) -> nir.NIR:
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

    # Convert the nodes and get indices
    edges = []
    nir_nodes = []
    indices = {}

    # Get all the NIR nodes
    for indx, node in enumerate(torch_graph.node_list):
        # Convert the node type
        mapped_node = model_map(node.elem)
        if isinstance(mapped_node, list):  # Node maps to multiple nodes
            nir_nodes.extend(mapped_node)
            # Add edges sequentially between the nodes
            for n_idx in range(len(mapped_node[:-1])):
                edges.append((indx + n_idx, indx + n_idx + 1))
        elif isinstance(mapped_node, nir.NIRNode):
            nir_nodes.append(mapped_node)

        indices[node] = indx

    # Get all the edges
    for node in torch_graph.node_list:
        for destination in node.outgoing_nodes:
            edges.append((indices[node], indices[destination]))

    return nir.NIR(nir_nodes, edges)

import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, Callable, List
import nir
from .graph import extract_torch_graph

def extract_nir_graph(
    model: nn.Module,
    model_map: Callable[[nn.Module], nir.NIRNode],
    sample_data: Any,
    model_name: Optional[str] = "model",
) -> nir.NIR:
    """Given a `model`, generate an NIR representation using the specified `model_map`.

    Args:
        model (nn.Module): The model of interest
        model_map (Callable[[nn.Module], nir.NIRNode]): A method that converts a given module type to an NIRNode type
        sample_data (Any): Sample input data to be used for model extraction
        model_name (Optional[str], optional): The name of the top level module. Defaults to "model".

    Returns:
        nir.NIR: Returns the generated NIR graph representation.
    """

    if len(list(model.children())):
        # If the model has submodules, ignore the top level module
        model_name = None

    # Extract a torch graph given the model
    torch_graph = extract_torch_graph(model, sample_data=sample_data, model_name=model_name).ignore_tensors()

    # Convert the nodes and get indices
    edges = []
    nir_nodes = []
    indices = {}

    # Get all the NIR nodes
    for indx, node in enumerate(torch_graph.node_list):
        # Convert the node type
        nir_nodes.append(model_map(node.elem))
        indices[node] = indx

    # Get all the edges
    for node in torch_graph.node_list:
        for destination in node.outgoing_nodes:
            edges.append((indices[node], indices[destination]))

    print(indices)
    print(nir_nodes)
    print(edges)

    return nir.NIR(nir_nodes, edges)


    

    

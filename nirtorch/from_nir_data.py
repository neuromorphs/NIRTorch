"""
Convert NIRGraphData to a torch-compatible spike representation.
"""

import torch
from nir.data_ir import NIRGraphData, NIRNodeData


def from_nir_data(
    nir_graph_data: NIRGraphData,
    dt: float,
    dynamic_before_transition: bool = True,
    dimension_order: tuple = ("time", "batch", "neuron"),
    time_unit: float = 1.0,
) -> dict:
    """
    Args:
        nir_graph_data: A NIRGraphData object containing the spike data.
        dt: The time step size of the spike data.
        dynamic_before_transition: bool, optional
            If True, it is assumed that the framework evolves the state (e.g.,
            membrane potential) of the neurons before checking if the threshold
            has been crossed and generating an event (transition). If False,
            the state is updated after the event generation.
        dimension_order: The desired ordering of dimensions in the output spike
            tensors. Defaults to ('time', 'batch', 'neuron').
            If the output spike tensors should have a different dimension
            ordering, this can be used to specify the correct ordering.
        time_unit: The unit of time for the spike data and dt. Defaults to 1.0,
            which corresponds to seconds. For milliseconds, set this to 1e-3.
    Returns:
        A dictionary mapping node names to spike tensors.
    """
    torch_dict = {}

    for node_key, nir_node_data in nir_graph_data.nodes.items():
        if isinstance(nir_node_data, NIRNodeData):
            for observable, data in nir_node_data.observables.items():
                if observable == "spikes":
                    data = data.get_time_gridded(
                        dt=dt * time_unit,
                        dimension_order=dimension_order,
                        dynamic_before_transition=dynamic_before_transition,
                    )
                    torch_spikes = torch.tensor(data.data).float()
                else:
                    raise NotImplementedError(
                        "Only spikes are supported as observables yet."
                    )
        else:
            raise NotImplementedError(
                "The translation of nested NIRGraphData is not supported."
            )

        torch_dict[node_key] = torch_spikes

    return torch_dict

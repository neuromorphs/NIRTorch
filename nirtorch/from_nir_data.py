"""
Convert NIRGraphData to a torch-compatible spike representation.
"""

import torch
from nir.data_ir import EventData, NIRGraphData, NIRNodeData


def from_nir_data(
    nir_graph_data: NIRGraphData,
    dt: float,
    dynamic_before_transition: bool = True,
    shape: tuple = ("T", "B", "N"),
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
        shape: The desired shape of the output spike tensors. Defaults to
            ('T', 'B', 'N') for (time, batch, neurons).
            If the output spike tensors should have a different shape, this can
            be used to specify the correct ordering of dimensions.
        time_unit: The unit of time for the spike data and dt. Defaults to 1.0,
            which corresponds to seconds. For milliseconds, set this to 1e-3.
    Returns:
        A dictionary mapping node names to spike tensors.
    """
    nirdata_shape = ("B", "T", "N")
    torch_dict = {}

    for node_key, nir_node_data in nir_graph_data.nodes.items():
        if isinstance(nir_node_data, NIRNodeData):
            for observable, data in nir_node_data.observables.items():
                if observable == "spikes":
                    if isinstance(data, EventData):
                        data = data.to_time_gridded(
                            dt=dt * time_unit,
                            dynamic_before_transition=dynamic_before_transition
                        )
                    else:
                        if data.dynamic_before_transition != dynamic_before_transition:
                            data = data.toggle_dynamic_before_transition()
                    torch_spikes = torch.tensor(data.data)
                    # reorder dimensions to shape specified by `shape` argument
                    torch_spikes = torch_spikes.permute(
                        nirdata_shape.index(shape[0]),
                        nirdata_shape.index(shape[1]),
                        nirdata_shape.index(shape[2]),
                    )
                else:
                    raise NotImplementedError(
                        "Only spikes are supported as observables yet."
                    )
        else:
            raise NotImplementedError(
                "The translation of nested NIRGraphData is not supported."
            )

        torch_dict[node_key] = torch_spikes.float()

    return torch_dict

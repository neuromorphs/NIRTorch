"""
Convert spikes from a torch tensor to a NIRGraphData object.
"""

from nir.data_ir import NIRGraphData, NIRNodeData, TimeGriddedData


def to_nir_data(
    torch_dict: dict,
    dt: float,
    time_unit: float = 1.0,
    dimension_order: tuple = ("time", "batch", "neuron"),
    dynamic_before_transition: bool = True,
) -> NIRGraphData:
    """
    Args:
        torch_dict: A dictionary mapping node names to spike tensors. If the
            shape is not (n_time_steps, batch_size, n_neurons), the `shape`
            argument can be used to specify the correct ordering of dimensions.
        dt: The time step size of the spike data.
        time_unit: The unit of time for the spike data and dt. Defaults to 1.0,
            which corresponds to seconds. For milliseconds, set this to 1e-3.
        dimension_order: The order of dimensions in the input spike tensors.
            Defaults to ("time", "batch", "neuron").
        dynamic_before_transition: If True, the membrane potential is updated
            before checking if the threshold has been crossed and generating an
            event (transition). If False, the state is updated after the event
            generation. Default is True.
    Returns:
        A NIRGraphData object containing the spike data.
    """
    nir_nodes = {}

    for key, spikes in torch_dict.items():
        spikes = spikes.detach().cpu().numpy().astype(bool)

        nir_node_data = NIRNodeData(
            {
                "spikes": TimeGriddedData(
                    spikes,
                    dt * time_unit,
                    dimension_order=dimension_order,
                    dynamic_before_transition=dynamic_before_transition,
                )
            }
        )
        nir_nodes[key] = nir_node_data

    nir_data = NIRGraphData(nir_nodes)
    return nir_data

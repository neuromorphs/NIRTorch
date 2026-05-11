"""
Convert spikes from a torch tensor to a NIRGraphData object.
"""

import torch
from nir.data_ir import NIRGraphData, NIRNodeData, TimeGriddedData


def to_nir_data(
    torch_dict: dict,
    dt: float,
    time_unit: float = 1.0,
    shape: tuple = ("T", "B", "N"),
) -> NIRGraphData:
    """
    Args:
        torch_dict: A dictionary mapping node names to spike tensors. If the
            shape is not (n_time_steps, batch_size, n_neurons), the `shape`
            argument can be used to specify the correct ordering of dimensions.
        dt: The time step size of the spike data.
        time_unit: The unit of time for the spike data and dt. Defaults to 1.0,
            which corresponds to seconds. For milliseconds, set this to 1e-3.
        shape: The shape of the input spike tensors. Defaults to
            ('T', 'B', 'N') for (time, batch, neurons). If the spike tensors
            have a different shape, this can be used to specify the correct
            ordering of dimensions.
    Returns:
        A NIRGraphData object containing the spike data.
    """
    nir_nodes = {}

    for key, spikes in torch_dict.items():
        # reorder dimensions to (batch, time, neurons)
        spikes = torch.permute(
            spikes, (shape.index("B"), shape.index("T"), shape.index("N"))
        )

        spikes = spikes.detach().cpu().numpy().astype(bool)

        nir_node_data = NIRNodeData({"spikes": TimeGriddedData(spikes, dt * time_unit)})
        nir_nodes[key] = nir_node_data

    nir_data = NIRGraphData(nir_nodes)
    return nir_data

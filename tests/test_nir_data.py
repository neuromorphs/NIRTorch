import nir
from nirtorch import from_nir_data, to_nir_data
import numpy as np
import torch



def test_event_data_from_nir():
    nir_data = nir.NIRGraphData(
        nodes={
            "lif": nir.NIRNodeData(
                observables={
                    "spikes": nir.EventData(
                        idx=np.random.randint(0, 10, (3, 5)),
                        time=np.random.rand(3, 5) * 0.1,
                        n_neurons=10,
                        t_max=0.1,
                    )
                }
            )
        }
    )

    torch_dict = from_nir_data(nir_data, dt=0.001)

    assert "lif" in torch_dict, "Node 'lif' not found in the converted dictionary"
    assert torch_dict["lif"].shape == (100, 3, 10), (
        f"Expected shape (100, 3, 10) for node 'lif', but got {torch_dict['lif'].shape}"
    )


def test_stable_conversion():
    original_spikes = {"lif": torch.randint(0, 2, (4, 10, 10), dtype=torch.float32)}

    nir_data = to_nir_data(original_spikes, dt=0.001)
    converted_spikes = from_nir_data(nir_data, dt=0.001)

    assert torch.equal(original_spikes["lif"], converted_spikes["lif"]), (
        "Mismatch in spikes for node 'lif'"
    )


def test_stable_conversion_with_time_unit():
    original_spikes = {"lif": torch.randint(0, 2, (4, 10, 10), dtype=torch.float32)}

    nir_data = to_nir_data(original_spikes, dt=1e-3, time_unit=1e-3)
    converted_spikes = from_nir_data(nir_data, dt=1e-3, time_unit=1e-3)

    assert torch.equal(original_spikes["lif"], converted_spikes["lif"]), (
        "Mismatch in spikes for node 'lif' with time unit conversion"
    )

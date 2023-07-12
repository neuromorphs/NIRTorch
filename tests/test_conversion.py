import norse
import sinabs
import sinabs.layers as sl
import torch
import torch.nn as nn
from norse.torch import IAFCell, SequentialState


def test_norse_to_sinabs():
    norse_model = SequentialState(
        IAFCell(p=norse.torch.functional.iaf.IAFParameters(v_th=torch.tensor(0.5))),
        torch.nn.Linear(10, 2),
        IAFCell(p=norse.torch.functional.iaf.IAFParameters(v_th=torch.tensor(0.5))),
        torch.nn.Linear(2, 1),
    )
    batch_size = 2
    nir_graph = norse.torch.utils.export.to_nir(
        norse_model, torch.randn(batch_size, 10)
    )

    sinabs_model = sinabs.from_nir(nir_graph, batch_size=batch_size)
    assert type(sinabs_model[0]) == sl.IAFSqueeze
    assert type(sinabs_model[1]) == nn.Linear
    assert type(sinabs_model[2]) == sl.IAFSqueeze
    assert type(sinabs_model[3]) == nn.Linear
    assert sinabs_model[0].spike_threshold == 0.5
    assert sinabs_model[2].spike_threshold == 0.5
    torch.testing.assert_close(norse_model[1].weight, sinabs_model[1].weight)
    torch.testing.assert_close(norse_model[3].weight, sinabs_model[3].weight)
    torch.testing.assert_close(norse_model[1].bias, sinabs_model[1].bias)
    torch.testing.assert_close(norse_model[3].bias, sinabs_model[3].bias)

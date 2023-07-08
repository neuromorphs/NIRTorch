import torch
# import norse

import nir
from nirtorch.from_nir import load, ExtractedModel

def _torch_model_map(m: nir.NIRNode, device: str = "cpu") -> torch.nn.Module:
  if isinstance(m, nir.Linear):
     l = torch.nn.Linear(*m.weights.shape[-2:])
     l.weight.data = torch.tensor(m.weights, device=device)
     l.bias.data = torch.tensor(m.bias, device=device)
     return l
  # if isinstance(m, norse.torch.LIFBoxCell):
    #  return nir.LIF(m.p.tau_mem_inv, m.p.v_leak, torch.ones_like(m.p.tau_mem_int), m.p.v_th)
  elif isinstance(m, nir.Input) or isinstance(m, nir.Output):
     return None
  else:
     raise NotImplementedError(f"Unsupported module {m}")


def test_extract_empty():
    g = nir.NIR([], [])
    m = load(g, _torch_model_map)
    assert isinstance(m, ExtractedModel)
    assert len(list(m.children())) == 0

def test_extract_lin():
    lin = nir.Linear(torch.randn(1, 2), torch.randn(1))
    g = nir.NIR([lin, lin], [(0, 1)])
    m = load(g, _torch_model_map)
    print(m.root_node)
    assert isinstance(m.root_node.elem, torch.nn.Linear)
    assert torch.allclose(m.root_node.elem.weight, lin.weights)
    assert torch.allclose(m.root_node.elem.bias, lin.bias)
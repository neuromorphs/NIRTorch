# NIRTorch

PyTorch helpers for the [Neuromorphic Intermediate Representation (NIR)](https://github.com/neuromorphs/nir).
This is a no frills python package to enable torch based libraries to translate to NIR.

## Installation
```shell
pip install nirtorch
```

## Usage

NIRTorch is typically only interfaced by library/hardwarae developers.
NIRTorch provides the `extract_nir_graph` function that takes as input a `torch.nn.Module` and a means to map Torch modules into NIR nodes.
An NIR node is an element in the NIR compute graph, corresponding to neuromorphic ODEs.

Here is an example from the [Norse](https://github.com/norse/norse) library:

```python
def _extract_norse_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    if isinstance(module, LIFBoxCell):
        return nir.LIF(
            tau=module.p.tau_mem_inv,
            v_th=module.p.v_th,
            v_leak=module.p.v_leak,
            r=torch.ones_like(module.p.v_leak),
        )
    elif isinstance(module, torch.nn.Linear):
        return nir.Linear(module.weight, module.bias)
    elif ...

    return None

def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "norse"
) -> nir.NIRNode:
    return extract_nir_graph(
        module, _extract_norse_module, sample_data, model_name=model_name
    )
```

## Acknowledgements
If you use NIR torch in your work, please cite the [following Zenodo reference](https://zenodo.org/record/8105042)

```
@software{nir2023,
  author       = {Abreu, Steven and
                  Bauer, Felix and
                  Eshraghian, Jason and
                  Jobst, Matthias and
                  Lenz, Gregor and
                  Pedersen, Jens Egholm and
                  Sheik, Sadique},
  title        = {Neuromorphic Intermediate Representation},
  month        = jul,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {0.0.1},
  doi          = {10.5281/zenodo.8105042},
  url          = {https://doi.org/10.5281/zenodo.8105042}
}
```

## For developers
If you want to make sure that your code is linted correctly on your local machine, use [pre-commit](https://pre-commit.com/) to automatically perform checks before every git commit. To use it, first install the package in your environment
```
pip install pre-commit
```
and then install the pre-commit hooks that are listed in the root of this repository
```
pre-commit install
```
Next time you commit some changes, all the checks will be run!

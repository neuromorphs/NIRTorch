<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://github.com/neuromorphs/NIR/raw/main/docs/logo_dark.png">
<img alt="NIR Logo" src="https://github.com/neuromorphs/NIR/raw/main/docs/logo_light.png">
</picture>

# NIRTorch - Torch for the Neuromorphic Intermediate Representation

[![Nature Communications Paper](https://zenodo.org/badge/DOI/10.1038/s41467-024-52259-9.svg)](https://doi.org/10.1038/s41467-024-52259-9)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/nirtorch?logo=pypi)](https://pypi.org/project/nirtorch/)
[![GitHub Tag](https://img.shields.io/github/v/tag/neuromorphs/nirtorch?logo=github)](https://github.com/neuromorphs/NIRTorch/releases)
[![Discord](https://img.shields.io/discord/1209533869733453844?logo=discord)](https://discord.gg/JRMRGP9h3c)


PyTorch helpers for the [Neuromorphic Intermediate Representation (NIR)](https://github.com/neuromorphs/nir).
This is a no frills python package to enable torch based libraries to translate to NIR.

## Installation
```shell
pip install nirtorch
```

## Usage

> [!TIP]
> Read the full documentation at [neuroir.org/docs](https://neuroir.org/docs/dev_pytorch.html).


NIRTorch is typically only interfaced by library/hardwarae developers.
NIRTorch provides two functions `nir_to_torch` and `torch_to_nir` that maps NIR graphs to Torch and vice versa.

### Translating from NIR to Torch
NIRTorch automatically creates the Torch graph and module behind the scenes.
The only thing you have to do as a developer is to provide a mapping between NIR nodes to PyTorch modules.
That is, a dictionary of nodes with associated functions that produces Torch modules, given a NIR Node (`Dict[nir.NIRNode, Callable[[nir.NIRNode], torch.nn.Module]]`), where a NIR node is an element in the NIR compute graph, corresponding to neuromorphic ODEs.
Here is one example taken from the [Norse](https://github.com/norse/norse) library:

```python
# Define NIR -> Norse mapping
my_node_dictionary = {}
def _map_linear(linear: nir.Linear) -> torch.nn.Module:
    output_shape, input_shape = linear.weight.shape[-2:]
    l = torch.nn.Linear(input_shape, output_shape, bias=False)
    l.weight.data = linear.weight
    return l
_my_node_dictionary[nir.Linear] = _map_linear
... # And so on for other modules

# Declare a user-facing function to output Norse modules
def from_nir(module: nir.NIRGraph) -> torch.nn.Module:
    return nirtorch.nir_to_torch(
        module,              # The NIR Graph to map
        _my_node_dictionary  # The dictionary used to map modules to Norse
    )
```

### Translating from Torch to NIR
NIRTorch helps construct a NIR graph by tracing through the PyTorch graph module by module.
Our job as a developer is now to provide a mapping that helps translate PyTorch modules into NIR nodes, where a NIR node is an element in the NIR compute graph.
That is, a `Dict[torch.nn.Module, Callable[[torch.nn.Module], nir.NIRNode]]`.
Note that the output node may be a subgraph.

def torch_to_nir(
    module: torch.nn.Module,
    module_map: Dict[torch.nn.Module, Callable[[torch.nn.Module], nir.NIRNode]],
    default_dict: Optional[
        Dict[torch.nn.Module, Callable[[torch.nn.Module], nir.NIRNode]]
    ] = None,
) -> nir.NIRGraph:

```python
# Define Norse -> NIR mapping
_my_module_dict = {}
def _extract_lif_module(module: norse.LIFBoxCell) -> Optional[nir.NIRNode]:
    return nir.LIF(
        tau=module.p.tau_mem_inv,
        v_th=module.p.v_th,
        v_leak=module.p.v_leak,
        r=torch.ones_like(module.p.v_leak),
    )
_my_module_dict[norse.LIFBoxcell] =_extract_lif_module

# Declare a user-facing function to output NIR graphs
def to_nir(module: torch.nn.Module) -> nir.NIRNode:
    return nirtorch.torch_to_nir(
        module,          # The Norse/Torch module to parse
        _my_module_dict  # The dictionary that maps Norse module to NIR
    )
```
Read the full documentation at [neuroir.org/docs](https://neuroir.org/docs/dev_pytorch.html).


## Acknowledgements
If you use NIRTorch in your work, please cite the [our work in Nature Communications](https://www.nature.com/articles/s41467-024-52259-9)

```
article{NIR2024, 
    title={Neuromorphic intermediate representation: A unified instruction set for interoperable brain-inspired computing}, 
    author={Pedersen, Jens E. and Abreu, Steven and Jobst, Matthias and Lenz, Gregor and Fra, Vittorio and Bauer, Felix Christian and Muir, Dylan Richard and Zhou, Peng and Vogginger, Bernhard and Heckel, Kade and Urgese, Gianvito and Shankar, Sadasivan and Stewart, Terrence C. and Sheik, Sadique and Eshraghian, Jason K.}, 
    rights={2024 The Author(s)},
    DOI={10.1038/s41467-024-52259-9}, 
    number={1},
    journal={Nature Communications}, 
    volume={15},
    year={2024}, 
    month=sep, 
    pages={8122},
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

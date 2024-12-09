from typing import Any, Callable, Dict, Set, Tuple, TypeAlias

import numpy as np
from torch.nn.modules import Module

import nir
import torch
from torch.fx import GraphModule, Tracer, Transformer
from torch.fx.passes import shape_prop

NIRTORCH_MAPPING: TypeAlias = Dict[
    torch.nn.Module, Callable[[torch.nn.Module], nir.NIRNode]
]

DEFAULT_DICT: NIRTORCH_MAPPING = {
    torch.nn.Linear: (
        lambda module: nir.Affine(
            module.weight.detach().numpy(), module.bias.detach().numpy()
        )
    )
}


class NIRTorchTracer(Tracer):

    def __init__(self, custom_leaf_modules: Tuple[torch.nn.Module] = None, **kwargs):
        """Extends PyTorch's default symbolic tracing with a set of custom leaf nodes"""
        super().__init__(**kwargs)
        if custom_leaf_modules is not None and not isinstance(
            custom_leaf_modules, tuple
        ):
            custom_leaf_modules = tuple(custom_leaf_modules)
        self.custom_leaf_modules = custom_leaf_modules

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        # Tests that the module is in the list of custom leaves
        if self.custom_leaf_modules and isinstance(m, self.custom_leaf_modules):
            return True

        if hasattr(m, "_is_leaf_module") and m._is_leaf_module:
            return True

        # return m.__module__.startswith("torch.nn") and not isinstance(
            # m, torch.nn.Sequential
        # )
        return super().is_leaf_module(m, module_qualified_name)

class NIRTorchTransformer(Transformer):
    def call_function(self, target: str, args: Tuple, kwargs: Dict) -> Any:
        print("sup", target)
        return super().call_function(target, args, kwargs)

    def call_method(self, target: str, args: Tuple, kwargs: Dict) -> Any:
        return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        print("mod", target)
        return super().call_module(target, args, kwargs)


def trace_pytorch_graph(
    module: torch.nn.Module, module_map: NIRTORCH_MAPPING, use_default_dict: bool = True
) -> nir.NIRGraph:
    # Merge the default dictionary, if requested
    if use_default_dict:
        module_map = module_map | DEFAULT_DICT
    # Cover the edge case that the
    if module.__class__ in module_map:
        return module_map[module.__class__](module)

    # Trace the graph
    tracer = NIRTorchTracer(module_map.keys())
    traced = tracer.trace(module)

    graph_module = GraphModule(tracer.root, traced)

    # transformer = NIRTorchTransformer(graph_module)
    # transformed = transformer.transform()
    # print(transformed)
    shapes = shape_prop.ShapeProp(graph_module)

    # Create NIR nodes
    nodes = {}
    edges = []
    for node in traced.nodes:
        # Add Node
        if node.op == "placeholder":
            if node.target == "input":
                module = nir.Input(np.array([1]))
            else:
                continue
        elif node.op == "output":
            module = nir.Output(np.array([1]))
        else:
            module = graph_module.get_submodule(node.target)
            module = module_map[module.__class__](module)
        nodes[node.name] = module

        # Add edges
        for in_edge in node.all_input_nodes:
            edges.append((in_edge, node.name))
    graph = nir.NIRGraph(nodes=nodes, edges=edges)
    graph.infer_types()
    return graph

    # Create NIR edges
    


if __name__ == "__main__":
    module = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Linear(1, 1))
    graph = trace_pytorch_graph(module, DEFAULT_DICT)

    import pprint

    pprint.pprint(graph)

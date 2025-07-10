from typing import Any, Callable, Dict, Set, Tuple
import operator

import numpy as np

import nir
import torch


def _map_linear(module: torch.nn.Module) -> nir.NIRNode:
    if module.bias is None:
        return nir.Linear(module.weight.detach().numpy())
    else:
        return nir.Affine(module.weight.detach().numpy(), module.bias.detach().numpy())


DEFAULT_MAP: Dict[torch.nn.Module, Callable[[torch.nn.Module], nir.NIRNode]] = {
    torch.nn.Linear: _map_linear
    # TODO: Add more default nodes
    # https://github.com/neuromorphs/NIRTorch/issues/25
}


class NIRTorchTracer(torch.fx.Tracer):

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

        return super().is_leaf_module(m, module_qualified_name)


class NIRTorchTransformer(torch.fx.Transformer):
    def call_function(self, target: str, args: Tuple, kwargs: Dict) -> Any:
        print("sup", target)
        return super().call_function(target, args, kwargs)

    def call_method(self, target: str, args: Tuple, kwargs: Dict) -> Any:
        return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        print("mod", target)
        return super().call_module(target, args, kwargs)


def torch_to_nir(
    module: torch.nn.Module,
    module_map: Dict[torch.nn.Module, Callable[[torch.nn.Module], nir.NIRNode]],
    default_dict: Dict[
        torch.nn.Module, Callable[[torch.nn.Module], nir.NIRNode]
    ] = DEFAULT_MAP,
    type_check: bool = True,
) -> nir.NIRGraph:
    """
    Traces a PyTorch module and converts it to a NIR graph using the specified module map.

    >>> import nir, nirtorch, numpy as np, torch
    >>> # First, we describe the PyTorch module we want to convert
    >>> torch_module = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=0, padding=1)
    >>> # Second, we define the dictionary
    >>> torch_to_nir_map = {
    >>>     torch.nn.AvgPool2d: lambda module: nir.AvgPool2d(
    >>>         kernel_size=np.array(module.kernel_size),
    >>>         stride=np.array(module.stride),
    >>>         padding=np.array(module.padding)
    >>>     )
    >>> }
    >>> # Finally, we call nirtorch with the node and dictionary
    >>> torch_module = nirtorch.torch_to_nir(torch_module, torch_to_nir_map)
    >>> output, state = torch_module(input)        # Note the module returns a tuple of (output, state)
    >>> output, state = torch_module(input, state) # This can go on for many (time)steps

    Args:
        module (torch.nn.Module): The module of interest
        module_map (Dict[torch.nn.Module, Callable[[torch.nn.Module], nir.NIRNode]]): A dictionary that maps
            a given module type to a function that can convert the model to an NIRNode type
        default_dict (Dict[torch.nn.Module, Callable[[torch.nn.Module], nir.NIRNode]]): An dictionary
            of default mappings that, by default, maps trivial modules like torch.nn.Linear. Override
            the dictionary to provide custom mappings.
        type_check (bool): Whether to run type checking on generated NIRGraphs
    """
    # Merge the default dictionary, if it exists
    if default_dict is not None:
        module_map = module_map | default_dict

    # Cover the edge case that the incoming module is a leaf node
    if module.__class__ in module_map:
        return module_map[module.__class__](module)

    # Trace the graph
    tracer = NIRTorchTracer(module_map.keys())
    traced = tracer.trace(module)

    if len(traced.nodes) == 2 and len(list(tracer.root.children())) == 0:
        raise ValueError(
            "The module is a leaf node, but does not appear in the module map. We cannot trace it further"
        )

    graph_module = torch.fx.GraphModule(tracer.root, traced)

    # Create NIR nodes
    nodes = {}
    edges = []
    ignored_nodes = set()
    bypass_nodes = set()

    def _find_users(node: torch.fx.Node) -> Set[torch.fx.Node]:
        """
        Finds all the users (outputs) of a given node, recursively if the node is registered as a bypass node
        """
        nodes = set()
        for user in node.users:
            if user in ignored_nodes:
                continue
            elif user in bypass_nodes:
                nodes |= _find_users(user)
            else:
                nodes.add(user)
        return nodes

    def _find_inputs(node: torch.fx.Node) -> Set[torch.fx.Node]:
        """
        Finds all the inputs (inputs) of a given node, recursively if the node is registered as a circumvented node
        """
        nodes = set()
        for in_node in node.all_input_nodes:
            if in_node in ignored_nodes:
                continue
            elif in_node in bypass_nodes:
                nodes |= _find_inputs(in_node)
            else:
                nodes.add(in_node)
        return nodes

    for node in traced.nodes:
        # Add Node
        if node.op == "placeholder":
            if node.target == "input" or node.prev.op == "root":
                nodes[str(node.name)] = nir.Input(input_type=None)
            else:
                ignored_nodes.add(node)
                continue
        elif node.op == "output":
            nodes[str(node.name)] = nir.Output(output_type=None)
        elif node.op == "call_function":
            # Ensure that we bypass add nodes
            # TODO: Consider using transformations for this
            #       https://pytorch.org/docs/stable/fx.html#torch.fx.Transformer
            if node.target == operator.add:
                bypass_nodes.add(node)
            # Raise a warning if we encounter other methods than addition
            else:
                raise ValueError(
                    "The only supported function is addition. Please modify your model or raise an issue on GitHub"
                )
        elif node.op == "call_method":
            # Bypass add methods
            if node.target == "add":
                bypass_nodes.add(node)
            else:
                raise ValueError(
                    "The only supported method is addition. Please modify your model or raise an issue on GitHub"
                )
        elif node.op == "call_module":
            torch_module = graph_module.get_submodule(node.target)
            nir_module = module_map[torch_module.__class__](torch_module)
            nodes[str(node.name)] = nir_module
        elif node.op == "get_attr":
            # Bypass attribute
            bypass_nodes.add(node)
        else:
            raise ValueError(
                f"Unsupported operation {node.op}. Please modify your model or raise an issue on GitHub"
            )

    # Create edges
    # - This is done in a separate loop to ensure that we correctly ignore the edges in case the nodes
    #   are ignored out-of-order
    for node in traced.nodes:
        if node in ignored_nodes:
            continue

        # Add edges
        for in_node in node.all_input_nodes:
            if in_node in ignored_nodes or in_node in bypass_nodes:
                continue
            # If the function is set to be bypassed, we simply forward the input to all the outputs
            if node in bypass_nodes:
                for next_node in _find_users(node):
                    edges.append((in_node.name, next_node.name))
            # Ignore additions as incoming edges
            elif in_node.op == "call_function" and in_node.target == operator.add:
                break
            # Otherwise, add an edge
            elif in_node not in ignored_nodes:
                edges.append((in_node.name, node.name))

    # Update input_type for all Input nodes based on the follower nodes' input_type.
    # The Input.input_type has been set to a placeholder value (np.array([1])) in the code above.
    for node_name, node in nodes.items():
        if isinstance(node, nir.Input):
            follower_edges = [edge for edge in edges if edge[0] == node_name]
            follower_nodes = [nodes[edge[1]] for edge in follower_edges]

            # Check that all follower nodes have the same input_type
            first_input_type = None
            for follower_node in follower_nodes:
                if first_input_type is None:
                    first_input_type = follower_node.input_type
                else:
                    # Verify they match (code taken from to NIRGraph._check_types)
                    if len(first_input_type) != len(follower_node.input_type):
                        raise ValueError(
                            f"Input type length mismatch for followers of {node_name}"
                        )

                    if len(first_input_type.keys()) == 1:
                        first_type = list(first_input_type.values())[0]
                        follower_type = list(follower_node.input_type.values())[0]
                        if not np.array_equal(first_type, follower_type):
                            raise ValueError(
                                f"Input type mismatch for followers of {node_name}: {first_type} vs {follower_type}"
                            )
                    else:
                        raise NotImplementedError(
                            "Multiple input/output types not supported yet"
                        )

            # Update the input node's input_type if we found a valid type
            if first_input_type is not None:
                node.input_type = first_input_type
                node.output_type = first_input_type

    graph = nir.NIRGraph(nodes=nodes, edges=edges, type_check=type_check)
    return graph


if __name__ == "__main__":
    module = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Linear(1, 1))
    graph = torch_to_nir(module)

    import pprint

    pprint.pprint(graph)

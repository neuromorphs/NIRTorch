import typing

import nir
import torch
import inspect

NodeMapType = typing.Dict[nir.NIRNode, typing.Callable[[nir.NIRNode], torch.nn.Module]]


def _default_map_affine(linear: nir.Affine) -> torch.nn.Linear:
    module = torch.nn.Linear(linear.weight.shape[-1], linear.weight.shape[-2], bias=True)
    module.weight.data = torch.from_numpy(linear.weight)
    module.weight.bias = torch.from_numpy(linear.bias)
    return module


def _default_map_conv1d(conv: nir.Conv1d) -> torch.nn.Conv1d:
    module = torch.nn.Conv1d(
        conv.weight.shape[-3],
        conv.weight.shape[-2],
        conv.weight.shape[-1],
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    module.weight.data = torch.from_numpy(conv.weight)
    module.bias.data = torch.from_numpy(conv.bias)
    return module


def _default_map_conv2d(conv: nir.Conv2d) -> torch.nn.Conv2d:
    c = torch.nn.Conv2d(
        conv.weight.shape[-4],
        conv.weight.shape[-3],
        conv.weight.shape[-2:],
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    c.weight.data = torch.from_numpy(conv.weight)
    c.bias.data = torch.from_numpy(conv.bias)
    return c


def _default_map_linear(linear: nir.Linear) -> torch.nn.Linear:
    module = torch.nn.Linear(linear.weight.shape[-1], linear.weight.shape[-2], bias=False)
    module.weight.data = torch.from_numpy(linear.weight)
    return module


DEFAULT_MAP: NodeMapType = {
    nir.Input: lambda input: torch.nn.Identity(),
    nir.Output: lambda output: torch.nn.Identity(),
    # Node mappings
    nir.Affine: _default_map_affine,
    nir.Conv1d: _default_map_conv1d,
    nir.Conv2d: _default_map_conv2d,
    nir.Linear: _default_map_linear,
}


def _map_nir_node_to_torch(
    node: nir.NIRNode, node_map: NodeMapType
) -> typing.Optional[torch.nn.Module]:
    if type(node) in node_map:
        return node_map[type(node)](node)
    else:
        raise ValueError(
            f"Unknown node type {type(node)}, mapping does not exist in node_map"
        )


def _construct_module_dict_recursive(
    nir_graph: nir.NIRGraph, node_map: NodeMapType
) -> torch.nn.ModuleDict:
    owning_module = torch.nn.ModuleDict()
    for name, node in nir_graph.nodes.items():
        # Recurse into subgraphs
        if isinstance(node, nir.NIRGraph):
            owning_module[name] = _construct_module_dict_recursive(node, node_map)
        else:
            mapped_module = _map_nir_node_to_torch(node, node_map=node_map)
            if mapped_module is not None:
                owning_module[name] = mapped_module
    return owning_module


def _find_result_names(
    name: str, nir_graph: nir.NIRGraph, node_outputs: typing.Tuple[typing.Any]
):
    input_nodes = list(filter(lambda t: t[1] == name, nir_graph.edges))
    return tuple(node_outputs[a] for a, _ in input_nodes)


def _is_stateful(module: torch.nn.Module) -> bool:
    signature = inspect.signature(module.forward)
    return (
        "state" in signature.parameters  # Contains a state input argument
        or isinstance(module, torch.nn.RNNBase)  # RNN modules are stateful
        or isinstance(
            module, torch.fx.GraphModule
        )  # GraphModules created by us are always stateful
    )


def _construct_state_recursive(
    parent_module: torch.nn.ModuleDict,
) -> typing.Dict[str, typing.Any]:
    state = {}
    for name, module in parent_module.items():
        if isinstance(module, torch.nn.ModuleDict):
            state[name] = _construct_state_recursive(module)
        else:
            state[name] = None
    return state


def _construct_fx_graph(
    owning_module: torch.nn.ModuleDict, nir_graph: nir.NIRGraph
) -> torch.fx.GraphModule:
    node_outputs = {}
    torch_graph = torch.fx.Graph(owning_module)

    for module_name, module in nir_graph.nodes.items():
        if isinstance(module, nir.Input):
            if len(module.output_type) > 1:
                raise ValueError("Multiple inputs are currently not supported")
            for input_name, shape in module.input_type.items():
                node_outputs[module_name] = torch_graph.create_node(
                    "placeholder", input_name
                )

            # Create a dummy state variable for each module
            default_state = _construct_state_recursive(owning_module)
            # - This has to happen *after* we create the input placeholders to avoid adding a parameter with default values before the input argument
            state_argument = torch_graph.placeholder(
                "state",
                type_expr=typing.Dict[str, typing.Any],
                default_value=default_state,
            )
        elif isinstance(module, nir.Output):
            if len(module.output_type) > 1:
                raise ValueError("Multiple outputs are currently not supported")
            for input_name, output in module.input_type.items():
                # Return both the module output and the state dict node
                module_output = _find_result_names(module_name, nir_graph, node_outputs)
                output = torch_graph.output((*module_output, state_argument))
        else:
            # Recursively wire subgraphs
            if isinstance(module, nir.NIRGraph):
                owning_module[module_name] = _construct_fx_graph(
                    owning_module[module_name], module
                )

            # Call the module
            kwargs = {}
            # - If the module has a state parameter, provide a state argument and store the state
            is_stateful = _is_stateful(owning_module[module_name])
            if is_stateful:
                kwargs["state"] = torch_graph.call_method(
                    "__getitem__", (state_argument, module_name)
                )
            output = torch_graph.call_module(
                module_name,
                _find_result_names(module_name, nir_graph, node_outputs),
                kwargs,
            )
            # - If the module is stateful, we assume the second part is a state object and must separate that from the output
            if is_stateful:
                # Add the raw output to the graph
                node_outputs[f"{module_name}_raw"] = output
                # Add the module state to the graph for use as the input to the next module
                node_outputs[f"{module_name}"] = torch_graph.call_method(
                    "__getitem__", (output, 0)
                )
                # Add the state to the state dictionary
                state_node = node_outputs[f"{module_name}_state"] = (
                    torch_graph.call_method("__getitem__", (output, 1))
                )
                torch_graph.call_method(
                    "__setitem__", (state_argument, module_name, state_node)
                )
            # - If the output is not a tuple, return as normal
            else:
                node_outputs[module_name] = output

    return torch.fx.GraphModule(owning_module, torch_graph)


def to_torch(
    nir_graph: nir.NIRGraph,
    node_map: NodeMapType,
    default_map: NodeMapType = DEFAULT_MAP,
) -> torch.fx.GraphModule:
    """
    Maps a NIRGraph as an executable PyTorch GraphModule (torch.fx.GraphModule).
    We first map all individual nodes using the node_map, where a common set of mappings are provided by default (e. g. Linear, Conv, etc.)
    Then, we wire all the nodes together into an executable torch.fx.GraphModule.
    Finally, we wrap the execution in a StatefulInterpreter, to ensure that the internal state of modules are handled correctly.
    """
    map_with_defaults = dict(default_map)
    map_with_defaults.update(node_map)  # Overwrite defaults with node_map
    # First convert all nodes into a module dictionary
    owning_module = _construct_module_dict_recursive(nir_graph, map_with_defaults)
    # Then wire the graph recursively
    return _construct_fx_graph(owning_module=owning_module, nir_graph=nir_graph)

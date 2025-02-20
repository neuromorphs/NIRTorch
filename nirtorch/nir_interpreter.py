import collections
import typing

import nir
import torch
import inspect

NodeMapType = typing.Dict[nir.NIRNode, typing.Callable[[nir.NIRNode], torch.nn.Module]]


def _default_map_affine(linear: nir.Affine) -> torch.nn.Linear:
    module = torch.nn.Linear(
        linear.weight.shape[-1], linear.weight.shape[-2], bias=True
    )
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
    module = torch.nn.Linear(
        linear.weight.shape[-1], linear.weight.shape[-2], bias=False
    )
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


def _find_input_nodes(
    name: str,
    edges: typing.List[typing.Tuple[str, str]],
    node_outputs: typing.Dict[str, torch.fx.Node],
):
    """
    Looks through the edges and find nodes that are connected to the given node as inputs.
    If one of the inputs doesn't exist because it hasn't been defined, we return None.

    # TODO: Multiple inputs should be added together
    """
    input_node_names = list(filter(lambda t: t[1] == name, edges))
    output_nodes = tuple(node_outputs.get(a, None) for a, _ in input_node_names)
    return output_nodes


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
    recursion_counter = collections.Counter(nir_graph.nodes.keys())
    # The maximum iterations per node (see https://github.com/neuromorphs/NIRTorch/pull/28#discussion_r1959343951)
    max_iterations = min(3, len(recursion_counter))
    torch_graph = torch.fx.Graph(owning_module)
    # Create a queue of the modules where we can re-insert modules for re-processing
    # in case of self-reference or if the graph is given out of order
    module_queue = collections.deque(nir_graph.nodes.items())

    # Create some dummy input for use when processing recursive elements
    dummy_input = torch_graph.call_function(torch.ones, (1,))

    # Loop through all the nodes in the queue
    while module_queue:
        module_name, module = module_queue.popleft()
        if recursion_counter[module_name] > max_iterations:
            raise RecursionError(
                f"Module {module_name} has been traversed multiple times"
                " which may be a bug in the graph or in the implementation."
                " Please file an issue at github.com/neuromorphs/nirtorch"
            )

        if isinstance(module, nir.Input):
            if len(module.output_type) > 1:
                raise NotImplementedError("Multiple inputs are currently not supported")
            for input_name, _ in module.input_type.items():
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
            # Only add the output if the module is the final node in the queue
            # - Output nodes are essentially return statements
            if len(module_queue) > 0:
                module_queue.append((module_name, module))
            else:
                if len(module.output_type) > 1:
                    raise ValueError("Multiple outputs are currently not supported")
                for input_name, output in module.input_type.items():
                    # First fetch the required input nodes
                    module_input_nodes = _find_input_nodes(
                        module_name, edges=nir_graph.edges, node_outputs=node_outputs
                    )
                    # If the module uses input that is not yet defined, set the inputs to some dummy value
                    # and enqueue the module again for processing (where it's hopefully defined)
                    if None in module_input_nodes:
                        module_input_nodes = [dummy_input for _ in module_input_nodes]
                        module_queue.append((module_name, module))

                    # # If we visited the node before, simply update the new inputs
                    # if recursion_counter[module_name] > 1:
                    #     output_node = node_outputs[module_name]
                    #     for index, input_node in enumerate(module_input_nodes):
                    #         output_node.update_arg(index, input_node)
                    # If it has not been visited before, add the node
                    node_outputs[module_name] = torch_graph.output(
                        (*module_input_nodes, state_argument)
                    )
        else:
            # 1. Recursively wire subgraphs
            if isinstance(module, nir.NIRGraph):
                owning_module[module_name] = _construct_fx_graph(
                    owning_module[module_name], module
                )

            # 2. Call the module
            kwargs = {}
            # - If the module has a state parameter, provide a state argument and store the state
            is_stateful = _is_stateful(owning_module[module_name])
            if is_stateful:
                kwargs["state"] = torch_graph.call_method(
                    "__getitem__", (state_argument, module_name)
                )
            # - If the module uses input that is not yet defined, set the input to some dummy values
            #   and enqueue the module again for processing (where it's hopefully defined)
            module_input_nodes = _find_input_nodes(
                module_name, nir_graph.edges, node_outputs=node_outputs
            )
            if None in module_input_nodes:
                # This module depends on another module that hasn't yet been constructed
                module_input_nodes = tuple([dummy_input for _ in module_input_nodes])
                # Enqueue for later processing
                module_queue.append((module_name, module))
            # Otherwise, call the module as usual
            else:
                output = torch_graph.call_module(
                    module_name,
                    module_input_nodes,
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

        recursion_counter[module_name] += 1

    # Ensure correctness
    torch_graph.lint()

    return torch.fx.GraphModule(owning_module, torch_graph)


def nir_to_torch(
    nir_node: nir.NIRNode,
    node_map: NodeMapType,
    default_map: NodeMapType = DEFAULT_MAP,
) -> torch.fx.GraphModule:
    """
    Maps a NIRGraph as an executable PyTorch GraphModule (torch.fx.GraphModule).
    We first map all individual nodes using the node_map, where a common set of mappings are provided by default (e. g. Linear, Conv, etc.)
    Then, we wire all the nodes together into an executable torch.fx.GraphModule.
    Finally, we wrap the execution in a StatefulInterpreter, to ensure that the internal state of modules are handled correctly.

    Example:
    >>> # First, we describe the NIR graph
    >>> nir_avgpool = nir.AvgPool2d(kernel_size=np.array([2, 2]), stride=np.array([1]), padding=np.array([0, 0]))
    >>> nir_linear = nir.Linear(weight=np.ones((5, 5), dtype=np.float32))
    >>> nir_graph = nir.NIRGraph.from_list(nir_avgpool, nir_linear) # Constructs a graph with a single node: AvgPool2d
    >>> # Second, we define the mapping
    >>> nir_to_torch_map = {
    >>>     nir.AvgPool2d: lambda node: torch.nn.AvgPool2d(
    >>>         kernel_size=tuple(torch.from_numpy(node.kernel_size)),
    >>>         stride=torch.from_numpy(node.stride),
    >>>         padding=tuple(torch.from_numpy(node.padding))
    >>>     )
    >>> }
    >>> # Finally, we call nirtorch with the node and dictionary
    >>> converted_module = nirtorch.nir_to_torch(nir_graph, nir_to_torch_map)
    >>> output, state = torch_module(torch.ones(1)) # Note the state return a tuple of (value, state)
    >>> output, state = torch_module(input, state)  # This can go on for many (time)steps

    Args:
        nir_node (nir.NIRNode): The input NIR node to convert to torch, typically a nir.NIRGraph
        node_map (Dict[nir.NIRNode, Callable[[nir.NIRNode], torch.nn.Module]]): A dictionary that
            maps NIR nodes into Torch modules.
        default_map (Dict[nir.NIRNode, Callable[[nir.NIRNode], torch.nn.Module]]): A dictionary with
            default values to use in case where `node_map` entries are missing. The default value of
            this parameter defines mappings for simple modules like nir.Linear and nir.Input. Override
            this to provide custom defaults.
    """
    map_with_defaults = dict(default_map)
    map_with_defaults.update(node_map)  # Overwrite defaults with node_map
    # First convert all nodes into a module dictionary
    owning_module = _construct_module_dict_recursive(nir_node, map_with_defaults)
    # Then wire the graph recursively
    return _construct_fx_graph(owning_module=owning_module, nir_graph=nir_node)

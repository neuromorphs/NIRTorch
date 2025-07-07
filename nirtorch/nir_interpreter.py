import collections
import keyword
import typing
import pathlib

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


def _default_map_flatten(flatten: nir.Flatten) -> torch.nn.Flatten:
    return torch.nn.Flatten(start_dim=flatten.start_dim, end_dim=flatten.end_dim)


def _default_map_linear(linear: nir.Linear) -> torch.nn.Linear:
    module = torch.nn.Linear(
        linear.weight.shape[-1], linear.weight.shape[-2], bias=False
    )
    module.weight.data = torch.from_numpy(linear.weight)
    return module


def _default_map_avgpool2d(pool: nir.AvgPool2d):
    return torch.nn.AvgPool2d(
        kernel_size=pool.kernel_size, stride=pool.stride, padding=pool.padding
    )


def _default_map_sumpool2d(pool: nir.SumPool2d):
    # TODO: Add support for padding
    return torch.nn.LPPool2d(
        norm_type=1, kernel_size=pool.kernel_size, stride=pool.stride
    )


DEFAULT_MAP: NodeMapType = {
    nir.Input: lambda input: torch.nn.Identity(),
    nir.Output: lambda output: torch.nn.Identity(),
    # Node mappings
    nir.Affine: _default_map_affine,
    nir.AvgPool2d: _default_map_avgpool2d,
    nir.Conv1d: _default_map_conv1d,
    nir.Conv2d: _default_map_conv2d,
    nir.Flatten: _default_map_flatten,
    nir.Linear: _default_map_linear,
    nir.SumPool2d: _default_map_sumpool2d,
}


def _sanitize_name(name: str) -> str:
    """Sanitize module name to ensure torch.fx doesn't write any keywords in code
    such as "if", "else", "for", etc.
    We also replace any periods (.) with underscores (_) in the names, since PyTorch does not allow that.
    """

    name = name.replace(".", "_")
    if keyword.iskeyword(name):
        return "nir_node_" + name
    else:
        return name


def _map_nir_node_to_torch(
    node: nir.NIRNode, node_map: NodeMapType
) -> typing.Optional[torch.nn.Module]:
    """Maps a single NIR node to PyTorch using the given node map.
    If the type is not present in the node map, raise an error.

    Returns:
        A module if the node_map can convert the NIR node to a module or an error.
    """
    if type(node) in node_map:
        return node_map[type(node)](node)
    else:
        raise ValueError(
            f"Unknown node type {type(node)}, mapping does not exist in node_map"
        )


def _construct_module_dict_recursive(
    nir_graph: nir.NIRGraph, node_map: NodeMapType
) -> torch.nn.ModuleDict:
    """Takes a NIRGraph and a node map (that can map NIR nodes to PyTorch nodes) and
    provides a torch.nn.ModuleDict with mapped nodes. The keys correspond to the
    names in the NIRGraph, with the one exception that keywords (if, for, else, ...)
    are sanitized in the _sanitize_name function.

    Returns:
        A dictionary of modules where the keys are the names of the nodes in the NIRGraph
        and the values are modules that have been converted given the node_map.
    """
    owning_module = torch.nn.ModuleDict()
    for name, node in nir_graph.nodes.items():
        # Recurse into subgraphs
        if isinstance(node, nir.NIRGraph):
            owning_module[_sanitize_name(name)] = _construct_module_dict_recursive(
                node, node_map
            )
        else:
            mapped_module = _map_nir_node_to_torch(node, node_map=node_map)
            if mapped_module is not None:
                owning_module[_sanitize_name(name)] = mapped_module
    return owning_module


def _can_reach(
    start: str, end: str, edges: typing.List[typing.Tuple[str, str]]
) -> bool:
    """Helper function to determine if there's a path from start to end in the graph."""
    visited = set()
    queue = [start]

    while queue:
        current = queue.pop(0)
        if current == end:
            return True

        if current in visited:
            continue

        visited.add(current)
        neighbors = [e for s, e in edges if s == current]
        queue.extend(neighbors)

    return False


def _find_recursive_inputs(node: str, edges: typing.Set[typing.Tuple[str, str]]):
    """Locates one or more inputs to the node from *future* nodes.
    That is, find other nodes that uses the output of this node as their input and whose
    output this node requires.

    Example:
        # A is self-recursive
        A ----+----> B
        \\--<--/

        # A needs the output of B as input
        A --> B
        \\--<--/

    Returns:
        A set of nodes whose output this node requires as input.
    """
    recursive_inputs = set()

    # Find all nodes that our target node outputs to
    outputs = set(end for start, end in edges if start == node)

    # Find all nodes that provide input to our target node
    inputs = set(start for start, end in edges if end == node)

    # Iterate over all the nodes that provide input to this module
    for other_node in inputs:
        # Check if there's a path from the other node back to our node
        # (meaning our node requires its output as input)
        if other_node in outputs or _can_reach(node, other_node, edges):
            recursive_inputs.add(other_node)

    return recursive_inputs


def _find_input_nodes(
    name: str,
    edges: typing.List[typing.Tuple[str, str]],
    node_outputs: typing.Dict[str, torch.fx.Node],
    torch_graph: typing.Optional[torch.fx.Graph] = None,
    state_node: typing.Optional[torch.fx.Node] = None,
) -> typing.Tuple[typing.Set[str], bool]:
    """
    Looks through the edges and find nodes that are connected to the given node as inputs.
    If one of the inputs doesn't exist because it hasn't been defined, we return None.

    If torch_graph is provided, multiple tensor inputs will be summed together into a single node,
    while preserving any state inputs. If it is not provided (such as for the nir.Input node),
    they are not summed together. Which may be problematic because NIR nodes typically only accept
    single inputs.

    See https://github.com/neuromorphs/NIRTorch/issues/30

    Returns:
        The set of torch.fx nodes that yields input to the given node
    """

    input_names = set(filter(lambda t: t[1] == name, edges))
    input_names_recursive = _find_recursive_inputs(name, edges)

    output_nodes = []
    for src, _ in input_names:
        # The node whose output we need as input
        input_node = node_outputs.get(src, None)
        needs_recursive_input = src in input_names_recursive and len(input_names) > 1
        # If the node is None and recursive,
        if input_node is None and needs_recursive_input:
            dummy_state = torch_graph.call_function(torch.zeros, (1,))
            prev_output_node = torch_graph.call_method(
                "get", (state_node, f"{src}_prev_output", dummy_state)
            )
            output_nodes.append(prev_output_node)
        # If the input is not recursive, add it as usual
        else:
            output_nodes.append(input_node)

    # If we have torch_graph and multiple valid inputs, sum them
    if (
        torch_graph is not None
        and len(output_nodes) > 1
        and not any(node is None for node in output_nodes)
    ):
        # Create a sum node in the graph
        sum_node = torch_graph.call_function(
            torch.add, (output_nodes[0], output_nodes[1])
        )
        # Add each remaining node to the sum
        for node in output_nodes[2:]:
            sum_node = torch_graph.call_function(torch.add, (sum_node, node))
        return (sum_node,)  # Return as a single-element tuple for consistency

    return tuple(output_nodes)


def _is_stateful(module: torch.nn.Module) -> bool:
    """Determines whether a PyTorch module requires a state input value.
    This is the case for canonical PyTorch RNN modules, where the second value
    typically represents a recurring state.

    Arguments:
        module (torch.nn.Module): The module to study.

    Returns:
        True if the method signature contains "state", if the module is a subclass
        to torch.nn.RNNBase, or if the module is a torch.fx.GraphModule.
    """
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
    """Takes a dictionary of modules and constructs a recursive dictionary of empty states
    (set to None), which will be populated once the module has any output.

    Returns:
        A dictionary of states set to None, with potentially nested dictionaries.
    """
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
    """Creates a GraphModule from a module dictionary and a NIRGraph by walking over all the edges.
    The GraphModule consists of a torch.fx Graph with nodes representing function and module calls to
    the mapped PyTorch modules in the order specified by the NIRGraph. The Graph is then mapped in the
    GraphModule, which generates Python code for future execution.

    We use a queue (collections.deque) in case we encounter recursive connections. In that case,
    we have to trace the output of previous nodes. But, since the previous node is not defined yet,
    we enqueue the current node in the hope that it'll be defined next time around.

    The function _find_input_nodes finds all the nodes that provide inputs to a given node.
    And _find_recursive_inputs help figure out which of the input nodes are using input from "future"
    nodes in the graph, that is, nodes that uses the output of the current nodes as input. If such
    a node is discovered, we prepare a state variable. Initially, that state variable is zero, but
    it's later updated if the module is called (we know to set the state variable because the module
    is flagged as recursive).

    Arguments:
        owning_module (torch.nn.ModuleDict): The module dictionary contains nodes that are already mapped from NIR to PyTorch.
        nir_graph (nir.NIRGraph): The NIRGraph to trace. Mainly used for connectivity (edges) at this point, since
            the modules have already been converted.
    """
    # Ensure that the graph has input and output nodes
    has_input = any(isinstance(node, nir.Input) for node in nir_graph.nodes.values())
    has_output = any(isinstance(node, nir.Output) for node in nir_graph.nodes.values())
    if not (has_input and has_output):
        if not has_input and not has_output:
            message = "neither input nor output was found"
        elif not has_input:
            message = "no input node was found"
        else:  # not has_output
            message = "no output node was found"
        raise ValueError(
            f"NIR graphs must contain input and output nodes, but {message}"
        )

    node_outputs = {}
    visited_nodes = set()
    pending_nodes = set()
    recursive_modules = set()
    sanitized_edges = [
        (_sanitize_name(a), _sanitize_name(b)) for a, b in nir_graph.edges
    ]
    torch_graph = torch.fx.Graph(owning_module)

    # Create a queue of the modules where we can re-insert modules for re-processing
    # in case of self-reference or if the graph is given out of order
    module_queue = collections.deque(nir_graph.nodes.items())

    # Create some dummy input for use when processing recursive elements
    dummy_input = torch_graph.call_function(torch.ones, (1,))

    # Loop through all the nodes in the queue
    while module_queue:
        module_name, module = module_queue.popleft()
        # Ensure that we start with an input node to not mess up the order
        if len(visited_nodes) == 0 and not isinstance(module, nir.Input):
            module_queue.append((module_name, module))
            continue

        # Sanitize the module name to avoid writing keywords in the generated Python code
        module_name = _sanitize_name(module_name)

        # Track nodes that are being processed to detect cycles
        if module_name in pending_nodes:
            recursive_modules.add(module_name)
        pending_nodes.add(module_name)

        if isinstance(module, nir.Input):
            if len(module.output_type) > 1:
                raise NotImplementedError(
                    "Multiple inputs to a graph are currently not supported"
                )
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
            visited_nodes.add(module_name)

        elif isinstance(module, nir.Output):
            # Only add the output if the module is the final node in the queue
            # - Output nodes are essentially return statements
            if len(module_queue) > 0 and module_name not in visited_nodes:
                module_queue.append((module_name, module))
            else:
                if len(module.output_type) > 1:
                    raise ValueError("Multiple outputs are currently not supported")
                for input_name, output in module.input_type.items():
                    # First fetch the required input nodes
                    module_input_nodes = _find_input_nodes(
                        module_name, edges=sanitized_edges, node_outputs=node_outputs
                    )
                    # If the module uses input that is not yet defined, set the inputs to some dummy value
                    # and enqueue the module again for processing (where it's hopefully defined)
                    if None in module_input_nodes and module_name not in visited_nodes:
                        module_input_nodes = [dummy_input for _ in module_input_nodes]
                        module_queue.append((module_name, module))
                    else:
                        # For output nodes, we create a new node with updated inputs
                        node_outputs[module_name] = torch_graph.output(
                            (*module_input_nodes, state_argument)
                        )

                        # If the module has not been visited before, flag it
                        if not (
                            module_name in visited_nodes and module_name in node_outputs
                        ):
                            visited_nodes.add(module_name)
        else:

            # Else, we know that this is a regular node which means we have to
            # (1) check for sub-graphs, (2) deal with recursive calls,
            # (3) handle module state, (4) gather module inputs, (5) add the node to the graph, and
            # (6) handle output and state

            # 1. Recursively wire subgraphs
            if isinstance(module, nir.NIRGraph):
                owning_module[module_name] = _construct_fx_graph(
                    owning_module[module_name], module
                )

            # 2. Determine if this is a recursive module
            has_recursive_inputs = _find_recursive_inputs(module_name, nir_graph.edges)
            is_self_recursive = any(
                src == module_name and target == module_name
                for src, target in sanitized_edges
            )
            is_recursive = has_recursive_inputs or is_self_recursive

            if is_recursive and module_name not in recursive_modules:
                recursive_modules.add(module_name)

            # 3. Handle state
            kwargs = {}
            is_stateful = _is_stateful(owning_module[module_name])
            if is_stateful:
                kwargs["state"] = torch_graph.call_method(
                    "__getitem__", (state_argument, module_name)
                )

            # 4. Gather inputs
            # - The call automatically sums additional inputs (while excluding any state argument)
            module_input_nodes = _find_input_nodes(
                module_name,
                sanitized_edges,
                node_outputs=node_outputs,
                torch_graph=torch_graph,
                state_node=state_argument,
            )

            # - Handle missing inputs
            if None in module_input_nodes:
                if is_recursive:
                    # Use available inputs where possible, dummy inputs where needed
                    module_input_nodes = tuple(
                        node if node is not None else dummy_input
                        for node in module_input_nodes
                    )
                else:
                    # This module depends on another module that hasn't yet been constructed
                    # Enqueue for later processing
                    module_input_nodes = tuple(
                        [dummy_input for _ in module_input_nodes]
                    )
                    module_queue.append((module_name, module))
                    continue

            # 5. Call the module
            # - If we've already processed this node and it's not a recursive module, move on
            if module_name in visited_nodes and module_name not in recursive_modules:
                continue

            # - Otherwise, create call node
            output = torch_graph.call_module(
                module_name,
                module_input_nodes,
                kwargs,
            )

            # 6. Handle the output
            if is_stateful:
                # Add the raw output to the graph
                node_outputs[f"{module_name}_raw"] = output

                # Separate output and state
                node_outputs[module_name] = torch_graph.call_method(
                    "__getitem__", (output, 0)
                )

                # Update state
                state_node = torch_graph.call_method("__getitem__", (output, 1))
                torch_graph.call_method(
                    "__setitem__", (state_argument, module_name, state_node)
                )

                # For recursive modules, store current output for next iteration
                if is_recursive:
                    prev_output_key = f"{module_name}_prev_output"
                    torch_graph.call_method(
                        "__setitem__",
                        (state_argument, prev_output_key, node_outputs[module_name]),
                    )
            else:
                node_outputs[module_name] = output

                # For recursive modules without explicit state, store output in state anyway
                if is_recursive:
                    prev_output_key = f"{module_name}_prev_output"
                    torch_graph.call_method(
                        "__setitem__", (state_argument, prev_output_key, output)
                    )

            visited_nodes.add(module_name)

    # Ensure correctness
    torch_graph.lint()

    return torch.fx.GraphModule(owning_module, torch_graph)


def nir_to_torch(
    nir_node: typing.Union[str, nir.NIRNode],
    node_map: NodeMapType,
    default_map: NodeMapType = DEFAULT_MAP,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.fx.GraphModule:
    """
    Maps a NIRGraph as an executable PyTorch GraphModule (torch.fx.GraphModule).
    We first map all individual nodes using the node_map, where a common set of mappings are provided by default (e. g. Linear, Conv, etc.)
    Then, we wire all the nodes together into an executable torch.fx.GraphModule.
    Finally, we wrap the execution in a StatefulInterpreter, to ensure that the internal state of modules are handled correctly.

    Example:

    >>> # Using an existing graph
    >>> nir_graph = ...
    >>> torch_module = nirtorch.nir_to_torch(nir_graph)
    >>> torch.module(torch.randn(...)) # The module is now ready to use

    >>> # Using a custom graph
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
    >>>                                             # Note that state is mutable!

    Args:
        nir_node (Union[nir.NIRNode, str, pathlib.Path]): The input NIR node to convert to torch, typically a nir.NIRGraph.
            Can also be a string or a path, in which case, we use `nir.read` to fetch the graph from the file first.
        node_map (Dict[nir.NIRNode, Callable[[nir.NIRNode], torch.nn.Module]]): A dictionary that
            maps NIR nodes into Torch modules.
        default_map (Dict[nir.NIRNode, Callable[[nir.NIRNode], torch.nn.Module]]): A dictionary with
            default values to use in case where `node_map` entries are missing. The default value of
            this parameter defines mappings for simple modules like nir.Linear and nir.Input. Override
            this to provide custom defaults.
        device (torch.device): The device to load the modules and parameters on. Defaults to "cpu"
        dtype (torch.dtype): The precision with which to load the modules and parameters. Defaults to torch.float32
    """
    if isinstance(nir_node, str) or isinstance(nir_node, pathlib.Path):
        nir_node = nir.read(nir_node)
    map_with_defaults = dict(default_map)
    map_with_defaults.update(node_map)  # Overwrite defaults with node_map

    # If the node is a leaf node (not a graph), we only mad that single node
    if not isinstance(nir_node, nir.NIRGraph):
        mapped_node = _map_nir_node_to_torch(nir_node, map_with_defaults)
        return mapped_node

    # If the node is a graph, we map it recursively
    # - First convert all nodes into a module dictionary
    owning_module = _construct_module_dict_recursive(nir_node, map_with_defaults)
    owning_module = owning_module.to(device=device, dtype=dtype)
    # - Then wire the graph recursively
    return _construct_fx_graph(owning_module=owning_module, nir_graph=nir_node)

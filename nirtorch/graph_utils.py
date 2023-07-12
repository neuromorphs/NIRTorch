def find_children(node, edges):
    """Given a node and the edges of a graph, find all direct children of that node."""
    return set(child for (parent, child) in edges if parent == node)


def find_parents(node, edges):
    """Given a node and the edges of a graph, find all direct parents of that node."""
    return set(parent for (parent, child) in edges if child == node)


def find_all_ancestors(
    node, edges, roots=None, parents_found=None, nodes_inspected=None
):
    """Given a node and the edges of a graph, find all ancesters of that node."""
    roots = roots  # or find_roots(edges)
    if node in roots:
        return set()

    parents_found = parents_found or set()
    nodes_inspected = nodes_inspected or set()
    if node in nodes_inspected:
        return set()
    # Direct parents
    parents_found.update(find_parents(node, edges))
    nodes_inspected.add(node)
    # Find grandparents
    for parent in parents_found.difference(nodes_inspected):
        # Account for nodes that have been added after start of the loop
        if parent not in nodes_inspected:
            parents_found.update(
                find_all_ancestors(parent, edges, roots, parents_found, nodes_inspected)
            )
    return parents_found


# def is_acyclic(edges):
#     for node in all_nodes(edges):
#         all_parents = find_all_ancestors(node, edges)
#         if node in all_parents:
#             print(f"Node {node} is part of a cycle")
#             return False
#     return True
#
#
# def execution_order_up_to_node(node, edges, execution_order=None):
#     """
#     Given a node and graph edges, determine the execution order such
#     that any node is only executed after its parents.
#     """
#     execution_order = execution_order or []
#     # Make sure all parents are part of execution order
#     parents = find_all_ancestors(node, edges)
#     still_missing = parents.difference(execution_order)
#     for missing_node in still_missing:
#         execution_order = generate_model_order_up_to(
#             missing_node, edges, execution_order
#         )
#     execution_order.append(node)
#
#     return execution_order
#

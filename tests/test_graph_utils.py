from collections import defaultdict

from nirtorch.graph_utils import trace_execution


class StringNode:
    def __init__(self, name, edges):
        self.name = name
        self.edges = edges

    @staticmethod
    def get_children(node):
        return [StringNode(x, node.edges) for x in node.edges[node.name]]

    @staticmethod
    def from_string(graph):
        edges = defaultdict(list)
        for edge in graph.split(" "):
            edges[edge[0]].append(edge[2])
        return StringNode(graph[0], edges)

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __eq__(self, other: object) -> bool:
        return self.name == other.name


def test_trace_linear():
    graph = "a-b b-c c-d"
    node = StringNode.from_string(graph)
    seen = trace_execution(node, node.get_children)
    assert "".join([x.name for x in seen]) == "abcd"


def test_trace_recursive():
    node = StringNode.from_string("a-b b-a")
    seen = trace_execution(node, node.get_children)
    assert "".join([x.name for x in seen]) == "ab"


def test_trace_recursive_complex():
    node = StringNode.from_string("a-b b-a b-c b-c c-d d-e")
    seen = trace_execution(node, node.get_children)
    assert "".join([x.name for x in seen]) == "abcde"

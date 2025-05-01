import ast
from typing import List, Tuple

import networkx as nx


def get_tree(edge_list: List[Tuple[int, int]]) -> nx.Graph:
    "Construct an undirected tree from a list of edges."
    return nx.from_edgelist(edge_list, create_using=nx.Graph())


def parse_edge_list(edge_list: str) -> List[Tuple[int, int]]:
    "Parse a string representation of an edge list into a list of tuples."
    res = ast.literal_eval(edge_list)
    if not isinstance(res, list):
        raise ValueError("Input must be a list of edges.")
    for edge in res:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("Each edge must be a tuple of two integers.")
        if not all(isinstance(node, int) for node in edge):
            raise ValueError("Each node in the edge must be an integer.")
    return res

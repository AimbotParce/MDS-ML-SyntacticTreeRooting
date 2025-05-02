import ast
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
import spektral.data
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data


def to_spektral(
    nx_tree: nx.Graph,
    node_features: Dict[int, NDArray[np.float32]] = None,
    node_labels: Dict[int, NDArray[np.float32]] = None,
) -> spektral.data.Graph:
    """
    Convert a NetworkX graph to a Spektral graph.
    Take the opportunity to add node features.
    """
    # Get the adjacency matrix
    A = nx.to_numpy_array(nx_tree)

    node_to_index = {node: i for i, node in enumerate(nx_tree.nodes())}

    if node_features is not None:
        # Create the node features array
        node_features_length = len(next(iter(node_features.values())))
        X = np.zeros((nx_tree.number_of_nodes(), node_features_length))
        for node in enumerate(nx_tree.nodes()):
            X[node_to_index[node]] = node_features[node]
    else:
        X = np.zeros((nx_tree.number_of_nodes(), 1))

    if node_labels is not None:
        # Create the node labels array
        node_labels_length = len(next(iter(node_labels.values())))
        y = np.zeros((nx_tree.number_of_nodes(), node_labels_length))
        for node, label in node_labels.items():
            y[node_to_index[node]] = label
    else:
        y = np.zeros((nx_tree.number_of_nodes(), 1))

    # Create the Spektral graph
    G = spektral.data.Graph(x=X, a=A, y=y, num_nodes=nx_tree.number_of_nodes(), num_edges=nx_tree.number_of_edges())
    return G


def to_torch(
    nx_tree: nx.Graph,
    node_features: Dict[int, NDArray[np.float32]] = None,
    node_labels: Dict[int, NDArray[np.float32]] = None,
) -> Data:
    """
    Convert a NetworkX graph to a PyTorch Geometric graph.
    Take the opportunity to add node features.
    """
    # Get the adjacency matrix
    A = nx.to_numpy_array(nx_tree)

    node_to_index = {node: i for i, node in enumerate(nx_tree.nodes())}

    if node_features is not None:
        # Create the node features array
        node_features_length = len(next(iter(node_features.values())))
        X = np.zeros((nx_tree.number_of_nodes(), node_features_length))
        for node in nx_tree.nodes():
            X[node_to_index[node]] = node_features[node]
    else:
        X = np.zeros((nx_tree.number_of_nodes(), 1))

    if node_labels is not None:
        # Create the node labels array
        node_labels_length = len(next(iter(node_labels.values())))
        y = np.zeros((nx_tree.number_of_nodes(), node_labels_length))
        for node, label in node_labels.items():
            y[node_to_index[node]] = label
    else:
        y = np.zeros((nx_tree.number_of_nodes(), 1))

    G = Data(
        x=torch.from_numpy(X).float(),
        edge_index=torch.from_numpy(np.array(nx.to_numpy_array(nx_tree).nonzero())).long(),
        edge_attr=None,
        y=torch.from_numpy(y).float(),
    )
    return G


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

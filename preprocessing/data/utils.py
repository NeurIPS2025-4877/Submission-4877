import argparse
import itertools
from operator import itemgetter
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


def one_of_k_encoding(x: int, allowable_set: List) -> List:
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x: int, allowable_set: List) -> List:
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def graph_to_edges(graph: nx.Graph) -> List:
    return [(src, dst) for src, dst in graph.edges()]


def normalize_data(x: np.ndarray) -> np.ndarray:
    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    return x



def get_treatment_graphs(treatment_ids: list, id_to_graph_dict: dict) -> List:
    return [id_to_graph_dict[i] for i in treatment_ids]


def get_treatment_one_hot_encodings(
    treatment_ids: list, id_to_one_hot_encoding_dict: dict
) -> List:
    return [id_to_one_hot_encoding_dict[i] for i in treatment_ids]


def get_treatment_combinations(treatment_ids: list) -> List:
    return list(itertools.combinations(treatment_ids, 2))

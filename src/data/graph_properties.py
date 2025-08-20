import math
from typing import List, Iterable
from numpy.typing import NDArray

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_undirected, remove_self_loops
from src.common import (
    NORMALIZATION_EPS, 
    NORMALIZATION_STD_FALLBACK,
    TRIANGLE_DIVISOR,
    CLUSTERING_DIVISOR,
    TASK_ZERO_LOSS
)


class GraphPropertyCalculator:
    """
    Computes a fixed 15-D vector of graph structural properties from a PyG Data object.

    Properties:
      0: num_nodes
      1: num_edges (undirected, self-loops removed)
      2: density
      3: degree_mean
      4: degree_var
      5: degree_max
      6: average_clustering
      7: transitivity
      8: triangles (total, not per-node)
      9: num_connected_components
     10: diameter (on largest connected component)
     11: degree_assortativity
     12: degree_centralization (Freeman)
     13: closeness_centralization
      14: betweenness_centralization
    """

    def __call__(self, graph: Data) -> Tensor:
        num_nodes = graph.x.shape[0]
        edge_index = graph.edge_index

        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        pyg_simple = Data(edge_index=edge_index, num_nodes=num_nodes)
        G = to_networkx(pyg_simple, to_undirected=True)

        N = G.number_of_nodes()
        E = G.number_of_edges()

        degree_dict = dict(G.degree())
        degrees = np.array(list(degree_dict.values()), dtype=float)
        deg_mean = float(degrees.mean())
        deg_var = float(degrees.var())
        deg_max = float(degrees.max())

        density = float(nx.density(G))

        clustering_global = float(nx.average_clustering(G))
        transitivity = float(nx.transitivity(G)) if N > 2 else 0.0

        if N > 2:
            tri_dict = nx.triangles(G)
            triangles = float(sum(tri_dict.values()) / TRIANGLE_DIVISOR)
        else:
            triangles = 0.0

        num_components = float(nx.number_connected_components(G))

        diameter = 0.0
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        H = max(components, key=lambda g: g.number_of_nodes())
        diameter = float(nx.diameter(H))

        if float(degrees.std()) == TASK_ZERO_LOSS:
            assortativity = 0.0
        else:
            assortativity = float(nx.degree_assortativity_coefficient(G))
            if math.isnan(assortativity) or math.isinf(assortativity):
                assortativity = 0.0

        if N > 2:
            numerator = float((degrees.max() - degrees).sum())
            denominator = float((N - 1) * (N - 2))
            degree_centralization = numerator / denominator
        else:
            degree_centralization = 0.0

        closeness_dict = nx.closeness_centrality(G)
        closeness = np.array(list(closeness_dict.values()), dtype=float)
        c_max = closeness.max()
        closeness_centralization = float((c_max - closeness).sum() / (N - 1))

        if N > 2:
            betw_dict = nx.betweenness_centrality(G, normalized=True)
            betw = np.array(list(betw_dict.values()), dtype=float)
            b_max = betw.max()
            denom = float((N - 1) * (N - 2) / CLUSTERING_DIVISOR)
            betweenness_centralization = float((b_max - betw).sum() / denom)
        else:
            betweenness_centralization = 0.0

        props = [
            float(N),
            float(E),
            density,
            deg_mean,
            deg_var,
            deg_max,
            clustering_global,
            transitivity,
            triangles,
            num_components,
            diameter,
            assortativity,
            degree_centralization,
            closeness_centralization,
            betweenness_centralization,
        ]

        return torch.tensor(props, dtype=torch.float32)

    def compute_for_dataset(self, dataset: Iterable[Data]) -> Tensor:
        """
        Compute graph property vectors for an entire dataset and return a stacked tensor.

        Args:
            dataset: Iterable of PyG Data graphs.

        Returns:
            Tensor of shape [num_graphs, 15]
        """
        dataset_list = list(dataset) if not isinstance(dataset, list) else dataset
        props_tensor = torch.zeros((len(dataset_list), 15), dtype=torch.float32)

        for i, g in enumerate(dataset_list):
            props_tensor[i] = self(g)

        return props_tensor

    def compute_and_standardize_for_dataset(
        self,
        dataset: Iterable[Data],
        train_idx: NDArray[np.int_],
    ) -> Tensor:
        """
        Compute graph properties for all graphs and standardize using statistics from
        the subset indexed by train_idx.

        Args:
            dataset: Iterable of PyG Data graphs.
            train_idx: Indices of graphs to use for computing mean and std.

        Returns:
            Tensor of shape [num_graphs, 15] standardized via z-score using train stats.
        """
        all_props = self.compute_for_dataset(dataset)

        train_props = all_props[train_idx]
        mean = train_props.mean(dim=0)
        std = train_props.std(dim=0, unbiased=True)
        std[std < NORMALIZATION_EPS] = NORMALIZATION_STD_FALLBACK

        all_props = (all_props - mean) / std
        return all_props

import math
from typing import Iterable

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_networkx, to_undirected

GRAPH_PROPERTY_DIM = 12


class GraphPropertyCalculator:
    def __call__(self, graph: Data) -> Tensor:
        num_nodes = graph.num_nodes
        edge_index = graph.edge_index.clone()

        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        pyg_simple = Data(x=graph.x, edge_index=edge_index)
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

        num_components = float(nx.number_connected_components(G))

        try:
            components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
            H = max(components, key=lambda g: g.number_of_nodes())
            diameter = float(nx.diameter(H))
        except (nx.NetworkXError, ValueError):
            diameter = 0.0

        if deg_var == 0.0:
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

        props = [
            float(N),
            float(E),
            density,
            deg_mean,
            deg_var,
            deg_max,
            clustering_global,
            transitivity,
            num_components,
            diameter,
            assortativity,
            degree_centralization,
        ]

        return torch.tensor(props, dtype=torch.float32)

    def compute_for_dataset(self, dataset: Iterable[Data]) -> Tensor:
        dataset_list = list(dataset)
        props_tensor = torch.zeros((len(dataset_list), GRAPH_PROPERTY_DIM), dtype=torch.float32)

        for i, g in enumerate(dataset_list):
            props_tensor[i] = self(g)

        return props_tensor

    def compute_and_standardize_for_dataset(self, dataset: Iterable[Data], train_idx: NDArray[np.int64]) -> Tensor:
        all_props = self.compute_for_dataset(dataset)

        scaler = StandardScaler()
        scaler.fit(all_props[train_idx].numpy())

        all_props_scaled = scaler.transform(all_props.numpy())
        return torch.from_numpy(all_props_scaled).float()

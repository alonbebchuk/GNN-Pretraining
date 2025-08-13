import math
from typing import List

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


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
        num_nodes = int(graph.x.size(0))
        edge_index = graph.edge_index

        # Build undirected simple graph in networkx (deduplicate, remove self-loops)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        ei = edge_index.detach().cpu().numpy()
        for u, v in ei.T:
            if u == v:
                continue
            G.add_edge(int(u), int(v))

        N = G.number_of_nodes()
        E = G.number_of_edges()

        # Degrees
        degrees = np.array([d for _, d in G.degree()], dtype=float)
        deg_mean = float(degrees.mean())
        deg_var = float(degrees.var())
        deg_max = float(degrees.max())

        # Density (nx handles N<=1)
        density = float(nx.density(G))

        # Clustering metrics
        clustering_global = float(nx.average_clustering(G))
        transitivity = float(nx.transitivity(G))

        tri_dict = nx.triangles(G)
        triangles = float(sum(tri_dict.values()) / 3.0)

        # Connectivity
        num_components = float(nx.number_connected_components(G))

        # Diameter on largest component
        diameter = 0.0
        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        if len(components) > 0:
            H = max(components, key=lambda g: g.number_of_nodes())
            if H.number_of_nodes() > 1:
                diameter = float(nx.diameter(H))

        # Assortativity
        if float(degrees.std()) == 0.0:
            assortativity = 0.0
        else:
            assortativity = float(nx.degree_assortativity_coefficient(G))
            if math.isnan(assortativity) or math.isinf(assortativity):
                assortativity = 0.0

        # Degree centralization (Freeman): sum(max_deg - deg_i) / ((N-1)*(N-2))
        if N > 2:
            numerator = float((degrees.max() - degrees).sum())
            denominator = float((N - 1) * (N - 2))
            degree_centralization = numerator / denominator
        else:
            degree_centralization = 0.0

        # Closeness centralization: sum(max_c - c_i) / (N-1)
        closeness = np.array(list(nx.closeness_centrality(G).values()), dtype=float)
        c_max = float(closeness.max())
        closeness_centralization = float(((c_max - closeness).sum()) / (N - 1))

        # Betweenness centralization: sum(max_b - b_i) / ((N-1)*(N-2)/2)
        if N > 2:
            betw = np.array(list(nx.betweenness_centrality(G, normalized=True).values()), dtype=float)
            b_max = float(betw.max()) if betw.size > 0 else 0.0
            denom = float((N - 1) * (N - 2) / 2.0)
            betweenness_centralization = float(((b_max - betw).sum()) / denom)
        else:
            betweenness_centralization = 0.0

        props: List[float] = [
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

from typing import List, Tuple

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph

ATTR_MASK_MIN_NUM_FEATURES = 3
ATTR_MASK_PROB = 0.2
ATTR_MASK_RATE = 0.2
EDGE_DROP_MIN_NUM_EDGES = 3
EDGE_DROP_PROB = 0.2
EDGE_DROP_RATE = 0.2
NODE_DROP_MIN_NUM_NODES = 3
NODE_DROP_RATE = 0.2


def _attribute_mask(data: Data, generator: torch.Generator) -> Data:
    num_features = data.num_node_features

    if num_features < ATTR_MASK_MIN_NUM_FEATURES:
        return data

    num_features_to_mask = max(1, int(num_features * ATTR_MASK_RATE))
    mask_indices = torch.randperm(num_features, generator=generator)[:num_features_to_mask].to(data.x.device)

    data.x[:, mask_indices] = 0.0
    return data


def _edge_drop(data: Data, generator: torch.Generator) -> Data:
    num_edges = data.num_edges

    if num_edges < EDGE_DROP_MIN_NUM_EDGES:
        return data

    num_edges_to_drop = max(1, int(num_edges * EDGE_DROP_RATE))
    num_edges_to_keep = num_edges - num_edges_to_drop

    keep_indices = torch.randperm(num_edges, generator=generator)[:num_edges_to_keep].to(data.edge_index.device)
    data.edge_index = data.edge_index[:, keep_indices]
    return data


def _node_drop(data: Data, generator: torch.Generator) -> Tuple[Data, torch.Tensor]:
    num_nodes = data.num_nodes

    if num_nodes < NODE_DROP_MIN_NUM_NODES:
        return data, torch.arange(num_nodes, device=data.x.device)

    num_nodes_to_drop = max(1, int(num_nodes * NODE_DROP_RATE))
    num_nodes_to_keep = num_nodes - num_nodes_to_drop

    kept_nodes_indices = torch.randperm(num_nodes, generator=generator)[:num_nodes_to_keep].to(data.x.device)
    kept_nodes_indices = kept_nodes_indices.sort()[0]

    edge_index, _ = subgraph(kept_nodes_indices, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
    data.x = data.x[kept_nodes_indices]
    data.edge_index = edge_index

    return data, kept_nodes_indices


def _create_augmented_view(data: Data, generator: torch.Generator) -> Tuple[Data, torch.Tensor]:
    aug_data = data.clone()

    aug_data, kept_nodes_indices = _node_drop(aug_data, generator)

    if torch.rand(1, generator=generator).item() < EDGE_DROP_PROB:
        aug_data = _edge_drop(aug_data, generator)

    if torch.rand(1, generator=generator).item() < ATTR_MASK_PROB:
        aug_data = _attribute_mask(aug_data, generator)

    return aug_data, kept_nodes_indices


def _find_common_nodes_for_contrastive_loss(kept_nodes_1: torch.Tensor, kept_nodes_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat((kept_nodes_1, kept_nodes_2))
    uniques, counts = combined.unique(return_counts=True)
    common_original_nodes = uniques[counts == 2]

    mask_1 = torch.isin(kept_nodes_1, common_original_nodes)
    mask_2 = torch.isin(kept_nodes_2, common_original_nodes)

    return mask_1, mask_2


class GraphAugmentor:
    @staticmethod
    def create_two_views(batch: Batch, generator: torch.Generator) -> Tuple[Batch, Batch, List[torch.Tensor], List[torch.Tensor]]:
        graphs = batch.to_data_list()

        view_1_graphs = []
        view_2_graphs = []
        mask_1_list = []
        mask_2_list = []

        for graph in graphs:
            view_1, kept_nodes_1 = _create_augmented_view(graph, generator)
            view_2, kept_nodes_2 = _create_augmented_view(graph, generator)
            mask_1, mask_2 = _find_common_nodes_for_contrastive_loss(kept_nodes_1, kept_nodes_2)

            view_1_graphs.append(view_1)
            view_2_graphs.append(view_2)
            mask_1_list.append(mask_1)
            mask_2_list.append(mask_2)

        batch_view_1 = Batch.from_data_list(view_1_graphs)
        batch_view_2 = Batch.from_data_list(view_2_graphs)

        return batch_view_1, batch_view_2, mask_1_list, mask_2_list

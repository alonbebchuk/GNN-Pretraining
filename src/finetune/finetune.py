import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import wandb
from torch.optim import AdamW
from torch_geometric.data import Batch
from torch_geometric.utils import negative_sampling

from src.data.finetune_data_loaders import create_finetune_data_loader
from src.finetune.metrics import compute_batch_metrics, compute_training_metrics, compute_validation_metrics, compute_test_metrics
from src.models.finetune_model import FinetuneGNN, create_finetune_model

import torch.nn.functional as F

from src.data.data_setup import TASK_TYPES, NUM_CLASSES

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "finetune"
PROJECT_NAME = "gnn-pretraining-finetune"

BATCH_SIZES = {
    'ENZYMES': 32,
    'PTC_MR': 32,
    'Cora_NC': -1,
    'CiteSeer_NC': -1,
    'Cora_LP': 256,
    'CiteSeer_LP': 256,
}
EPOCHS = {
    'ENZYMES': 100,
    'PTC_MR': 100,
    'Cora_NC': 200,
    'CiteSeer_NC': 200,
    'Cora_LP': 300,
    'CiteSeer_LP': 300,
}
HARD_NEGATIVE_RATIO = 0.3
MIN_HARD_NEGATIVES = 8
PATIENCE_FRACTION = 0.5


class LinkPredictionHardNegativeMiner:
    def mine_hard_negatives_for_edges(self, node_embeddings: torch.Tensor, positive_edges: torch.Tensor, num_negatives: int, existing_edges: torch.Tensor) -> torch.Tensor:
        device = node_embeddings.device
        num_nodes = node_embeddings.size(0)

        embeddings_norm = F.normalize(node_embeddings, dim=1)
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

        edge_mask = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.bool)

        if existing_edges.size(1) > 0:
            edge_mask[existing_edges[0], existing_edges[1]] = True
            edge_mask[existing_edges[1], existing_edges[0]] = True

        edge_mask.fill_diagonal_(True)

        potential_negatives_mask = ~edge_mask

        potential_scores = similarity_matrix[potential_negatives_mask]
        potential_indices = torch.where(potential_negatives_mask)

        if len(potential_scores) == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        num_hard = max(MIN_HARD_NEGATIVES, int(len(potential_scores) * HARD_NEGATIVE_RATIO))
        num_hard = min(num_hard, len(potential_scores), num_negatives)

        if num_hard > 0:
            _, hard_indices = torch.topk(potential_scores, num_hard, largest=True)
            hard_src = potential_indices[0][hard_indices]
            hard_dst = potential_indices[1][hard_indices]
            hard_negatives = torch.stack([hard_src, hard_dst], dim=0)
        else:
            hard_negatives = torch.empty(2, 0, dtype=torch.long, device=device)

        remaining = num_negatives - num_hard
        if remaining > 0:
            remaining_mask = potential_negatives_mask.clone()
            if num_hard > 0:
                remaining_mask[hard_src, hard_dst] = False
                remaining_mask[hard_dst, hard_src] = False

            remaining_indices = torch.where(remaining_mask)
            num_available = len(remaining_indices[0])

            if num_available > 0:
                num_to_sample = min(remaining, num_available)
                rand_idx = torch.randperm(num_available, device=device)[:num_to_sample]
                rand_src = remaining_indices[0][rand_idx]
                rand_dst = remaining_indices[1][rand_idx]
                rand_negatives = torch.stack([rand_src, rand_dst], dim=0)

                if num_hard > 0:
                    negative_edges = torch.cat([hard_negatives, rand_negatives], dim=1)
                else:
                    negative_edges = rand_negatives
            else:
                negative_edges = hard_negatives
        else:
            negative_edges = hard_negatives

        return negative_edges


@dataclass
class FinetuneConfig:
    domain_name: str
    finetune_strategy: str
    pretrained_scheme: str
    seed: int

    exp_name: str = None
    task_type: str = None
    batch_size: int = None
    epochs: int = None
    patience: int = None

    def __post_init__(self):
        self.exp_name = f"{self.domain_name}_{self.finetune_strategy}_{self.pretrained_scheme}"
        self.task_type = TASK_TYPES[self.domain_name]
        self.batch_size = BATCH_SIZES[self.domain_name]
        self.epochs = EPOCHS[self.domain_name]
        self.patience = int(self.epochs * PATIENCE_FRACTION)


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_batch(
    model: torch.nn.Module,
    batch: Batch,
    device: torch.device,
    task_type: str,
    domain_name: str,
    hard_negative_miner: LinkPredictionHardNegativeMiner,
    train_edges_for_hard_mining: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if task_type == 'graph_classification':
        batch = batch.to(device)
        logits = model(batch)
        targets = batch.y

        if NUM_CLASSES[domain_name] == 2:
            binary_logits = logits[:, 1]
            targets_float = targets.float()
            loss = F.binary_cross_entropy_with_logits(binary_logits, targets_float)
        else:
            loss = F.cross_entropy(logits, targets)

        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        return loss, targets, predictions, probabilities

    elif task_type == 'node_classification':
        data, node_indices, targets = batch
        data, node_indices, targets = data.to(device), node_indices.to(device), targets.to(device)

        full_logits = model(data, message_passing_edges=train_edges_for_hard_mining)
        logits = full_logits[node_indices]

        if NUM_CLASSES[domain_name] == 2:
            binary_logits = logits[:, 1]
            targets_float = targets.float()
            loss = F.binary_cross_entropy_with_logits(binary_logits, targets_float)
        else:
            loss = F.cross_entropy(logits, targets)

        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        return loss, targets, predictions, probabilities

    elif task_type == 'link_prediction':
        if model.training:
            data, pos_edges, _ = batch
            data, pos_edges = data.to(device), pos_edges.to(device)

            with torch.no_grad():
                node_embeddings = model.gnn_backbone(model.input_encoder(data.x), train_edges_for_hard_mining)

            neg_edges = hard_negative_miner.mine_hard_negatives_for_edges(node_embeddings=node_embeddings, positive_edges=pos_edges, num_negatives=pos_edges.size(1), existing_edges=train_edges_for_hard_mining).to(device)
                
            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            edge_labels = torch.cat([
                torch.ones(pos_edges.size(1), device=device),
                torch.zeros(neg_edges.size(1), device=device)
            ])
        else:
            data, all_edges, edge_labels = batch
            data, all_edges, edge_labels = data.to(device), all_edges.to(device), edge_labels.to(device)

        edge_probs = model(data, edge_index=all_edges, message_passing_edges=train_edges_for_hard_mining)

        loss = F.binary_cross_entropy(edge_probs, edge_labels)
        targets = edge_labels.long()
        predictions = (edge_probs > 0.5).long()
        probabilities = torch.stack([1 - edge_probs, edge_probs], dim=1)

        return loss, targets, predictions, probabilities


def compute_loss_and_metrics(
    model: torch.nn.Module,
    batch, device: torch.device,
    task_type: str,
    domain_name: str,
    prefix: str,
    hard_negative_miner: LinkPredictionHardNegativeMiner,
    train_edges_for_mining: torch.Tensor
) -> Dict[str, float]:
    model.eval()

    with torch.no_grad():
        loss, targets, predictions, probabilities = process_batch(
            model,
            batch,
            device,
            task_type,
            domain_name,
            hard_negative_miner,
            train_edges_for_mining
        )
        metrics = compute_batch_metrics(
            domain_name,
            targets,
            predictions,
            probabilities,
            loss,
            prefix
        )

    return metrics


@torch.no_grad()
def run_evaluation(
    model: FinetuneGNN,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    best_val_metric: float,
    epochs_since_improvement: int,
    cfg: FinetuneConfig,
    global_step: List[int],
    hard_negative_miner: LinkPredictionHardNegativeMiner,
    train_edges_for_mining: torch.Tensor,
    model_path: str,
    model_name: str
) -> Tuple[float, int]:
    model.eval()

    batch_metrics = []
    for batch_or_data in val_loader:
        metrics = compute_loss_and_metrics(
            model, batch_or_data, device, cfg.task_type, cfg.domain_name, 'val', hard_negative_miner, train_edges_for_mining
        )
        batch_metrics.append(metrics)

    val_metrics = compute_validation_metrics(batch_metrics, epoch)

    val_metric = val_metrics['val/auc' if cfg.task_type == 'link_prediction' else 'val/accuracy']
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        epochs_since_improvement = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_metrics': val_metrics,
        }
        torch.save(checkpoint, model_path)

        artifact = wandb.Artifact(name=model_name, type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
    else:
        epochs_since_improvement += 1

    wandb.log(val_metrics, step=global_step[0])

    return best_val_metric, epochs_since_improvement


def run_training(
    model: FinetuneGNN,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    global_step_ref: List[int],
    cfg: FinetuneConfig,
    hard_negative_miner: LinkPredictionHardNegativeMiner
) -> None:
    model.train()

    for batch in train_loader:
        step_start_time = time.time()
        global_step_ref[0] += 1

        train_edges_for_mining = train_loader.dataset.train_edges if hasattr(train_loader.dataset, 'train_edges') else None
        if train_edges_for_mining is not None:
            train_edges_for_mining = train_edges_for_mining.to(device)
        
        loss, targets, predictions, probabilities = process_batch(
            model,
            batch,
            device,
            cfg.task_type,
            cfg.domain_name,
            hard_negative_miner,
            train_edges_for_mining
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_metrics = compute_training_metrics(
            epoch=epoch,
            step=global_step_ref[0],
            loss=loss,
            optimizer=optimizer,
            domain_name=cfg.domain_name,
            targets=targets,
            predictions=predictions,
            probabilities=probabilities,
            step_start_time=step_start_time,
            model=model
        )

        wandb.log(train_metrics, step=global_step_ref[0])


def finetune(cfg: FinetuneConfig) -> None:
    training_start_time = time.time()

    set_global_seed(cfg.seed)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=PROJECT_NAME, name=f"{cfg.exp_name}_{cfg.seed}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_name = f"model_{cfg.exp_name}_{cfg.seed}"
    model_path = OUTPUT_DIR / f"{model_name}.pt"

    val_loader = create_finetune_data_loader(cfg.domain_name, 'val', cfg.batch_size, generator)
    test_loader = create_finetune_data_loader(cfg.domain_name, 'test', cfg.batch_size, generator)
    train_loader = create_finetune_data_loader(cfg.domain_name, 'train', cfg.batch_size, generator)

    model = create_finetune_model(device=device, cfg=cfg)
    optimizer = AdamW(model.param_groups)

    hard_negative_miner = None
    if cfg.task_type == 'link_prediction':
        hard_negative_miner = LinkPredictionHardNegativeMiner()

    initial_checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'val_metrics': {},
    }
    torch.save(initial_checkpoint, model_path)

    best_val_metric = -float('inf')
    epochs_since_improvement = 0

    global_step = [0]

    for epoch in range(1, cfg.epochs + 1):
        run_training(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            global_step,
            cfg,
            hard_negative_miner
        )

        train_edges_for_mining = train_loader.dataset.train_edges if hasattr(train_loader.dataset, 'train_edges') else None
        if train_edges_for_mining is not None:
            train_edges_for_mining = train_edges_for_mining.to(device)
        
        best_val_metric, epochs_since_improvement = run_evaluation(
            model,
            val_loader,
            device,
            epoch,
            best_val_metric,
            epochs_since_improvement,
            cfg,
            global_step,
            hard_negative_miner,
            train_edges_for_mining,
            model_path,
            model_name
        )

        if epochs_since_improvement >= cfg.patience:
            break

    best_model = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_model['model_state_dict'])

    train_edges_for_test_mining = train_loader.dataset.train_edges if hasattr(train_loader.dataset, 'train_edges') else None
    if train_edges_for_test_mining is not None:
        train_edges_for_test_mining = train_edges_for_test_mining.to(device)

    batch_metrics = []
    for batch_or_data in test_loader:
        metrics = compute_loss_and_metrics(model, batch_or_data, device, cfg.task_type, cfg.domain_name, 'test', hard_negative_miner, train_edges_for_test_mining)
        batch_metrics.append(metrics)

    test_metrics = compute_test_metrics(
        batch_metrics=batch_metrics,
        epoch=epoch,
        epochs_since_improvement=epochs_since_improvement,
        training_start_time=training_start_time,
        model=model
    )

    wandb.log(test_metrics, step=global_step[0])
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", type=str, required=True)
    parser.add_argument("--finetune_strategy", type=str, required=True)
    parser.add_argument("--pretrained_scheme", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = FinetuneConfig(
        domain_name=args.domain_name,
        finetune_strategy=args.finetune_strategy, 
        pretrained_scheme=args.pretrained_scheme,
        seed=args.seed
    )
    finetune(cfg)


if __name__ == "__main__":
    main()

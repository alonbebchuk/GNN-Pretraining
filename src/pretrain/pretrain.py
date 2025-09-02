import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import wandb
from torch.optim import AdamW

from src.data.data_setup import PRETRAIN_TUDATASETS
from src.data.pretrain_data_loaders import create_train_data_loader, create_val_data_loader
from src.models.pretrain_model import PretrainableGNN
from src.pretrain.grl_scheduler import GRLLambdaScheduler
from src.pretrain.tasks import (
    BasePretrainTask,
    DomainAdversarialTask,
    GraphContrastiveTask,
    GraphPropertyPredictionTask,
    LinkPredictionTask,
    NodeContrastiveTask,
    NodeFeatureMaskingTask,
)
from src.pretrain.adaptive_loss_balancer import AdaptiveLossBalancer

BATCH_SIZE = 32
EPOCHS = 50
LR_MODEL = 1e-5
MAX_GRAD_NORM = 0.5
MODEL_WEIGHT_DECAY = 1e-5
PATIENCE_FRACTION = 0.5

PRETRAIN_DOMAINS = {
    'b2': PRETRAIN_TUDATASETS,
    'b3': PRETRAIN_TUDATASETS, 
    'b4': ['ENZYMES'],
    's1': PRETRAIN_TUDATASETS,
    's2': PRETRAIN_TUDATASETS,
    's3': PRETRAIN_TUDATASETS,
    's4': PRETRAIN_TUDATASETS,
    's5': PRETRAIN_TUDATASETS,
}

ACTIVE_TASKS = {
    'b2': ['node_feat_mask'],
    'b3': ['node_contrast'],
    'b4': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop'],
    's1': ['node_feat_mask', 'link_pred'],
    's2': ['node_contrast', 'graph_contrast'],
    's3': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast'],
    's4': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop'],
    's5': ['node_feat_mask', 'link_pred', 'node_contrast', 'graph_contrast', 'graph_prop', 'domain_adv'],
}

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "pretrain"
PROJECT_NAME = "gnn-pretraining-pretrain"


@dataclass
class PretrainConfig:
    exp_name: str
    seed: int

    pretrain_domains: List[str] = None
    active_tasks: List[str] = None

    def __post_init__(self):
        self.pretrain_domains = PRETRAIN_DOMAINS[self.exp_name]
        self.active_tasks = ACTIVE_TASKS[self.exp_name]


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def instantiate_tasks(model: PretrainableGNN, active_tasks: List[str]) -> Dict[str, BasePretrainTask]:
    tasks = {}
    for name in active_tasks:
        if name == "node_feat_mask":
            tasks[name] = NodeFeatureMaskingTask(model)
        elif name == "link_pred":
            tasks[name] = LinkPredictionTask(model)
        elif name == "node_contrast":
            tasks[name] = NodeContrastiveTask(model)
        elif name == "graph_contrast":
            tasks[name] = GraphContrastiveTask(model)
        elif name == "graph_prop":
            tasks[name] = GraphPropertyPredictionTask(model)
        elif name == "domain_adv":
            tasks[name] = DomainAdversarialTask(model)
    return tasks


def run_training(
    model: PretrainableGNN,
    tasks: Dict[str, BasePretrainTask],
    optimizer: AdamW,
    train_loader: torch.utils.data.DataLoader,
    generator: torch.Generator,
    grl_sched: GRLLambdaScheduler,
    device: torch.device,
    epoch: int,
    global_step_ref: List[int],
    cfg: PretrainConfig,
    loss_balancer: AdaptiveLossBalancer,
) -> None:
    model.train()

    for domain_batches in train_loader:
        global_step_ref[0] += 1

        for domain_name in domain_batches:
            domain_batches[domain_name] = domain_batches[domain_name].to(device)

        per_task_losses = {}
        per_domain_per_task_losses_tensors = {domain: {} for domain in cfg.pretrain_domains}
        per_domain_per_task_losses = {domain: {} for domain in cfg.pretrain_domains}
        per_domain_losses = {}

        for task_name, task in tasks.items():
            task_loss, task_per_domain_loss = task.compute_loss(domain_batches, generator)
            per_task_losses[task_name] = task_loss
            for domain, per_domain_per_task_loss in task_per_domain_loss.items():
                per_domain_per_task_losses_tensors[domain][task_name] = per_domain_per_task_loss
                per_domain_per_task_losses[domain][task_name] = float(per_domain_per_task_loss.detach())

        for domain in per_domain_per_task_losses_tensors.keys():
            domain_task_losses = list(per_domain_per_task_losses_tensors[domain].values())
            domain_loss = torch.stack(domain_task_losses).sum()
            per_domain_losses[domain] = float(domain_loss.detach().cpu())

        lambda_val = grl_sched()

        total_loss = loss_balancer.balance_losses(per_task_losses, lambda_val)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        grl_sched.step()

        train_metrics = {}

        for domain in per_domain_per_task_losses:
            for task in per_domain_per_task_losses[domain]:
                train_metrics[f'train/loss/{domain}/{task}'] = per_domain_per_task_losses[domain][task]

        for task_name, loss_tensor in per_task_losses.items():
            train_metrics[f'train/loss/{task_name}'] = float(loss_tensor.detach().cpu())

        for domain_name, domain_loss in per_domain_losses.items():
            train_metrics[f'train/loss/{domain_name}'] = domain_loss

        train_metrics['train/loss/total'] = float(total_loss.detach().cpu())
        train_metrics['train/progress/epoch'] = epoch

        if 'domain_adv' in per_task_losses:
            train_metrics['train/domain_adv/lambda'] = lambda_val

        current_weights = loss_balancer.get_current_weights()
        for task_name, weight in current_weights.items():
            train_metrics[f'train/loss_balancer/weight/{task_name}'] = weight

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        train_metrics['train/gradients/model_grad_norm'] = total_norm

        wandb.log(train_metrics, step=global_step_ref[0])


@torch.no_grad()
def run_evaluation(
    model: PretrainableGNN,
    tasks: Dict[str, BasePretrainTask],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    generator: torch.Generator,
    grl_sched: GRLLambdaScheduler,
    device: torch.device,
    epoch: int,
    best_total_loss: float,
    epochs_since_improvement: int,
    cfg: PretrainConfig,
    global_step: List[int],
    loss_balancer: AdaptiveLossBalancer,
) -> Tuple[float, int]:
    model.eval()

    per_task_losses = {}
    per_domain_per_task_losses = {domain: {} for domain in val_loaders.keys()}

    for task_name, task in tasks.items():
        domain_losses = []

        for domain_name, val_loader in val_loaders.items():
            batch_losses = []
            for batch in val_loader:
                batch = batch.to(device)
                loss, _ = task.compute_loss({domain_name: batch}, generator)
                batch_losses.append(loss)
            
            domain_task_loss = torch.stack(batch_losses).mean()
            domain_losses.append(domain_task_loss)
            per_domain_per_task_losses[domain_name][task_name] = float(domain_task_loss.detach().cpu())
        
        per_task_losses[task_name] = torch.stack(domain_losses).mean()
    
    lambda_val = grl_sched()

    total_loss = loss_balancer.balance_losses(per_task_losses, lambda_val)

    per_domain_losses = {}
    for domain in per_domain_per_task_losses.keys():
        domain_task_losses = list(per_domain_per_task_losses[domain].values())
        domain_loss = sum(domain_task_losses) / len(domain_task_losses)
        per_domain_losses[domain] = float(domain_loss)

    val_metrics = {}

    for domain in per_domain_per_task_losses:
        for task in per_domain_per_task_losses[domain]:
            val_metrics[f'val/loss/{domain}/{task}'] = per_domain_per_task_losses[domain][task]

    for task_name, loss_tensor in per_task_losses.items():
        val_metrics[f'val/loss/{task_name}'] = float(loss_tensor.detach().cpu())

    for domain_name, domain_loss in per_domain_losses.items():
        val_metrics[f'val/loss/{domain_name}'] = domain_loss

    val_metrics['val/loss/total'] = float(total_loss.detach().cpu())
    val_metrics['val/progress/epoch'] = epoch

    if 'domain_adv' in per_task_losses:
        val_metrics['val/domain_adv/lambda'] = lambda_val

    if total_loss < best_total_loss:
        best_total_loss = total_loss
        epochs_since_improvement = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_metrics': val_metrics,
        }
        model_name = f"model_{cfg.exp_name}_{cfg.seed}"
        model_path = OUTPUT_DIR / f"{model_name}.pt"

        torch.save(checkpoint, model_path)

        artifact = wandb.Artifact(
            name=model_name,
            type="model",
        )
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
    else:
        epochs_since_improvement += 1

    wandb.log(val_metrics, step=global_step[0])

    return best_total_loss, epochs_since_improvement


def pretrain(cfg: PretrainConfig) -> None:
    set_global_seed(cfg.seed)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=PROJECT_NAME, name=f"{cfg.exp_name}_{cfg.seed}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_loaders = {}
    for domain in cfg.pretrain_domains:
        val_loaders[domain] = create_val_data_loader(domain, generator)
    train_loader = create_train_data_loader(cfg.pretrain_domains, generator)

    model = PretrainableGNN(
        device=device,
        domain_names=cfg.pretrain_domains,
        task_names=cfg.active_tasks,
    )
    tasks = instantiate_tasks(model, cfg.active_tasks)
    optimizer = AdamW(model.parameters(), lr=LR_MODEL, weight_decay=MODEL_WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    grl_sched = GRLLambdaScheduler(total_steps=total_steps)

    loss_balancer = AdaptiveLossBalancer()

    best_total_loss = float("inf")
    epochs_since_improvement = 0

    global_step = [0]

    for epoch in range(1, EPOCHS + 1):
        run_training(
            model,
            tasks,
            optimizer,
            train_loader,
            generator,
            grl_sched,
            device,
            epoch,
            global_step,
            cfg,
            loss_balancer
        )

        best_total_loss, epochs_since_improvement = run_evaluation(
            model,
            tasks,
            val_loaders,
            generator,
            grl_sched,
            device,
            epoch,
            best_total_loss,
            epochs_since_improvement,
            cfg,
            global_step,
            loss_balancer
        )

        if epochs_since_improvement >= int(EPOCHS * PATIENCE_FRACTION):
            break

    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = PretrainConfig(exp_name=args.exp_name, seed=args.seed)
    pretrain(cfg)


if __name__ == "__main__":
    main()

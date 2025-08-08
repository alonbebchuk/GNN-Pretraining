"""
Multi-task loss computation for GNN pre-training.

This module implements all loss functions needed for the pre-training tasks:
- Node feature masking (MSE)
- Link prediction (BCE)
- Node contrastive learning (NT-Xent)
- Graph contrastive learning (BCE)
- Graph property prediction (MSE)
- Domain adversarial training (CE)
- Uncertainty weighting for multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Import TaskConfig for type handling
try:
    from ..infrastructure.config import TaskConfig
except ImportError:
    try:
        from infrastructure.config import TaskConfig
    except ImportError:
        # Fallback for testing
        from dataclasses import dataclass
        
        @dataclass
        class TaskConfig:
            temperature: float = 0.1


@dataclass
class LossOutput:
    """Container for loss computation results."""
    total_loss: torch.Tensor
    individual_losses: Dict[str, torch.Tensor]
    weighted_losses: Dict[str, torch.Tensor]
    uncertainty_params: Optional[Dict[str, torch.Tensor]] = None


class NodeFeatureMaskingLoss(nn.Module):
    """Loss for node feature masking pre-training task."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target embeddings.
        
        Args:
            predictions: Predicted embeddings [num_masked_nodes, hidden_dim]
            targets: Target embeddings [num_masked_nodes, hidden_dim]
            
        Returns:
            MSE loss scalar
        """
        if predictions.shape[0] == 0:
            # No masked nodes
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        return self.mse_loss(predictions, targets)


class LinkPredictionLoss(nn.Module):
    """Loss for link prediction pre-training task."""
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, edge_logits: torch.Tensor, edge_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE loss for link prediction.
        
        Args:
            edge_logits: Raw edge prediction logits [num_edges]
            edge_labels: Binary edge labels [num_edges] (1 for positive, 0 for negative)
            
        Returns:
            BCE loss scalar
        """
        if edge_logits.shape[0] == 0:
            return torch.tensor(0.0, device=edge_logits.device, requires_grad=True)
        
        return self.bce_loss(edge_logits, edge_labels.float())


class NodeContrastiveLoss(nn.Module):
    """NT-Xent loss for node-level contrastive learning."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss between two augmented views.
        
        Args:
            z1: Node embeddings from first augmented view [num_nodes, proj_dim]
            z2: Node embeddings from second augmented view [num_nodes, proj_dim]
            
        Returns:
            NT-Xent loss scalar
        """
        if z1.shape[0] == 0:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)
        
        batch_size = z1.shape[0]
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrices
        # Positive pairs: z1[i] with z2[i]
        # Negative pairs: z1[i] with z2[j] where j != i, and z1[i] with z1[j] where j != i
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)  # [2*batch_size, proj_dim]
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(representations, representations.t()) / self.temperature
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size),
                           torch.arange(0, batch_size)], dim=0).to(z1.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = self.cross_entropy(similarity_matrix, labels)
        
        return loss


class GraphContrastiveLoss(nn.Module):
    """Loss for graph-level contrastive learning (InfoGraph style)."""
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE loss for graph contrastive learning.
        
        Args:
            scores: Discriminator scores [num_pairs]
            labels: Binary labels [num_pairs] (1 for positive, 0 for negative)
            
        Returns:
            BCE loss scalar
        """
        if scores.shape[0] == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        return self.bce_loss(scores, labels.float())


class GraphPropertyLoss(nn.Module):
    """Loss for graph property prediction."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss for graph property prediction.
        
        Args:
            predictions: Predicted properties [batch_size, num_properties]
            targets: Target properties [batch_size, num_properties]
            
        Returns:
            MSE loss scalar
        """
        if predictions.shape[0] == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        return self.mse_loss(predictions, targets)


class DomainAdversarialLoss(nn.Module):
    """Loss for domain adversarial training."""
    
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, domain_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for domain classification.
        
        Args:
            domain_logits: Domain prediction logits [batch_size, num_domains]
            domain_labels: True domain labels [batch_size]
            
        Returns:
            Cross-entropy loss scalar
        """
        if domain_logits.shape[0] == 0:
            return torch.tensor(0.0, device=domain_logits.device, requires_grad=True)
        
        return self.cross_entropy(domain_logits, domain_labels)


class UncertaintyWeighting(nn.Module):
    """
    Uncertainty weighting for multi-task learning.
    
    Implements the uncertainty weighting from "Multi-Task Learning Using Uncertainty to Weigh Losses".
    Each task has a learnable uncertainty parameter σ_i, and the weighted loss is:
    L_total = Σ [ (1 / (2 * σ_i²)) * L_i + log(σ_i) ]
    """
    
    def __init__(self, task_names: List[str], init_log_sigma: float = 0.0):
        """
        Initialize uncertainty parameters.
        
        Args:
            task_names: List of task names
            init_log_sigma: Initial value for log(σ)
        """
        super().__init__()
        
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        # Learnable log(σ²) parameters (one per task)
        # We learn log(σ²) instead of σ² for numerical stability
        self.log_sigma_sq = nn.Parameter(
            torch.full((self.num_tasks,), init_log_sigma * 2, dtype=torch.float32)
        )
        
        # Task name to index mapping
        self.task_to_idx = {name: idx for idx, name in enumerate(task_names)}
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute uncertainty-weighted total loss.
        
        Args:
            losses: Dictionary of individual task losses
            
        Returns:
            Tuple of (total_weighted_loss, individual_weighted_losses)
        """
        total_loss = torch.tensor(0.0, device=self.log_sigma_sq.device)
        weighted_losses = {}
        
        for task_name, loss_value in losses.items():
            if task_name in self.task_to_idx:
                task_idx = self.task_to_idx[task_name]
                log_sigma_sq = self.log_sigma_sq[task_idx]
                
                # Uncertainty weighting: (1 / (2 * σ²)) * L + log(σ)
                # Since we store log(σ²), we have: exp(-log_sigma_sq/2) * L + log_sigma_sq/2
                weighted_loss = 0.5 * torch.exp(-log_sigma_sq) * loss_value + 0.5 * log_sigma_sq
                
                weighted_losses[task_name] = weighted_loss
                total_loss = total_loss + weighted_loss
            else:
                # Task not in uncertainty weighting, use as-is
                weighted_losses[task_name] = loss_value
                total_loss = total_loss + loss_value
        
        return total_loss, weighted_losses
    
    def get_uncertainty_params(self) -> Dict[str, float]:
        """Get current uncertainty parameters (σ values)."""
        uncertainties = {}
        for task_name, task_idx in self.task_to_idx.items():
            log_sigma_sq = self.log_sigma_sq[task_idx].item()
            sigma = torch.exp(torch.tensor(log_sigma_sq / 2)).item()
            uncertainties[task_name] = sigma
        return uncertainties


class MultiTaskLossComputer:
    """
    Main loss computer that orchestrates all pre-training losses.
    
    This class handles:
    - Computing individual task losses
    - Applying uncertainty weighting
    - Managing domain adversarial training
    - Providing comprehensive loss statistics
    """
    
    def __init__(self, task_configs: Dict[str, Any], 
                 use_uncertainty_weighting: bool = True,
                 uncertainty_init: float = 0.0):
        """
        Initialize the multi-task loss computer.
        
        Args:
            task_configs: Dictionary of task configurations
            use_uncertainty_weighting: Whether to use uncertainty weighting
            uncertainty_init: Initial uncertainty parameter value
        """
        self.task_configs = task_configs
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Initialize individual loss functions
        self.loss_functions = {
            'node_feat_mask': NodeFeatureMaskingLoss(),
            'link_pred': LinkPredictionLoss(),
            'node_contrast': NodeContrastiveLoss(temperature=task_configs.get('node_contrast', TaskConfig()).temperature),
            'graph_contrast': GraphContrastiveLoss(),
            'graph_prop': GraphPropertyLoss(),
            'domain_adv': DomainAdversarialLoss()
        }
        
        # Initialize uncertainty weighting if enabled
        if use_uncertainty_weighting:
            enabled_tasks = [name for name, config in task_configs.items() if getattr(config, 'enabled', True)]
            self.uncertainty_weighting = UncertaintyWeighting(
                task_names=enabled_tasks,
                init_log_sigma=uncertainty_init
            )
        else:
            self.uncertainty_weighting = None
        
        logging.info(f"Initialized MultiTaskLossComputer with {len(self.loss_functions)} loss functions")
        if use_uncertainty_weighting:
            logging.info(f"Uncertainty weighting enabled for {len(enabled_tasks)} tasks")
    
    def compute_losses(self, model_outputs: Dict[str, Any], 
                      targets: Dict[str, Any],
                      task_weights: Optional[Dict[str, float]] = None,
                      lambda_da: float = 0.0) -> LossOutput:
        """
        Compute all task losses and return comprehensive results.
        
        Args:
            model_outputs: Dictionary containing model outputs for each task
            targets: Dictionary containing target values for each task
            task_weights: Optional manual task weights (overrides config weights)
            
        Returns:
            LossOutput containing total loss, individual losses, and statistics
        """
        individual_losses = {}
        
        # Compute individual task losses
        for task_name, loss_fn in self.loss_functions.items():
            if task_name not in self.task_configs or not getattr(self.task_configs[task_name], 'enabled', True):
                continue
            
            if task_name not in model_outputs or task_name not in targets:
                continue
            
            try:
                # Compute loss for this task
                if task_name == 'node_feat_mask':
                    loss = loss_fn(model_outputs[task_name], targets[task_name])
                elif task_name == 'link_pred':
                    loss = loss_fn(model_outputs[task_name], targets[task_name])
                elif task_name == 'node_contrast':
                    # Expects two augmented views
                    z1, z2 = model_outputs[task_name]
                    loss = loss_fn(z1, z2)
                elif task_name == 'graph_contrast':
                    loss = loss_fn(model_outputs[task_name], targets[task_name])
                elif task_name == 'graph_prop':
                    loss = loss_fn(model_outputs[task_name], targets[task_name])
                elif task_name == 'domain_adv':
                    loss = loss_fn(model_outputs[task_name], targets[task_name])
                else:
                    logging.warning(f"Unknown task: {task_name}")
                    continue
                
                individual_losses[task_name] = loss
                
            except Exception as e:
                logging.error(f"Error computing loss for task {task_name}: {str(e)}")
                continue
        
        # Separate domain adversarial loss for special handling (negative sign)
        domain_adv_loss = individual_losses.pop('domain_adv', None)
        
        # Apply weighting to non-domain-adversarial tasks
        if self.use_uncertainty_weighting and self.uncertainty_weighting is not None:
            # Use uncertainty weighting for main tasks
            total_loss, weighted_losses = self.uncertainty_weighting(individual_losses)
            uncertainty_params = self.uncertainty_weighting.get_uncertainty_params()
        else:
            # Use manual weights for main tasks
            weighted_losses = {}
            if individual_losses:
                total_loss = torch.tensor(0.0, device=next(iter(individual_losses.values())).device)
            else:
                total_loss = torch.tensor(0.0)
            
            for task_name, loss_value in individual_losses.items():
                # Get weight from task_weights or config
                if task_weights and task_name in task_weights:
                    weight = task_weights[task_name]
                else:
                    weight = getattr(self.task_configs.get(task_name, TaskConfig()), 'weight', 1.0)
                
                weighted_loss = weight * loss_value
                weighted_losses[task_name] = weighted_loss
                total_loss = total_loss + weighted_loss
            
            uncertainty_params = None
        
        # Handle domain adversarial loss with negative sign: - λ * L_domain
        # This implements the formula: L_total = Σ[(1/2σᵢ²)Lᵢ + log σᵢ] - λ L_domain
        if domain_adv_loss is not None:
            # Add back to individual losses for tracking
            individual_losses['domain_adv'] = domain_adv_loss
            
            # Apply negative lambda weighting
            domain_weighted_loss = -lambda_da * domain_adv_loss
            weighted_losses['domain_adv'] = domain_weighted_loss
            
            # Subtract from total loss (negative sign in the formula)
            total_loss = total_loss + domain_weighted_loss
        
        return LossOutput(
            total_loss=total_loss,
            individual_losses=individual_losses,
            weighted_losses=weighted_losses,
            uncertainty_params=uncertainty_params
        )
    
    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Get learnable parameters (uncertainty weights)."""
        if self.uncertainty_weighting is not None:
            return list(self.uncertainty_weighting.parameters())
        else:
            return []


def compute_domain_adversarial_lambda(epoch: int, total_epochs: int, 
                                    schedule_type: str = 'dann',
                                    initial_lambda: float = 0.0,
                                    final_lambda: float = 1.0,
                                    gamma: float = 10.0,
                                    warmup_epochs: int = 0) -> float:
    """
    Compute lambda value for domain adversarial training schedule.
    
    Args:
        epoch: Current epoch (0-based)
        total_epochs: Total number of epochs
        schedule_type: Type of schedule ('dann', 'linear', 'constant')
        initial_lambda: Initial lambda value
        final_lambda: Final lambda value
        gamma: Gamma parameter for DANN schedule
        warmup_epochs: Number of warmup epochs
        
    Returns:
        Lambda value for current epoch
    """
    if epoch < warmup_epochs:
        return initial_lambda
    
    if schedule_type == 'constant':
        return final_lambda
    elif schedule_type == 'linear':
        # Linear interpolation
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return initial_lambda + progress * (final_lambda - initial_lambda)
    elif schedule_type == 'dann':
        # DANN schedule: λ_p = 2 / (1 + exp(-γ * p)) - 1
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        import math
        lambda_p = 2 / (1 + math.exp(-gamma * progress)) - 1
        return initial_lambda + lambda_p * (final_lambda - initial_lambda)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


if __name__ == '__main__':
    # Test the loss computation system
    
    # Test configuration using proper config objects
    from dataclasses import dataclass
    
    @dataclass
    class TestTaskConfig:
        enabled: bool = True
        weight: float = 1.0
        temperature: float = 0.1
    
    task_configs = {
        'node_feat_mask': TestTaskConfig(),
        'link_pred': TestTaskConfig(),
        'node_contrast': TestTaskConfig(temperature=0.1),
        'graph_contrast': TestTaskConfig(),
        'graph_prop': TestTaskConfig(),
        'domain_adv': TestTaskConfig()
    }
    
    # Initialize loss computer
    loss_computer = MultiTaskLossComputer(
        task_configs=task_configs,
        use_uncertainty_weighting=True
    )
    
    # Create dummy data
    batch_size = 8
    hidden_dim = 64
    num_domains = 4
    
    model_outputs = {
        'node_feat_mask': torch.randn(10, hidden_dim),
        'link_pred': torch.randn(20),
        'node_contrast': (torch.randn(16, 128), torch.randn(16, 128)),
        'graph_contrast': torch.randn(24),
        'graph_prop': torch.randn(batch_size, 3),
        'domain_adv': torch.randn(batch_size, num_domains)
    }
    
    targets = {
        'node_feat_mask': torch.randn(10, hidden_dim),
        'link_pred': torch.randint(0, 2, (20,)).float(),
        'graph_contrast': torch.randint(0, 2, (24,)).float(),
        'graph_prop': torch.randn(batch_size, 3),
        'domain_adv': torch.randint(0, num_domains, (batch_size,))
    }
    
    # Compute losses
    loss_output = loss_computer.compute_losses(model_outputs, targets, lambda_da=0.5)
    
    print("Loss computation test:")
    print(f"Total loss: {loss_output.total_loss.item():.4f}")
    print("Individual losses:")
    for task_name, loss_value in loss_output.individual_losses.items():
        print(f"  {task_name}: {loss_value.item():.4f}")
    
    if loss_output.uncertainty_params:
        print("Uncertainty parameters:")
        for task_name, sigma in loss_output.uncertainty_params.items():
            print(f"  {task_name}: σ = {sigma:.4f}")
    
    # Test domain adversarial lambda schedule
    print("\nDomain adversarial lambda schedule:")
    for epoch in [0, 10, 50, 100]:
        lambda_val = compute_domain_adversarial_lambda(
            epoch=epoch, total_epochs=100, schedule_type='dann'
        )
        print(f"  Epoch {epoch}: λ = {lambda_val:.4f}")
    
    print("Loss computation test completed successfully!") 
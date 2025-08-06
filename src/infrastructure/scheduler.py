"""
Learning rate and lambda scheduling for GNN training.

This module provides various scheduling strategies for:
- Learning rate scheduling (cosine, linear, step)
- Domain adversarial lambda scheduling
- Warmup strategies
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, Union, List
import warnings


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.
    
    This scheduler implements:
    1. Linear warmup for the first warmup_steps
    2. Cosine annealing from peak LR to min_lr for remaining steps
    """
    
    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0,
                 min_lr_ratio: float = 0.01, last_epoch: int = -1):
        """
        Initialize cosine annealing with warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps
            min_lr_ratio: Minimum LR as fraction of base LR
            last_epoch: Last epoch index (for resuming)
        """
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_steps = self.total_steps - self.warmup_steps
            current_step = self.last_epoch - self.warmup_steps
            
            cosine_factor = 0.5 * (1 + math.cos(math.pi * current_step / max(1, cosine_steps)))
            lr_factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            
            return [base_lr * lr_factor for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup followed by constant learning rate."""
    
    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1):
        """
        Initialize linear warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch index (for resuming)
        """
        self.warmup_steps = warmup_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Constant LR after warmup
            return self.base_lrs


class PolynomialDecayScheduler(_LRScheduler):
    """Polynomial decay learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0,
                 power: float = 1.0, min_lr_ratio: float = 0.0, last_epoch: int = -1):
        """
        Initialize polynomial decay scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps
            power: Power for polynomial decay
            min_lr_ratio: Minimum LR as fraction of base LR
            last_epoch: Last epoch index (for resuming)
        """
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        
        super(PolynomialDecayScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            decay_steps = self.total_steps - self.warmup_steps
            current_step = self.last_epoch - self.warmup_steps
            
            decay_factor = (1 - current_step / max(1, decay_steps)) ** self.power
            lr_factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * decay_factor
            
            return [base_lr * lr_factor for base_lr in self.base_lrs]


class DomainAdversarialScheduler:
    """
    Scheduler for domain adversarial lambda parameter.
    
    Implements various scheduling strategies for the lambda parameter
    in domain adversarial training.
    """
    
    def __init__(self, schedule_type: str = 'dann',
                 initial_lambda: float = 0.0,
                 final_lambda: float = 1.0,
                 total_epochs: int = 100,
                 warmup_epochs: int = 0,
                 gamma: float = 10.0):
        """
        Initialize domain adversarial scheduler.
        
        Args:
            schedule_type: Type of schedule ('dann', 'linear', 'constant')
            initial_lambda: Initial lambda value
            final_lambda: Final lambda value
            total_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs
            gamma: Gamma parameter for DANN schedule
        """
        self.schedule_type = schedule_type
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        
        self.current_epoch = 0
        
        if schedule_type not in ['dann', 'linear', 'constant']:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def step(self, epoch: Optional[int] = None):
        """Update the current epoch."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
    
    def get_lambda(self) -> float:
        """Get current lambda value."""
        if self.current_epoch < self.warmup_epochs:
            return self.initial_lambda
        
        if self.schedule_type == 'constant':
            return self.final_lambda
        elif self.schedule_type == 'linear':
            # Linear interpolation
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            progress = min(1.0, progress)  # Clamp to [0, 1]
            return self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
        elif self.schedule_type == 'dann':
            # DANN schedule: λ_p = 2 / (1 + exp(-γ * p)) - 1
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            progress = min(1.0, progress)  # Clamp to [0, 1]
            
            lambda_p = 2 / (1 + math.exp(-self.gamma * progress)) - 1
            return self.initial_lambda + lambda_p * (self.final_lambda - self.initial_lambda)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def state_dict(self):
        """Return scheduler state."""
        return {
            'schedule_type': self.schedule_type,
            'initial_lambda': self.initial_lambda,
            'final_lambda': self.final_lambda,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'gamma': self.gamma,
            'current_epoch': self.current_epoch
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.schedule_type = state_dict['schedule_type']
        self.initial_lambda = state_dict['initial_lambda']
        self.final_lambda = state_dict['final_lambda']
        self.total_epochs = state_dict['total_epochs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.gamma = state_dict['gamma']
        self.current_epoch = state_dict['current_epoch']


def create_lr_scheduler(optimizer, scheduler_config, total_steps: int, 
                       total_epochs: int) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration object
        total_steps: Total number of training steps
        total_epochs: Total number of training epochs
        
    Returns:
        Learning rate scheduler or None if disabled
    """
    if scheduler_config.type == 'none':
        return None
    
    # Calculate warmup steps
    if hasattr(scheduler_config, 'warmup_epochs') and scheduler_config.warmup_epochs > 0:
        warmup_steps = scheduler_config.warmup_epochs * (total_steps // total_epochs)
    elif hasattr(scheduler_config, 'warmup_fraction') and scheduler_config.warmup_fraction > 0:
        warmup_steps = int(total_steps * scheduler_config.warmup_fraction)
    else:
        warmup_steps = 0
    
    if scheduler_config.type == 'cosine':
        return CosineAnnealingWithWarmup(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=getattr(scheduler_config, 'min_lr_ratio', 0.01)
        )
    elif scheduler_config.type == 'linear':
        return LinearWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps
        )
    elif scheduler_config.type == 'polynomial':
        return PolynomialDecayScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            power=getattr(scheduler_config, 'power', 1.0),
            min_lr_ratio=getattr(scheduler_config, 'min_lr_ratio', 0.0)
        )
    elif scheduler_config.type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=getattr(scheduler_config, 'step_size', 50),
            gamma=getattr(scheduler_config, 'gamma', 0.1)
        )
    elif scheduler_config.type == 'multistep':
        milestones = getattr(scheduler_config, 'milestones', [50, 80])
        return optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=getattr(scheduler_config, 'gamma', 0.1)
        )
    elif scheduler_config.type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=getattr(scheduler_config, 'gamma', 0.95)
        )
    else:
        warnings.warn(f"Unknown scheduler type: {scheduler_config.type}. Using no scheduler.")
        return None


def create_domain_adversarial_scheduler(da_config, total_epochs: int) -> DomainAdversarialScheduler:
    """
    Create domain adversarial scheduler based on configuration.
    
    Args:
        da_config: Domain adversarial configuration object
        total_epochs: Total number of training epochs
        
    Returns:
        Domain adversarial scheduler
    """
    return DomainAdversarialScheduler(
        schedule_type=da_config.schedule_type,
        initial_lambda=da_config.initial_lambda,
        final_lambda=da_config.final_lambda,
        total_epochs=total_epochs,
        warmup_epochs=da_config.warmup_epochs,
        gamma=da_config.gamma
    )


class SchedulerManager:
    """
    Manager class for coordinating multiple schedulers.
    
    This class manages both learning rate schedulers and domain adversarial
    lambda schedulers, providing a unified interface for stepping and
    getting current values.
    """
    
    def __init__(self, lr_scheduler: Optional[_LRScheduler] = None,
                 da_scheduler: Optional[DomainAdversarialScheduler] = None):
        """
        Initialize scheduler manager.
        
        Args:
            lr_scheduler: Learning rate scheduler
            da_scheduler: Domain adversarial scheduler
        """
        self.lr_scheduler = lr_scheduler
        self.da_scheduler = da_scheduler
    
    def step(self, epoch: Optional[int] = None, step: Optional[int] = None):
        """
        Step all schedulers.
        
        Args:
            epoch: Current epoch (for epoch-based schedulers)
            step: Current step (for step-based schedulers)
        """
        if self.lr_scheduler is not None:
            if hasattr(self.lr_scheduler, 'step'):
                if step is not None and hasattr(self.lr_scheduler, 'last_epoch'):
                    # Step-based scheduler
                    self.lr_scheduler.last_epoch = step - 1
                    self.lr_scheduler.step()
                elif epoch is not None:
                    # Epoch-based scheduler
                    self.lr_scheduler.step(epoch)
                else:
                    # Default step
                    self.lr_scheduler.step()
        
        if self.da_scheduler is not None:
            self.da_scheduler.step(epoch)
    
    def get_current_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_last_lr()
        else:
            return [0.0]  # Default if no scheduler
    
    def get_current_lambda(self) -> float:
        """Get current domain adversarial lambda."""
        if self.da_scheduler is not None:
            return self.da_scheduler.get_lambda()
        else:
            return 0.0  # Default if no scheduler
    
    def state_dict(self):
        """Get state dictionary for all schedulers."""
        state = {}
        
        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        if self.da_scheduler is not None:
            state['da_scheduler'] = self.da_scheduler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dictionary for all schedulers."""
        if 'lr_scheduler' in state_dict and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        
        if 'da_scheduler' in state_dict and self.da_scheduler is not None:
            self.da_scheduler.load_state_dict(state_dict['da_scheduler'])


if __name__ == '__main__':
    # Test the scheduling system
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create dummy optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test cosine annealing with warmup
    total_steps = 1000
    warmup_steps = 100
    
    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.1
    )
    
    # Collect learning rates
    lrs = []
    for step in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    # Test domain adversarial scheduler
    da_scheduler = DomainAdversarialScheduler(
        schedule_type='dann',
        initial_lambda=0.0,
        final_lambda=1.0,
        total_epochs=100,
        warmup_epochs=10,
        gamma=10.0
    )
    
    lambdas = []
    for epoch in range(100):
        lambdas.append(da_scheduler.get_lambda())
        da_scheduler.step()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(lrs)
    ax1.set_title('Learning Rate Schedule')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Learning Rate')
    ax1.grid(True)
    
    ax2.plot(lambdas)
    ax2.set_title('Domain Adversarial Lambda Schedule')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Lambda')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('scheduler_test.png')
    print("Scheduler test completed! Check 'scheduler_test.png' for visualization.")
    
    # Test scheduler manager
    manager = SchedulerManager(lr_scheduler=scheduler, da_scheduler=da_scheduler)
    
    print(f"Current LR: {manager.get_current_lr()}")
    print(f"Current Lambda: {manager.get_current_lambda()}")
    
    print("Scheduler system test completed successfully!") 
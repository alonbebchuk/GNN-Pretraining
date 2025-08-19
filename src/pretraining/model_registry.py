"""
Model registry for tracking all pre-trained models.
Maintains a central index of all saved models for easy lookup.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ModelEntry:
    """Entry for a single pre-trained model."""
    exp_name: str
    seed: int
    checkpoint_path: str
    manifest_path: str
    best_val_total: float
    best_epoch: int
    training_completed: bool
    wandb_run_id: Optional[str] = None
    wandb_artifact_name: Optional[str] = None
    timestamp: Optional[str] = None
    domains: Optional[List[str]] = None
    tasks: Optional[List[str]] = None

class ModelRegistry:
    """Central registry for all pre-trained models."""
    
    def __init__(self, registry_path: str = "outputs/pretrain/model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models: Dict[str, ModelEntry] = {}
        self.load_registry()
    
    def register_model(self, entry: ModelEntry) -> None:
        """Register a new model or update existing entry."""
        key = f"{entry.exp_name}_seed{entry.seed}"
        self.models[key] = entry
        self.save_registry()
    
    def get_model(self, exp_name: str, seed: int) -> Optional[ModelEntry]:
        """Get model entry by experiment name and seed."""
        key = f"{exp_name}_seed{seed}"
        return self.models.get(key)
    
    def get_models_by_scheme(self, exp_name: str) -> List[ModelEntry]:
        """Get all models for a specific experiment scheme."""
        return [model for model in self.models.values() if model.exp_name == exp_name]
    
    def get_best_model_per_scheme(self) -> Dict[str, ModelEntry]:
        """Get the best model (lowest val loss) for each scheme."""
        best_models = {}
        for model in self.models.values():
            if model.exp_name not in best_models:
                best_models[model.exp_name] = model
            elif model.best_val_total < best_models[model.exp_name].best_val_total:
                best_models[model.exp_name] = model
        return best_models
    
    def list_completed_models(self) -> List[ModelEntry]:
        """Get all successfully completed models."""
        return [model for model in self.models.values() if model.training_completed]
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the registry."""
        completed = self.list_completed_models()
        schemes = set(model.exp_name for model in completed)
        
        return {
            "total_models": len(self.models),
            "completed_models": len(completed),
            "unique_schemes": len(schemes),
            "schemes": sorted(schemes),
            "completion_rate": len(completed) / len(self.models) if self.models else 0,
            "avg_val_loss": sum(m.best_val_total for m in completed) / len(completed) if completed else 0
        }
    
    def load_registry(self) -> None:
        """Load registry from file."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                self.models = {k: ModelEntry(**v) for k, v in data.items()}
    
    def save_registry(self) -> None:
        """Save registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            data = {k: asdict(v) for k, v in self.models.items()}
            json.dump(data, f, indent=2)
    
    def print_summary(self) -> None:
        """Print a summary of all registered models."""
        stats = self.get_summary_stats()
        
        print("=" * 60)
        print("MODEL REGISTRY SUMMARY")
        print("=" * 60)
        print(f"Total models: {stats['total_models']}")
        print(f"Completed: {stats['completed_models']}")
        print(f"Completion rate: {stats['completion_rate']:.1%}")
        print(f"Unique schemes: {stats['unique_schemes']}")
        if stats['completed_models'] > 0:
            print(f"Average val loss: {stats['avg_val_loss']:.4f}")
        
        print("\nBest model per scheme:")
        best_models = self.get_best_model_per_scheme()
        for scheme, model in sorted(best_models.items()):
            print(f"  {scheme}: seed{model.seed} (val_loss: {model.best_val_total:.4f})")

# Global registry instance
_registry = None

def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry

def register_model_completion(exp_name: str, seed: int, checkpoint_path: str, 
                            manifest_path: str, best_val_total: float, best_epoch: int,
                            wandb_run_id: str = None, wandb_artifact_name: str = None,
                            domains: List[str] = None, tasks: List[str] = None) -> None:
    """Convenience function to register a completed model."""
    entry = ModelEntry(
        exp_name=exp_name,
        seed=seed,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        best_val_total=best_val_total,
        best_epoch=best_epoch,
        training_completed=True,
        wandb_run_id=wandb_run_id,
        wandb_artifact_name=wandb_artifact_name,
        timestamp=datetime.now().isoformat(),
        domains=domains,
        tasks=tasks
    )
    
    registry = get_registry()
    registry.register_model(entry)
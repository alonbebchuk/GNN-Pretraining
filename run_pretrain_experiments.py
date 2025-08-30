#!/usr/bin/env python3
"""
Run all pretraining experiments using sweep configuration.
Executes the full pretraining experimental design from the research plan.
"""

import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

PRETRAIN_SWEEP_CONFIG = "configs/sweeps/pretrain.yaml"

# Map absolute paths to local paths
CONFIG_MAPPING = {
    "/kaggle/working/gnn-pretraining/configs/pretrain/b2_nfm.yaml": "configs/pretrain/b2_nfm.yaml",
    "/kaggle/working/gnn-pretraining/configs/pretrain/b3_nc.yaml": "configs/pretrain/b3_nc.yaml", 
    "/kaggle/working/gnn-pretraining/configs/pretrain/b4_single_domain_all.yaml": "configs/pretrain/b4_single_domain_all.yaml",
    "/kaggle/working/gnn-pretraining/configs/pretrain/s1_multi_task_generative.yaml": "configs/pretrain/s1_multi_task_generative.yaml",
    "/kaggle/working/gnn-pretraining/configs/pretrain/s2_multi_task_contrastive.yaml": "configs/pretrain/s2_multi_task_contrastive.yaml",
    "/kaggle/working/gnn-pretraining/configs/pretrain/s3_all_self_supervised.yaml": "configs/pretrain/s3_all_self_supervised.yaml",
    "/kaggle/working/gnn-pretraining/configs/pretrain/s4_all_objectives.yaml": "configs/pretrain/s4_all_objectives.yaml",
    "/kaggle/working/gnn-pretraining/configs/pretrain/s5_all_objectives_da.yaml": "configs/pretrain/s5_all_objectives_da.yaml"
}

def load_pretrain_sweep_config() -> Dict[str, Any]:
    """Load pretraining sweep configuration."""
    with open(PRETRAIN_SWEEP_CONFIG, 'r') as f:
        return yaml.safe_load(f)

def generate_pretrain_combinations(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations for pretraining."""
    params = sweep_config['parameters']
    
    combinations = []
    for config_path in params['config']['values']:
        local_config = CONFIG_MAPPING.get(config_path, config_path)
        for seed in params['seed']['values']:
            combinations.append({
                'config': local_config,
                'seed': seed
            })
    
    return combinations

def extract_scheme_name(config_path: str) -> str:
    """Extract scheme name from config path for logging."""
    return Path(config_path).stem

def run_pretrain_experiment(config_path: str, seed: int) -> bool:
    """Run a single pretraining experiment."""
    cmd = [
        sys.executable, "-m", "src.pretrain.pretrain",
        "--config", config_path,
        "--seed", str(seed)
    ]
    
    scheme_name = extract_scheme_name(config_path)
    
    try:
        print(f"Running: {scheme_name} (seed {seed})")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success: {scheme_name} (seed {seed})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {scheme_name} (seed {seed})")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Run all pretraining experiments using sweep configuration."""
    print("🚀 Starting pretraining experiments...")
    
    # Load sweep config
    sweep_config = load_pretrain_sweep_config()
    combinations = generate_pretrain_combinations(sweep_config)
    
    total_experiments = len(combinations)
    print(f"   📋 Total experiments: {total_experiments}")
    print(f"   🎯 Schemes: 8 pretraining schemes (b2-b4, s1-s5)")
    print(f"   🎲 Seeds: [42, 84, 126]")
    
    successful = 0
    failed = 0
    
    # Run experiments
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{total_experiments}] ", end="")
        
        if run_pretrain_experiment(params['config'], params['seed']):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n📊 PRETRAINING RESULTS:")
    print(f"   Total experiments: {total_experiments}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success rate: {successful/total_experiments*100:.1f}%")
    
    if failed > 0:
        print(f"\n⚠️  {failed} experiments failed - check logs above")
        sys.exit(1)
    else:
        print(f"\n🎉 All pretraining experiments completed successfully!")
        print(f"   Ready for finetuning experiments...")

if __name__ == "__main__":
    main()

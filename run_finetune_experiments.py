#!/usr/bin/env python3
"""
Run all finetuning experiments using sweep configurations.
Clean alternative to 108 individual config files.
"""

import itertools
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Sweep configuration files
SWEEP_CONFIGS = [
    "configs/sweeps/finetune_graph_classification.yaml",
    "configs/sweeps/finetune_node_classification.yaml", 
    "configs/sweeps/finetune_link_prediction.yaml"
]

def load_sweep_config(sweep_file: str) -> Dict[str, Any]:
    """Load sweep configuration from YAML file."""
    with open(sweep_file, 'r') as f:
        return yaml.safe_load(f)

def generate_parameter_combinations(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from sweep config."""
    params = sweep_config['parameters']
    
    # Extract parameter names and values (excluding lr_backbone which is handled conditionally)
    param_names = []
    param_values = []
    
    for param_name, param_spec in params.items():
        if param_name in ['lr_backbone']:  # Skip conditional params
            continue
        param_names.append(param_name)
        param_values.append(param_spec['values'])
    
    # Generate Cartesian product
    combinations = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        
        # Add conditional lr_backbone parameter
        if param_dict['finetune_strategy'] == 'full_finetune':
            # Higher learning rate for node/link tasks
            if 'node_classification' in sweep_config.get('config_template', '') or 'link_prediction' in sweep_config.get('config_template', ''):
                param_dict['lr_backbone'] = 5e-4
            else:
                param_dict['lr_backbone'] = 1e-4
        else:  # linear_probe
            param_dict['lr_backbone'] = 0.0
            
        combinations.append(param_dict)
    
    return combinations

def create_config_from_template(template_path: str, params: Dict[str, Any]) -> str:
    """Create a concrete config file from template by substituting parameters."""
    # Load template
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Substitute parameters
    for param_name, param_value in params.items():
        placeholder = f"${{{param_name}}}"
        template_content = template_content.replace(placeholder, str(param_value))
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(template_content)
        return f.name

def run_experiment(template_path: str, params: Dict[str, Any]) -> bool:
    """Run a single experiment with given parameters."""
    # Create config file from template
    config_path = create_config_from_template(template_path, params)
    
    try:
        # Run experiment like pretrain does - just config + seed
        cmd = [
            sys.executable, "-m", "src.finetune.finetune",
            "--config", config_path,
            "--seed", str(params['seed'])
        ]
        
        print(f"Running: {params['domain_name']} {params['finetune_strategy']} {params['pretrained_scheme']} (seed {params['seed']})")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr}")
        return False
    finally:
        # Clean up temporary config file
        Path(config_path).unlink(missing_ok=True)

def main():
    """Run all finetuning experiments using sweep approach."""
    total_experiments = 0
    successful = 0
    failed = 0
    
    for sweep_file in SWEEP_CONFIGS:
        print(f"\n🔄 Processing {sweep_file}")
        
        # Load sweep config
        sweep_config = load_sweep_config(sweep_file)
        template_path = sweep_config['config_template']
        
        # Generate parameter combinations
        combinations = generate_parameter_combinations(sweep_config)
        print(f"   Generated {len(combinations)} parameter combinations")
        
        # Run experiments
        for i, params in enumerate(combinations):
            print(f"   [{i+1}/{len(combinations)}] ", end="")
            
            if run_experiment(template_path, params):
                successful += 1
            else:
                failed += 1
            
            total_experiments += 1
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Total experiments: {total_experiments}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()

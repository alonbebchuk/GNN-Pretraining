#!/usr/bin/env python3
"""
Generate finetuning configuration files using a template-based approach.

This script generates configs from base templates and sweep parameters,
providing a cleaner and more maintainable alternative to individual config files.

Based on the plan, we support:
- Downstream tasks: Cora_NC, CiteSeer_NC, Cora_LP, CiteSeer_LP (4)
- Pretraining schemes: B1-B4, S1-S5 (9 schemes)  
- Strategies: full_finetune, linear_probe (2)
Total: 4 × 9 × 2 = 72 configurations
"""

import os
import yaml
from pathlib import Path
from string import Template
from typing import Dict, List, Any

# Task configuration mapping - 4 dataset variants from 2 base datasets
TASK_CONFIGS = {
    # Node Classification Tasks
    'Cora_NC': {
        'task_type': 'node_classification',
        'template': 'base_node_classification.yaml',
        'batch_size': 512,
        'epochs': 200,
        'lr_backbone_full': 5e-4,
        'lr_head': 1e-3,
        'patience': 20,
        'comment': 'Node classification variant of Cora citation network'
    },
    'CiteSeer_NC': {
        'task_type': 'node_classification',
        'template': 'base_node_classification.yaml',
        'batch_size': 512, 
        'epochs': 200,
        'lr_backbone_full': 5e-4,
        'lr_head': 1e-3,
        'patience': 20,
        'comment': 'Node classification variant of CiteSeer citation network'
    },
    
    # Link Prediction Tasks
    'Cora_LP': {
        'task_type': 'link_prediction',
        'template': 'base_link_prediction.yaml',
        'batch_size': 256,
        'epochs': 300,
        'lr_backbone_full': 5e-4,
        'lr_head': 1e-3,
        'patience': 30,
        'comment': 'Link prediction variant of Cora citation network'
    },
    'CiteSeer_LP': {
        'task_type': 'link_prediction',
        'template': 'base_link_prediction.yaml',
        'batch_size': 256, 
        'epochs': 300,
        'lr_backbone_full': 5e-4,
        'lr_head': 1e-3,
        'patience': 30,
        'comment': 'Link prediction variant of CiteSeer citation network'
    }
}

PRETRAINING_SCHEMES = [
    # Baselines
    'b1_from_scratch',         # B1: From-Scratch baseline
    'b2_nfm',                  # B2: Single-Task (NFM) baseline  
    'b3_nc',                   # B3: Single-Task (NC) baseline
    'b4_single_domain_all',    # B4: Single-Domain baseline
    
    # Schemes
    's1_multi_task_generative',    # S1: Multi-Task Generative (NFM+LP)
    's2_multi_task_contrastive',   # S2: Multi-Task Contrastive (NC+GC)
    's3_all_self_supervised',      # S3: All Self-Supervised (NFM+LP+NC+GC)
    's4_all_objectives',           # S4: All Objectives (NFM+LP+NC+GC+GPP)
    's5_all_objectives_da',        # S5: Domain-Invariant (S4+DA)
]

FINETUNE_STRATEGIES = ['full_finetune', 'linear_probe']

# Seeds for experiments
SEEDS = [42, 84, 126]


def load_template(template_name: str) -> str:
    """Load a config template file."""
    project_root = Path(__file__).parent.parent.parent
    template_path = project_root / "configs" / "finetune" / template_name
    
    with template_path.open('r', encoding='utf-8') as f:
        return f.read()


def substitute_template(template_content: str, variables: Dict[str, Any]) -> str:
    """Substitute variables in template using safe string substitution."""
    template = Template(template_content)
    return template.safe_substitute(variables)


def generate_config_variables(
    domain_name: str,
    scheme: str,
    strategy: str
) -> Dict[str, Any]:
    """Generate variable dictionary for template substitution."""
    
    task_config = TASK_CONFIGS[domain_name]
    
    # Determine learning rates based on strategy
    if strategy == 'full_finetune':
        lr_backbone = task_config['lr_backbone_full']
    else:  # linear_probe
        lr_backbone = 0.0
    
    # Create variable dictionary
    variables = {
        'exp_name': f"{domain_name.lower()}_{strategy}_{scheme}",
        'domain_name': domain_name,
        'pretrained_scheme': scheme,
        'finetune_strategy': strategy,
        'batch_size': task_config['batch_size'],
        'epochs': task_config['epochs'],
        'lr_backbone': lr_backbone,
        'lr_head': task_config['lr_head'],
        'patience': task_config['patience'],
    }
    
    return variables


def generate_individual_configs() -> None:
    """Generate individual config files from templates (compatible with current system)."""
    
    project_root = Path(__file__).parent.parent.parent
    output_base = project_root / "configs" / "finetune" / "generated"
    
    # Create output directories
    node_dir = output_base / "node_classification"
    link_dir = output_base / "link_prediction"
    
    node_dir.mkdir(parents=True, exist_ok=True)
    link_dir.mkdir(parents=True, exist_ok=True)
    
    total_configs = 0
    
    for domain_name, task_config in TASK_CONFIGS.items():
        # Determine output directory
        if task_config['task_type'] == 'node_classification':
            output_dir = node_dir
        elif task_config['task_type'] == 'link_prediction':
            output_dir = link_dir
        else:
            raise ValueError(f"Unknown task type: {task_config['task_type']}")
        
        # Load template
        template_content = load_template(task_config['template'])
        
        for scheme in PRETRAINING_SCHEMES:
            for strategy in FINETUNE_STRATEGIES:
                # Generate variables
                variables = generate_config_variables(domain_name, scheme, strategy)
                
                # Substitute template
                config_content = substitute_template(template_content, variables)
                
                # Add comment header
                header = f"# {domain_name} {task_config['task_type'].replace('_', ' ').title()}\n"
                header += f"# {task_config['comment']}\n"
                header += f"# Generated from template: {task_config['template']}\n\n"
                
                final_content = header + config_content
                
                # Write config file
                filename = f"{domain_name.lower()}_{strategy}_{scheme}.yaml"
                config_path = output_dir / filename
                
                with config_path.open('w', encoding='utf-8') as f:
                    f.write(final_content)
                
                total_configs += 1
                
                if total_configs % 20 == 0:
                    print(f"Generated {total_configs} configs...")
    
    print(f"\n✅ Generated {total_configs} individual configuration files:")
    print(f"   📁 Node classification:  {len(list(node_dir.glob('*.yaml')))}")
    print(f"   📁 Link prediction:      {len(list(link_dir.glob('*.yaml')))}")


def generate_sweep_runner_script() -> None:
    """Generate a script to run experiments using sweep configs."""
    
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / "run_sweep_finetuning.py"
    
    script_content = '''#!/usr/bin/env python3
"""
Run finetuning experiments using sweep configurations.
This provides a cleaner alternative to individual config files.
"""

import itertools
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Load task configurations
TASK_CONFIGS = {
    'Cora_NC': {'task_type': 'node_classification'},
    'CiteSeer_NC': {'task_type': 'node_classification'},
    'Cora_LP': {'task_type': 'link_prediction'},
    'CiteSeer_LP': {'task_type': 'link_prediction'},
}

PRETRAINING_SCHEMES = [
    'b1_from_scratch', 'b2_nfm', 'b3_nc', 'b4_single_domain_all',
    's1_multi_task_generative', 's2_multi_task_contrastive', 
    's3_all_self_supervised', 's4_all_objectives', 's5_all_objectives_da'
]

FINETUNE_STRATEGIES = ['full_finetune', 'linear_probe']
SEEDS = [42, 84, 126]


def load_sweep_parameters(task_type: str) -> Dict[str, Any]:
    """Load sweep parameters for given task type."""
    sweep_file = Path(f"configs/sweeps/finetune_{task_type}.yaml")
    with sweep_file.open('r') as f:
        return yaml.safe_load(f)


def run_experiment_from_params(params: Dict[str, Any]) -> bool:
    """Run a single experiment with given parameters."""
    # Determine task type and base config
    domain_name = params['domain_name']
    task_type = TASK_CONFIGS[domain_name]['task_type']
    base_config = f"configs/finetune/base_{task_type}.yaml"
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.finetuning.main_finetune",
        "--config", base_config,
        "--seed", str(params['seed']),
        "--domain", params['domain_name'],
        "--scheme", params['pretrained_scheme'],
        "--strategy", params['finetune_strategy']
    ]
    
    try:
        print(f"Running: {params['domain_name']} {params['finetune_strategy']} {params['pretrained_scheme']} (seed {params['seed']})")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr}")
        return False


def main():
    """Run all experiments using sweep approach."""
    # Generate all parameter combinations
    experiments = []
    
    for domain_name in TASK_CONFIGS.keys():
        for scheme in PRETRAINING_SCHEMES:
            for strategy in FINETUNE_STRATEGIES:
                for seed in SEEDS:
                    experiments.append({
                        'domain_name': domain_name,
                        'pretrained_scheme': scheme,
                        'finetune_strategy': strategy,
                        'seed': seed
                    })
    
    total_runs = len(experiments)
    print(f"🚀 Starting {total_runs} finetuning experiments using sweep approach...")
    
    successful = 0
    failed = 0
    
    for i, params in enumerate(experiments):
        print(f"\\n[{i+1}/{total_runs}] ", end="")
        
        if run_experiment_from_params(params):
            successful += 1
        else:
            failed += 1
    
    print(f"\\n📊 RESULTS:")
    print(f"   ✅ Successful: {successful}/{total_runs}")
    print(f"   ❌ Failed: {failed}/{total_runs}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
    
    with script_path.open('w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"📜 Generated sweep-based experiment runner: {script_path}")


def update_legacy_runner() -> None:
    """Update the existing runner to use the new generated configs."""
    
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / "run_all_finetuning.py"
    
    # Build experiment list using generated configs
    experiments = []
    config_base = Path("configs") / "finetune" / "generated"
    
    for domain_name, task_config in TASK_CONFIGS.items():
        task_type = task_config['task_type']
        
        for scheme in PRETRAINING_SCHEMES:
            for strategy in FINETUNE_STRATEGIES:
                config_path = config_base / task_type / f"{domain_name.lower()}_{strategy}_{scheme}.yaml"
                experiments.append(str(config_path))
    
    # Generate updated script content
    script_content = f'''#!/usr/bin/env python3
"""
Run all finetuning experiments using generated config files.
Generated automatically by generate_configs.py.
"""

import subprocess
import sys
from pathlib import Path

# Seeds for statistical robustness
SEEDS = {SEEDS}

# All experiments (generated from templates)
EXPERIMENTS = [
'''
    
    for exp in experiments:
        script_content += f'    "{exp}",\n'
    
    script_content += ''']

def run_experiment(config_path: str, seed: int) -> bool:
    """Run a single experiment."""
    cmd = [
        sys.executable, "-m", "src.finetuning.main_finetune",
        "--config", config_path,
        "--seed", str(seed)
    ]
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success: {config_path} (seed {seed})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {config_path} (seed {seed})")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run all experiments."""
    total_runs = len(EXPERIMENTS) * len(SEEDS)
    print(f"🚀 Starting {total_runs} finetuning experiments...")
    print(f"   📋 Configs: {len(EXPERIMENTS)}")
    print(f"   🎲 Seeds: {SEEDS}")
    
    successful = 0
    failed = 0
    
    for i, config_path in enumerate(EXPERIMENTS):
        for seed in SEEDS:
            print(f"\\n[{successful + failed + 1}/{total_runs}] {config_path} (seed {seed})")
            
            if run_experiment(config_path, seed):
                successful += 1
            else:
                failed += 1
    
    print(f"\\n📊 RESULTS:")
    print(f"   ✅ Successful: {successful}/{total_runs}")
    print(f"   ❌ Failed: {failed}/{total_runs}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with script_path.open('w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"📜 Updated legacy experiment runner: {script_path}")


def main() -> None:
    """Main function to generate all configurations."""
    print("🏗️  Generating finetuning configurations using template-based approach...")
    
    # Generate individual config files (for backward compatibility)
    print("\n1. Generating individual config files from templates...")
    generate_individual_configs()
    
    # Generate sweep-based runner (new clean approach)
    print("\n2. Generating sweep-based experiment runner...")
    generate_sweep_runner_script()
    
    # Update legacy runner to use new generated configs
    print("\n3. Updating legacy experiment runner...")
    update_legacy_runner()
    
    print("\n🎉 All files generated successfully!")
    print("\n📋 Available approaches:")
    print("   • Template-based: Use base configs + sweep parameters")
    print("   • Individual configs: Generated from templates (legacy compatibility)")
    print("   • Sweep runner: run_sweep_finetuning.py (recommended)")
    print("   • Legacy runner: run_all_finetuning.py (uses individual configs)")
    
    print("\n🏃 Next steps:")
    print("1. Run pretraining experiments to generate pretrained models")
    print("2a. Clean approach: python run_sweep_finetuning.py")
    print("2b. Legacy approach: python run_all_finetuning.py")
    print("3. Analyze results for research questions")


if __name__ == "__main__":
    main()
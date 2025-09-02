import argparse
import subprocess
import sys
import torch
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path


def get_num_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def run_single_experiment(params):
    domain_name, finetune_strategy, pretrained_scheme, seed = params
    print(f"[GPU-{torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}] Running finetuning: {domain_name} | {finetune_strategy} | {pretrained_scheme} (seed={seed})")

    try:
        cmd = [
            sys.executable, "src/finetune/finetune.py",
            "--domain_name", domain_name,
            "--finetune_strategy", finetune_strategy,
            "--pretrained_scheme", pretrained_scheme,
            "--seed", str(seed)
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Completed: {domain_name} | {finetune_strategy} | {pretrained_scheme} (seed={seed})")
        return True, domain_name, finetune_strategy, pretrained_scheme, seed, None

    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}: {e.stderr}"
        print(f"✗ Failed: {domain_name} | {finetune_strategy} | {pretrained_scheme} (seed={seed}) - {error_msg}")
        return False, domain_name, finetune_strategy, pretrained_scheme, seed, error_msg


def run_sweep():
    domain_names = ["Cora_NC", "CiteSeer_NC", "Cora_LP", "CiteSeer_LP", "ENZYMES", "PTC_MR"]
    finetune_strategies = ["full_finetune", "linear_probe"]
    pretrained_schemes = ["b1", "b2", "b3", "b4", "s1", "s2", "s3", "s4", "s5"]
    seeds = [42, 84, 126]

    experiments = list(product(domain_names, finetune_strategies, pretrained_schemes, seeds))
    total_experiments = len(experiments)

    num_gpus = get_num_gpus()

    print(f"Starting finetuning sweep: {total_experiments} experiments")
    print(f"Using {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'} for parallel execution")
    print("=" * 50)

    completed = 0
    failed = 0
    failed_experiments = []

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(run_single_experiment, exp_params) for exp_params in experiments]

        for future in futures:
            success, domain_name, finetune_strategy, pretrained_scheme, seed, error_msg = future.result()
            if success:
                completed += 1
            else:
                failed += 1
                failed_experiments.append((domain_name, finetune_strategy, pretrained_scheme, seed, error_msg))

            print(f"Progress: {completed + failed}/{total_experiments} (✓{completed} ✗{failed})")

    print("=" * 50)
    print(f"Finetuning sweep completed!")
    print(f"✓ Successful: {completed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {total_experiments}")

    if failed_experiments:
        print("\nFailed experiments:")
        for domain_name, finetune_strategy, pretrained_scheme, seed, error_msg in failed_experiments:
            print(f"  - {domain_name}_{finetune_strategy}_{pretrained_scheme} (seed={seed}): {error_msg}")


def run_domain_sweep(domain_name: str):
    finetune_strategies = ["full_finetune", "linear_probe"]
    pretrained_schemes = ["b1", "b2", "b3", "b4", "s1", "s2", "s3", "s4", "s5"]
    # seeds = [42, 84, 126]
    seeds = [42]

    experiments = [(domain_name, strategy, scheme, seed) for strategy, scheme, seed in product(finetune_strategies, pretrained_schemes, seeds)]
    total_experiments = len(experiments)

    num_gpus = get_num_gpus()

    print(f"Starting finetuning sweep for {domain_name}: {total_experiments} experiments")
    print(f"Using {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'} for parallel execution")
    print("=" * 50)

    completed = 0
    failed = 0
    failed_experiments = []

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(run_single_experiment, exp_params) for exp_params in experiments]

        for future in futures:
            success, domain, strategy, scheme, seed, error_msg = future.result()
            if success:
                completed += 1
            else:
                failed += 1
                failed_experiments.append((domain, strategy, scheme, seed, error_msg))

            print(f"Progress: {completed + failed}/{total_experiments} (✓{completed} ✗{failed})")

    print("=" * 50)
    print(f"Finetuning sweep for {domain_name} completed!")
    print(f"✓ Successful: {completed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {total_experiments}")

    if failed_experiments:
        print("\nFailed experiments:")
        for domain, strategy, scheme, seed, error_msg in failed_experiments:
            print(f"  - {domain}_{strategy}_{scheme} (seed={seed}): {error_msg}")


def main():
    parser = argparse.ArgumentParser(description="Run finetuning experiments")
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    parser.add_argument("--domain_sweep", type=str, help="Run sweep for specific domain")
    parser.add_argument("--domain_name", type=str, help="Domain name")
    parser.add_argument("--finetune_strategy", type=str, help="Finetune strategy")
    parser.add_argument("--pretrained_scheme", type=str, help="Pretrained scheme")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    elif args.domain_sweep:
        run_domain_sweep(args.domain_sweep)
    elif args.domain_name and args.finetune_strategy and args.pretrained_scheme:
        success, _, _, _, _, error_msg = run_single_experiment((args.domain_name, args.finetune_strategy, args.pretrained_scheme, args.seed))
        if not success:
            print(f"Experiment failed: {error_msg}")
            sys.exit(1)
    else:
        print("Please specify one of:")
        print("  --sweep (run full sweep)")
        print("  --domain_sweep DOMAIN (run sweep for specific domain)")
        print("  --domain_name + --finetune_strategy + --pretrained_scheme + --seed (run single experiment)")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()

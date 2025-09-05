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
    exp_name, seed = params
    print(f"[GPU-{torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}] Running pretraining: {exp_name} (seed={seed})")

    try:
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())

        cmd = [
            sys.executable, "src/pretrain/pretrain.py",
            "--exp_name", exp_name,
            "--seed", str(seed)
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(f"✓ Completed: {exp_name} (seed={seed})")
        return True, exp_name, seed, None

    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}: {e.stderr}"
        print(f"✗ Failed: {exp_name} (seed={seed}) - {error_msg}")
        return False, exp_name, seed, error_msg


def run_sweep():
    exp_names = ["b2", "b3", "b4", "s1", "s2", "s3", "s4", "s5"]
    seeds = [42, 84, 126]

    experiments = list(product(exp_names, seeds))
    total_experiments = len(experiments)

    num_gpus = get_num_gpus()

    print(f"Starting pretraining sweep: {total_experiments} experiments")
    print(f"Using {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'} for parallel execution")
    print("=" * 50)

    completed = 0
    failed = 0
    failed_experiments = []

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(run_single_experiment, exp_params) for exp_params in experiments]

        for future in futures:
            success, exp_name, seed, error_msg = future.result()
            if success:
                completed += 1
            else:
                failed += 1
                failed_experiments.append((exp_name, seed, error_msg))

            print(f"Progress: {completed + failed}/{total_experiments} (✓{completed} ✗{failed})")

    print("=" * 50)
    print(f"Pretraining sweep completed!")
    print(f"✓ Successful: {completed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {total_experiments}")

    if failed_experiments:
        print("\nFailed experiments:")
        for exp_name, seed, error_msg in failed_experiments:
            print(f"  - {exp_name} (seed={seed}): {error_msg}")


def main():
    parser = argparse.ArgumentParser(description="Run pretraining experiments")
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    elif args.exp_name:
        success, _, _, error_msg = run_single_experiment((args.exp_name, args.seed))
        if not success:
            print(f"Experiment failed: {error_msg}")
            sys.exit(1)
    else:
        print("Please specify either --sweep or --exp_name + --seed")
        sys.exit(1)


if __name__ == "__main__":
    main()

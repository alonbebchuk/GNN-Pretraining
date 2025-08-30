from pathlib import Path

import wandb

PRETRAIN_OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "pretrain"
PROJECT_NAME = "gnn-pretraining-pretrain"


def download_wandb_artifact(artifact_name: str) -> str:
    api = wandb.Api()
    full_artifact_name = f"{PROJECT_NAME}/{artifact_name}"
    artifact = api.artifact(full_artifact_name)
    artifact.download(root=str(PRETRAIN_OUTPUT_DIR))


def get_pretrained_model_path(scheme_name: str, seed: int) -> str:
    model_name = f"best_model_{scheme_name}_{seed}"
    model_path = PRETRAIN_OUTPUT_DIR / f"{model_name}.pt"

    if not model_path.exists():
        download_wandb_artifact(artifact_name=model_name)

    return str(model_path)

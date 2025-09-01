# GNN Project — Google Cloud VM Setup & Team Workflow (Markdown Guide)

This guide shows **each teammate** how to create a personal Google Cloud VM with an **NVIDIA L4** GPU, set up a **project-local Conda environment**, and use **Weights & Biases (W&B)** to share runs and checkpoints. It also covers creating SSH keys (for GitHub and for VM login) and connecting over SSH.

> **Costs**: You pay while a VM is **running** (GPU + CPU + boot disk). **Stop** the VM when you’re not training to avoid GPU/CPU charges. Stopped VMs still incur disk costs.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Create SSH Keys (Local) & Add to GitHub](#create-ssh-keys-local--add-to-github)
3. [Weights & Biases Account & API Key](#weights--biases-account--api-key)
4. [GCP Quotas & Region](#gcp-quotas--region)
5. [Create the VM (NVIDIA L4, us-east1)](#create-the-vm-nvidia-l4-us-east1)
6. [First Login & GPU Driver Check](#first-login--gpu-driver-check)
7. [Add Your SSH Key to the VM & Local SSH Config](#add-your-ssh-key-to-the-vm--local-ssh-config)
8. [One-shot Environment Bootstrap Script](#one-shot-environment-bootstrap-script)
9. [Join the Same W&B Project](#join-the-same-wb-project)
10. [Day-to-day Workflow](#day-to-day-workflow)
11. [Troubleshooting](#troubleshooting)
12. [Appendix: Optional .bashrc Enhancements](#appendix-optional-bashrc-enhancements)

---

## Prerequisites

- A personal **Google Cloud** account (each teammate uses their own $300 free tier, if available).
- Project Git repo: `https://github.com/alonbebchuk/GNN-Pretraining.git`.
- Comfortable with a terminal (macOS/Linux) or **PowerShell** (Windows).

---

## Create SSH Keys (Local) & Add to GitHub

You’ll use a single SSH key pair for:

- **GitHub** (to push/pull without passwords), and
- **VM login** (public key goes in the VM’s `~/.ssh/authorized_keys`).

### 1) Generate a key (if you don’t already have one)

macOS / Linux

```bash
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
```

Windows (PowerShell)

```powershell
if (!(Test-Path "$HOME\.ssh\id_ed25519.pub")) { ssh-keygen -t ed25519 }
# Press Enter to accept defaults. It creates:
# Private key: ~/.ssh/id_ed25519
# Public key:  ~/.ssh/id_ed25519.pub
```

### 2) Add the public key to GitHub

Copy your public key to the clipboard:

- macOS: `pbcopy < ~/.ssh/id_ed25519.pub`
- Linux: `xclip -sel clip < ~/.ssh/id_ed25519.pub` (or open the file and copy manually)
- Windows: `Get-Content "$HOME\.ssh\id_ed25519.pub" | Set-Clipboard`

Go to GitHub → Settings → SSH and GPG keys → New SSH key. Paste the key, give it a title (e.g., “Laptop-ed25519”), and save.

Keep your private key secret. Never upload `id_ed25519` anywhere.

---

## Weights & Biases Account & API Key

Create an account at https://wandb.ai and sign in.

Find your API Key at https://wandb.ai/authorize (Profile → Settings → API keys). You’ll need to paste this once on your VM with `wandb login`.

We’ll use the shared project:

- Entity: `benc6116-tel-aviv-university`
- Project: `gnn-pretraining`

(You can change these strings if your team decides on different names.)

---

## GCP Quotas & Region

In IAM & Admin → Quotas, ensure you have:

- Compute Engine API → GPUs (all regions) ≥ 1
- Compute Engine API → NVIDIA L4 GPUs ≥ 1 in your chosen region (we use `us-east1`).

Region choice: `us-east1` is typically cheaper. Latency to Israel is fine for SSH/editing; training runs on the box.

---

## Create the VM (NVIDIA L4, us-east1)

Console → Compute Engine → VM instances → Create instance

Basics

- Name: `gnn-l4-01` (or your own)
- Region: `us-east1` (zone e.g., `us-east1-c`)
- Machine family: GPU
- Series: G2
- Machine type: `g2-standard-8` (8 vCPU, 32 GB RAM)

GPU

- Type: NVIDIA L4
- Count: 1

Boot disk

- Image: Deep Learning VM for PyTorch 2.4 with CUDA 12.4
- Size: 150 GB (adjust as needed)

Networking / Firewall

Leave defaults. No need for HTTP/HTTPS rules for SSH-only usage.

Click Create.

Billing tip: You pay while the VM is running. Stop the VM when idle to avoid GPU/CPU costs.

---

## First Login & GPU Driver Check

Open the VM row → click SSH (in the GCP Console). The DLVM image typically offers driver installation automatically on first boot.

Verify:

```bash
nvidia-smi
```

You should see NVIDIA L4 and CUDA 12.4.

---

## Add Your SSH Key to the VM & Local SSH Config

1) Append your public key to the VM’s `authorized_keys`

In the VM (web SSH terminal):

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA...YOUR_PUBLIC_KEY...' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Replace the string with the contents of your local `~/.ssh/id_ed25519.pub`.

2) Add a host alias on your local machine

Edit `~/.ssh/config` (create if absent) and add:

```ssh-config
Host gnn-l4-01
  HostName <VM-External-IP>   # shown in the VM list
  User <your-linux-username-on-vm>  # e.g., your VM login name
  IdentityFile ~/.ssh/id_ed25519
  ForwardAgent yes
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

Now you can connect directly:

```bash
ssh gnn-l4-01
```

---

## One-shot Environment Bootstrap Script

Run this on the VM (web SSH or your own SSH). It installs Miniconda, clones the repo, creates a project-local env `./.conda`, installs Torch 2.6.0 (CUDA 12.4), PyG wheels, common deps, and sets W&B defaults.

```bash
cat > ~/bootstrap_gnn_vm.sh <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------- editable variables -----------------
REPO_URL="https://github.com/alonbebchuk/GNN-Pretraining.git"
WORKSPACE="$HOME/workspace"
PROJECT_DIR="$WORKSPACE/GNN-Pretraining"
PYTHON_VER="3.10"

# Set this to the SAME entity & project for all teammates:
WANDB_ENTITY="benc6116-tel-aviv-university"
WANDB_PROJECT="gnn-pretraining"
# ------------------------------------------------------

echo "==> Basic tools"
sudo apt-get update -y
sudo apt-get install -y git curl build-essential tmux htop

echo "==> Ensure NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [[ -x /opt/deeplearning/install-driver.sh ]]; then
    sudo /opt/deeplearning/install-driver.sh
  fi
fi
nvidia-smi || true

echo "==> Miniconda"
if [[ ! -d "$HOME/miniconda" ]]; then
  curl -fsSLo "$HOME/miniconda.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda"
fi
eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
conda config --set always_yes true

echo "==> Accept Anaconda ToS (non-interactive)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "==> Clone repo"
mkdir -p "$WORKSPACE"
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

echo "==> Create & activate project-local conda env ($PROJECT_DIR/.conda)"
if [[ ! -d "$PROJECT_DIR/.conda" ]]; then
  conda create -p ./.conda "python=$PYTHON_VER"
fi
conda activate "$PROJECT_DIR/.conda"

echo "==> Upgrade pip"
pip install --upgrade pip

echo "==> PyTorch 2.6.0 (CUDA 12.4)"
pip install --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

echo "==> PyTorch Geometric (match torch 2.6.0 + cu124)"
pip install -U pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch-geometric

echo "==> Project utilities"
pip install wandb scikit-learn pyyaml tqdm networkx pandas matplotlib ogb

echo "==> Persist W&B defaults"
grep -q "WANDB_PROJECT=" "$HOME/.bashrc" || echo "export WANDB_PROJECT=$WANDB_PROJECT" >> "$HOME/.bashrc"
grep -q "WANDB_ENTITY="  "$HOME/.bashrc" || echo "export WANDB_ENTITY=$WANDB_ENTITY"  >> "$HOME/.bashrc"

echo "==> Summary"
python - <<'PY'
import torch, torch_geometric, os, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda ok:", torch.cuda.is_available(), "build:", torch.version.cuda)
print("pyg:", torch_geometric.__version__)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("W&B defaults ->", os.getenv("WANDB_ENTITY"), os.getenv("WANDB_PROJECT"))
PY

echo
echo "All set. Next:"
echo "  1) source ~/.bashrc"
echo "  2) cd \"$PROJECT_DIR\" && conda activate ./.conda"
echo "  3) wandb login   # paste your own API key"
echo
EOF

chmod +x ~/bootstrap_gnn_vm.sh
bash ~/bootstrap_gnn_vm.sh
```

When it finishes:

```bash
source ~/.bashrc
cd ~/workspace/GNN-Pretraining
conda activate ./.conda
wandb login   # paste your personal API key from https://wandb.ai/authorize
```

Quick verify:

```bash
python - <<'PY'
import torch, torch_geometric
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print(torch_geometric.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

---

## Join the Same W&B Project

The bootstrap script sets:

```bash
export WANDB_ENTITY=benc6116-tel-aviv-university
export WANDB_PROJECT=gnn-pretraining
```

Verify:

```bash
echo $WANDB_ENTITY
echo $WANDB_PROJECT
```

Every teammate’s runs & artifacts will appear at:

```
https://wandb.ai/<entity>/<project>
```

Log a test run:

```python
import os, wandb
run = wandb.init(project=os.getenv("WANDB_PROJECT"),
                 entity=os.getenv("WANDB_ENTITY"),
                 config={"smoke_test": True})
run.log({"hello": 1})
run.finish()
```

Artifacts (share checkpoints):

```python
# Save
art = wandb.Artifact("gnn-ckpt", type="checkpoint")
art.add_file("checkpoints/epoch_10.pt")
wandb.log_artifact(art)

# Restore (any teammate, any VM)
art = wandb.use_artifact(f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}/gnn-ckpt:latest", type="checkpoint")
path = art.download()
print("Downloaded to:", path)
```

---

## Day-to-day Workflow

Code lives in GitHub (single source of truth).

Runs / metrics / checkpoints live in W&B.

Stop your VM when idle: Console → VM instances → Stop.

Each repo keeps its own env at `./.conda` (avoids cross-project conflicts).

---

## Troubleshooting

Permission denied (publickey) when SSH’ing from your laptop:

Ensure your public key is in the VM:

```bash
# In the VM (web SSH), paste your public key into:
echo 'ssh-ed25519 AAAA...YOUR_PUBLIC_KEY...' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

Ensure your local `~/.ssh/config` entry has the correct External IP and User.

PyTorch/CUDA mismatch

We pin Torch `2.6.0 + cu124` to match driver CUDA 12.4 on the DLVM image.

Out of GPU memory

Reduce batch size / model size, or stop other GPU processes:

```bash
nvidia-smi
```

---

## Appendix: Optional .bashrc Enhancements

Append to `~/.bashrc` on the VM if you like these QoL tweaks (safe defaults):

```bash
# Handy aliases
alias ll='ls -lF --color=auto'
alias la='ls -aF --color=auto'
alias lt='ls -tlaF --color=auto'
alias grep='grep --color=auto'
alias reload='source ~/.bashrc'

# Colorful prompt with git branch
setPS1() {
  local BLUE="\[\033[1;34m\]" GREEN="\[\033[1;32m\]" WHITE="\[\033[1;37m\]" RESET="\[\033[0m\]"
  local BRANCH=""
  if git rev-parse --is-inside-work-tree &>/dev/null; then
    BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null)"
    [[ -n "$BRANCH" ]] && BRANCH="${WHITE}(${BRANCH})${RESET}"
  fi
  export PS1="${BLUE}\u@\h ${GREEN}\W ${BRANCH}${WHITE}$ ${RESET}"
}
PROMPT_COMMAND=setPS1
```
# GNN Project — Google Cloud VM Setup & Team Workflow (Markdown Guide)

This guide shows **each teammate** how to create a personal Google Cloud VM with an **NVIDIA L4** GPU, set up a **project-local Conda environment**, and use **Weights & Biases (W&B)** to share runs and checkpoints. It also covers creating SSH keys (for GitHub and for VM login) and connecting over SSH.

> **Costs**: You pay while a VM is **running** (GPU + CPU + boot disk). **Stop** the VM when you’re not training to avoid GPU/CPU charges. Stopped VMs still incur disk costs.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Create SSH Keys (Local) & Add to GitHub](#create-ssh-keys-local--add-to-github)
3. [Weights & Biases Account & API Key](#weights--biases-account--api-key)
4. [GCP Quotas & Region](#gcp-quotas--region)
5. [Create the VM (NVIDIA L4, us-east1)](#create-the-vm-nvidia-l4-us-east1)
6. [First Login & GPU Driver Check](#first-login--gpu-driver-check)
7. [Add Your SSH Key to the VM & Local SSH Config](#add-your-ssh-key-to-the-vm--local-ssh-config)
8. [One-shot Environment Bootstrap Script](#one-shot-environment-bootstrap-script)
9. [Join the Same W&B Project](#join-the-same-wb-project)
10. [Day-to-day Workflow](#day-to-day-workflow)
11. [Troubleshooting](#troubleshooting)
12. [Appendix: Optional .bashrc Enhancements](#appendix-optional-bashrc-enhancements)

---

## Prerequisites

- A personal **Google Cloud** account (each teammate uses their own $300 free tier, if available).
- Project Git repo: `https://github.com/alonbebchuk/GNN-Pretraining.git`.
- Comfortable with a terminal (macOS/Linux) or **PowerShell** (Windows).

---

## Create SSH Keys (Local) & Add to GitHub

You’ll use a single SSH key pair for:
- **GitHub** (to push/pull without passwords), and
- **VM login** (public key goes in the VM’s `~/.ssh/authorized_keys`).

### 1) Generate a key (if you don’t already have one)

**macOS/Linux:**
```bash
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
Windows (PowerShell):

powershell
Copy code
if (!(Test-Path "$HOME\.ssh\id_ed25519.pub")) { ssh-keygen -t ed25519 }
Press Enter to accept defaults. It creates:

Private key: ~/.ssh/id_ed25519

Public key: ~/.ssh/id_ed25519.pub

2) Add the public key to GitHub
Copy your public key to clipboard:

macOS: pbcopy < ~/.ssh/id_ed25519.pub

Linux: xclip -sel clip < ~/.ssh/id_ed25519.pub (or open the file and copy manually)

Windows: Get-Content "$HOME\.ssh\id_ed25519.pub" | Set-Clipboard

Go to GitHub → Settings → SSH and GPG keys → New SSH key.
Paste the key, give it a title (e.g., “Laptop-ed25519”), and save.

Keep your private key secret. Never upload id_ed25519 anywhere.

Weights & Biases Account & API Key
Create an account at https://wandb.ai and sign in.

Find your API Key at https://wandb.ai/authorize (Profile → Settings → API keys).
You’ll need to paste this once on your VM with wandb login.

We’ll use the shared project:

Entity: benc6116-tel-aviv-university

Project: gnn-pretraining

(You can change these strings if your team decides on different names.)

GCP Quotas & Region
In IAM & Admin → Quotas, ensure you have:

Compute Engine API → GPUs (all regions) ≥ 1

Compute Engine API → NVIDIA L4 GPUs ≥ 1 in your chosen region (we use us-east1).

Region choice: us-east1 is typically cheaper. Latency to Israel is fine for SSH/editing; training runs on the box.

Create the VM (NVIDIA L4, us-east1)
Console → Compute Engine → VM instances → Create instance

Basics

Name: gnn-l4-01 (or your own)

Region: us-east1 (zone e.g., us-east1-c)

Machine family: GPU

Series: G2

Machine type: g2-standard-8 (8 vCPU, 32 GB RAM)

GPU

Type: NVIDIA L4

Count: 1

Boot disk

Image: Deep Learning VM for PyTorch 2.4 with CUDA 12.4

Size: 150 GB (adjust as needed)

Networking / Firewall

Leave defaults. No need for HTTP/HTTPS rules for SSH-only usage.

Click Create.

Billing tip: You pay while the VM is running. Stop the VM when idle to avoid GPU/CPU costs.

First Login & GPU Driver Check
Open the VM row → click SSH (in the GCP Console). The DLVM image typically offers driver installation automatically on first boot.

Verify:

bash
Copy code
nvidia-smi
You should see NVIDIA L4 and CUDA 12.4.

Add Your SSH Key to the VM & Local SSH Config
1) Append your public key to the VM’s authorized_keys
In the VM (web SSH terminal):

bash
Copy code
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA...YOUR_PUBLIC_KEY...' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
Replace the string with the contents of your local ~/.ssh/id_ed25519.pub.

2) Add a host alias on your local machine
Edit ~/.ssh/config (create if absent) and add:

sshconfig
Copy code
Host gnn-l4-01
  HostName <VM-External-IP>   # shown in the VM list
  User <your-linux-username-on-vm>  # e.g., your VM login name
  IdentityFile ~/.ssh/id_ed25519
  ForwardAgent yes
  ServerAliveInterval 30
  ServerAliveCountMax 3
Now you can connect directly:

bash
Copy code
ssh gnn-l4-01
One-shot Environment Bootstrap Script
Run this on the VM (web SSH or your own SSH). It installs Miniconda, clones the repo, creates a project-local env ./.conda, installs Torch 2.6.0 (CUDA 12.4), PyG wheels, common deps, and sets W&B defaults.

bash
Copy code
cat > ~/bootstrap_gnn_vm.sh <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

# ----------------- editable variables -----------------
REPO_URL="https://github.com/alonbebchuk/GNN-Pretraining.git"
WORKSPACE="$HOME/workspace"
PROJECT_DIR="$WORKSPACE/GNN-Pretraining"
PYTHON_VER="3.10"

# Set this to the SAME entity & project for all teammates:
WANDB_ENTITY="benc6116-tel-aviv-university"
WANDB_PROJECT="gnn-pretraining"
# ------------------------------------------------------

echo "==> Basic tools"
sudo apt-get update -y
sudo apt-get install -y git curl build-essential tmux htop

echo "==> Ensure NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [[ -x /opt/deeplearning/install-driver.sh ]]; then
    sudo /opt/deeplearning/install-driver.sh
  fi
fi
nvidia-smi || true

echo "==> Miniconda"
if [[ ! -d "$HOME/miniconda" ]]; then
  curl -fsSLo "$HOME/miniconda.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda"
fi
eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
conda config --set always_yes true

echo "==> Accept Anaconda ToS (non-interactive)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "==> Clone repo"
mkdir -p "$WORKSPACE"
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

echo "==> Create & activate project-local conda env ($PROJECT_DIR/.conda)"
if [[ ! -d "$PROJECT_DIR/.conda" ]]; then
  conda create -p ./.conda "python=$PYTHON_VER"
fi
conda activate "$PROJECT_DIR/.conda"

echo "==> Upgrade pip"
pip install --upgrade pip

echo "==> PyTorch 2.6.0 (CUDA 12.4)"
pip install --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

echo "==> PyTorch Geometric (match torch 2.6.0 + cu124)"
pip install -U pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install torch-geometric

echo "==> Project utilities"
pip install wandb scikit-learn pyyaml tqdm networkx pandas matplotlib ogb

echo "==> Persist W&B defaults"
grep -q "WANDB_PROJECT=" "$HOME/.bashrc" || echo "export WANDB_PROJECT=$WANDB_PROJECT" >> "$HOME/.bashrc"
grep -q "WANDB_ENTITY="  "$HOME/.bashrc" || echo "export WANDB_ENTITY=$WANDB_ENTITY"  >> "$HOME/.bashrc"

echo "==> Summary"
python - <<'PY'
import torch, torch_geometric, os, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda ok:", torch.cuda.is_available(), "build:", torch.version.cuda)
print("pyg:", torch_geometric.__version__)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("W&B defaults ->", os.getenv("WANDB_ENTITY"), os.getenv("WANDB_PROJECT"))
PY

echo
echo "All set. Next:"
echo "  1) source ~/.bashrc"
echo "  2) cd \"$PROJECT_DIR\" && conda activate ./.conda"
echo "  3) wandb login   # paste your own API key"
echo
EOF

chmod +x ~/bootstrap_gnn_vm.sh
bash ~/bootstrap_gnn_vm.sh
When it finishes:

bash
Copy code
source ~/.bashrc
cd ~/workspace/GNN-Pretraining
conda activate ./.conda
wandb login   # paste your personal API key from https://wandb.ai/authorize
Quick verify:

bash
Copy code
python - <<'PY'
import torch, torch_geometric
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print(torch_geometric.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
Join the Same W&B Project
The bootstrap script sets:

bash
Copy code
export WANDB_ENTITY=benc6116-tel-aviv-university
export WANDB_PROJECT=gnn-pretraining
Verify:

bash
Copy code
echo $WANDB_ENTITY
echo $WANDB_PROJECT
Every teammate’s runs & artifacts will appear at:

php-template
Copy code
https://wandb.ai/<entity>/<project>
Log a test run:

python
Copy code
import os, wandb
run = wandb.init(project=os.getenv("WANDB_PROJECT"),
                 entity=os.getenv("WANDB_ENTITY"),
                 config={"smoke_test": True})
run.log({"hello": 1})
run.finish()
Artifacts (share checkpoints):

python
Copy code
# Save
art = wandb.Artifact("gnn-ckpt", type="checkpoint")
art.add_file("checkpoints/epoch_10.pt")
wandb.log_artifact(art)

# Restore (any teammate, any VM)
art = wandb.use_artifact(f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}/gnn-ckpt:latest", type="checkpoint")
path = art.download()
print("Downloaded to:", path)
Day-to-day Workflow
Code lives in GitHub (single source of truth).

Runs / metrics / checkpoints live in W&B.

Stop your VM when idle: Console → VM instances → Stop.

Each repo keeps its own env at ./.conda (avoids cross-project conflicts).

Troubleshooting
Permission denied (publickey) when SSH’ing from your laptop:

Ensure your public key is in the VM:

bash
Copy code
# In the VM (web SSH), paste your public key into:
echo 'ssh-ed25519 AAAA...YOUR_PUBLIC_KEY...' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
Ensure your local ~/.ssh/config entry has the correct External IP and User.

PyTorch/CUDA mismatch

We pin Torch 2.6.0 + cu124 to match driver CUDA 12.4 on the DLVM image.

Out of GPU memory

Reduce batch size / model size, or stop other GPU processes:

bash
Copy code
nvidia-smi
Appendix: Optional .bashrc Enhancements
Append to ~/.bashrc on the VM if you like these QoL tweaks (safe defaults):

bash
Copy code
# Handy aliases
alias ll='ls -lF --color=auto'
alias la='ls -aF --color=auto'
alias lt='ls -tlaF --color=auto'
alias grep='grep --color=auto'
alias reload='source ~/.bashrc'

# Colorful prompt with git branch
setPS1() {
  local BLUE="\[\033[1;34m\]" GREEN="\[\033[1;32m\]" WHITE="\[\033[1;37m\]" RESET="\[\033[0m\]"
  local BRANCH=""
  if git rev-parse --is-inside-work-tree &>/dev/null; then
    BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null)"
    [[ -n "$BRANCH" ]] && BRANCH="${WHITE}(${BRANCH})${RESET}"
  fi
  export PS1="${BLUE}\u@\h ${GREEN}\W ${BRANCH}${WHITE}$ ${RESET}"
}
PROMPT_COMMAND=setPS1
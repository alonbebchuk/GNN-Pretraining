@echo off

REM Set your wandb API key here (get it from https://wandb.ai/settings)
set WANDB_API_KEY=your_api_key_here

REM Optional: Uncomment for offline mode during development/debugging
REM set WANDB_MODE=offline

REM Prepare data (runs once)
python -m src.data.data_setup

REM Define seeds and configs
set SEEDS=42 84 126
set CONFIGS=^
  configs\pretrain\b2_nfm.yaml ^
  configs\pretrain\b3_nc.yaml ^
  configs\pretrain\b4_single_domain_all.yaml ^
  configs\pretrain\s1_multi_task_generative.yaml ^
  configs\pretrain\s2_multi_task_contrastive.yaml ^
  configs\pretrain\s3_all_self_supervised.yaml ^
  configs\pretrain\s4_all_objectives.yaml ^
  configs\pretrain\s5_all_objectives_da.yaml

REM Launch all (config, seed) runs in parallel
for %%C in (%CONFIGS%) do (
  for %%S in (%SEEDS%) do (
    echo Starting pretrain: config=%%C seed=%%S
    start "pretrain %%~nC s%%S" /b cmd /c python -m src.pretraining.main_pretrain --config %%C --seed %%S
  )
)

echo All pretraining jobs started in background.
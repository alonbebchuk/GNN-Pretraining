@echo off

REM Set your wandb API key here (get it from https://wandb.ai/settings)
set WANDB_API_KEY=your_api_key_here

REM Optional: Uncomment for offline mode during development/debugging
REM set WANDB_MODE=offline

REM Prepare data (runs once)
echo [SETUP] Preparing data...
python -m src.data.data_setup

REM ============================================================================
REM PHASE 1: SYSTEMATIC PRE-TRAINING WITH SWEEPS
REM ============================================================================

echo [PRETRAIN] Starting systematic pre-training sweep...

REM Create and start the systematic pre-training sweep
wandb sweep configs\sweeps\systematic_pretrain.yaml > sweep_id.tmp
set /p SWEEP_ID=<sweep_id.tmp
del sweep_id.tmp

echo Sweep created with ID: %SWEEP_ID%
echo Starting sweep agents...

REM Launch multiple agents to run experiments in parallel
REM Adjust the number based on your hardware (e.g., 4 agents for 4 GPUs)
set NUM_AGENTS=4

for /l %%i in (1,1,%NUM_AGENTS%) do (
    echo Starting agent %%i...
    start "Sweep Agent %%i" cmd /c wandb agent %SWEEP_ID%
)

echo.
echo [PRETRAIN] All sweep agents started!
echo Monitor progress at: https://wandb.ai/your-username/gnn-pretraining/sweeps/%SWEEP_ID%
echo.
echo Waiting for all pre-training to complete...
echo Press Ctrl+C when all agents have finished, then continue with downstream evaluation.
pause

REM ============================================================================
REM PHASE 2: DOWNSTREAM EVALUATION (Future - after pre-training completes)
REM ============================================================================

echo [DOWNSTREAM] Starting downstream evaluation...
REM TODO: Add downstream fine-tuning pipeline here
REM This would iterate through saved checkpoints and evaluate on downstream tasks

REM ============================================================================
REM PHASE 3: ANALYSIS AND REPORTING
REM ============================================================================

echo [ANALYSIS] Generating analysis reports...

REM Create comprehensive analysis report
python wandb_analysis\generate_reports.py --sweep-id %SWEEP_ID%

echo.
echo [COMPLETE] Pipeline finished!
echo Check your WandB project for results and reports.
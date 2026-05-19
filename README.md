# motion-agent-grpo

An experimental repository for GRPO-based reinforcement learning on top of a motion generation codebase.

## What This Repo Contains

This repository now keeps the GRPO-oriented workflow and the model/runtime pieces needed to train, score, generate, and inspect text-to-motion samples. Older upstream demo, SFT, and InterHuman-only entrypoints have been removed to keep the workspace focused.

- GRPO training and reward shaping
- motion generation scripts for local checkpoints
- reusable motion constraint executors for signal/count/temporal rewards
- evaluation, generation, visualization, and monitoring helpers

This public repository intentionally excludes large local assets such as datasets, checkpoints, generated samples, logs, and virtual environments.

## Project Status

This is a research and experimentation repository rather than a polished package. The GRPO workflow and detector-based reward system are the main active direction; legacy upstream files that were not used by this workflow have been pruned.

## Repository Layout

```text
assets/               Example assets and previews
dataset/              Dataset loader used by GRPO training/evaluation
models/               Motion and language model components
options/              GRPO/model option definitions
prepare/              Helper scripts for downloading dependencies
utils/                Shared evaluation and motion utilities

train_grpo.py         GRPO training entrypoint
eval_grpo.py          GRPO-focused evaluation script
generate_motion_simple.py
generate_multiple.py
visualize_motion.py
monitor_training.py
grpo_reward.py
motion_constraint_executor.py
spatiotemporal_reward.py
```

## Setup

### 1. Create an environment

```bash
conda create -n motionagent python=3.10
conda activate motionagent
pip install -r requirements.txt
```

### 2. Download required assets

The repo does not include pretrained weights or dataset payloads. Download them separately as needed.

Example helper scripts included in this repo:

```bash
bash prepare/download_ckpt.sh
bash prepare/download_glove.sh
bash prepare/download_extractor.sh
```

If you use a gated or external LLM backbone, make sure you have access to the model and have authenticated locally.

## Quick Start

### Run GRPO training

```bash
python train_grpo.py
```

### Run evaluation

```bash
python eval_grpo.py
```

### Generate a motion sample

```bash
python generate_motion_simple.py --help
```

## Data Policy

This GitHub repository does not track:

- raw datasets or dataset archives
- generated motions and experiment outputs
- pretrained checkpoints and model weights
- local logs and scratch artifacts
- virtual environments and machine-specific files

If you want to reproduce experiments, place those assets locally in the expected folders after cloning.

## Notes For Public Use

- Some scripts assume local directory layouts developed during experimentation.
- Several utilities are intended for internal research workflows and may require path or checkpoint adjustments.
- If you plan to make this repo broadly reusable, a next good step would be centralizing configuration and documenting expected input formats per script.

## Contribution Scope

To avoid confusion:

- the GRPO reinforcement learning direction and related experimentation in this repository are my own work
- much of the surrounding motion-generation infrastructure was cloned from an existing repository and then modified locally
- this repository should be read as an experimental adaptation rather than a fully from-scratch implementation

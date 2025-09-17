---
title: "MultiGrid Multi-Agent PPO"
excerpt: "Lightweight multi-agent PPO experiments on Gym MultiGrid with shaped rewards, sliding-window checkpoints, and automated trajectory visualizations<br/><img src='/images/multigrid-ppo-demo.gif'>"
collection: portfolio
---

## Overview
This research sandbox explores cooperative reinforcement learning policies for the MultiGrid suite of partially observable grid worlds. The current v8 controller trains three agents with a minimalist PyTorch PPO implementation, emphasizing reproducible reward shaping and fast iteration while keeping compatibility with the original `gym-multigrid` API.

## My Contributions
- Rewrote the PPO training loop (`v8_robust_ppo.py`) with orthogonal weight init, shared convolutional encoders, and a lightweight actor-critic head tailored to MultiGrid observations.
- Designed heuristic reward shaping that combines goal contact bonuses, distance-to-target deltas, idleness penalties, and action incentives to stabilize sparse reward environments.
- Built sliding-window evaluation that snapshots the best checkpoint, emits JSON summaries, and can publish metrics to Weights & Biases when available.
- Authored trajectory analysis utilities (`generate_trajectory_video.py`) that overlay per-agent partial views, cumulative reward traces, and action annotations onto exported videos.
- Packaged reproducible configs (`config/default.yaml`) and environment wrappers under `envs/gym_multigrid` so experiments can be rerun without external dependencies.

## Highlights
- **Reward shaping engine** - Distance-based shaping with movement rewards and stationary penalties tracked per agent for finer credit assignment.
- **Training logistics** - Deterministic seeding, GAE advantage estimation, generalized clipping, and checkpoint rotation every 1k episodes for long horizons.
- **Visualization pipeline** - Frame generator stitches together global maps, per-agent observations, reward curves, and textual action callouts before passing to FFmpeg.


## Technical Stack
`Python`, `PyTorch`, `NumPy`, `Matplotlib`, `Seaborn`

## Learn More
- GitHub repository: [cxh42/multigrid_RL](https://github.com/cxh42/multigrid_RL)
- Sample trajectory script: `python generate_trajectory_video.py --model-path models8/best_performance --env MultiGrid-Cluttered-Fixed-15x15`
- Training entry point: `python v8_robust_ppo.py --episodes 100000 --env MultiGrid-Cluttered-Fixed-15x15`

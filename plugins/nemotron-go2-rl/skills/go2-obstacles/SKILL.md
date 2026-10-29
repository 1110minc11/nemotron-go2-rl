---
name: go2-obstacles
description: Stage 2 of nemotron-go2-rl — Nemotron designs obstacle scenarios (single block, scatter, slalom, gate, wall doorway, corridor, side clutter), a validator gates them, and Isaac Lab PPO fine-tunes the goal-reaching + LiDAR-avoidance policy on a flat plane. Use when the user wants to add obstacles, generate or revise obstacle scenarios, train, evaluate or record obstacle avoidance for the Go2.
---

# go2-obstacles — obstacle scenarios designed by Nemotron, learned by PPO

Run from the nemotron-go2-rl repo root. Isaac Sim commands need the Isaac Lab Python environment with
`unitree_rl_lab` installed (the tasks reuse its `Unitree-Go2-Unified-Navigation` MDP and a 51 -> 2 actor).

## Pipeline

```text
Nemotron (JSON schema output) -> validator (+ repair loop, guard) -> configs/obstacles/<name>.json
   -> train_avoid.py (512 Go2, frozen 50 Hz stage-1 walking policy, warm start from model_247)
   -> evaluate_avoid.py (same layouts for every policy) -> record_avoid.py -> report_avoid.py
```

## Scenarios

Families: `open`, `single_block`, `scatter` (1-5 objects), `slalom`, `gate` (two boxes), `wall_gap` (doorway in a
wall), `corridor` (two walls the robot must walk through), `side_clutter` (never blocking). Each scenario gives ranges for
goal distance and bearing, where obstacles sit, lateral offset, count and gap, plus difficulty and weight. The layout
sampler (`nemotron_go2_rl/obstacles/layout.py`) turns a scenario into 8 obstacle slots per environment (3 pillars,
3 boxes, 2 wall pieces) with rejection sampling: start and goal stay >= 0.9 m from every obstacle, openings stay >= 0.9 m.

## Commands

```bash
# 1. scenarios (Ollama by default; ~10-15 min for the 30B model; do not run while Isaac Sim trains)
python scripts/generate_obstacle_cases.py --name go2_obstacles_v2 --out configs/obstacles/go2_obstacles_v2.json

# 2. train (350 iterations, ~40 min on an RTX 3090); the control uses the original single-box distribution
python scripts/train_avoid.py --headless --catalog configs/obstacles/go2_obstacles_v1.json \
    --resume_path <unitree_rl_lab>/logs/rsl_rl/unitree_go2_unified_navigation/<run>/model_247.pt \
    --max_iterations 350 --run_name nemotron_obstacles_v1
python scripts/train_avoid.py --headless --task Nemotron-Go2-Obstacles-Control --resume_path ... --run_name control_v1

# 3. evaluate every checkpoint on the Nemotron scenarios (32 environments per scenario, seeded)
python scripts/evaluate_avoid.py --headless --policy "label=<ckpt>" --policy "label2=<ckpt>" --out outputs/eval/stage2.json

# 4. video and report
python scripts/record_avoid.py --headless --policy "label=<ckpt>" --episodes 16 --out outputs/videos/avoid.mp4
python scripts/report_avoid.py --eval outputs/eval/stage2.json --run "label=<run dir>" --out docs/results_stage2
```

## Rules

- The actor sees only the goal (distance, bearing), a 36-ray front 180 degree LiDAR (4 m) and body state. Doorway positions are
  privileged shaping for the critic and the progress reward only.
- Collision = root point closer than 0.30 m to an obstacle surface, or any body part other than a foot touching (>= 3 N).
- Report measured numbers. If the starting policy already solves most scenarios, say so; compare against the pretrained
  `model_247` zero-shot and against the single-box control, not only against zero.

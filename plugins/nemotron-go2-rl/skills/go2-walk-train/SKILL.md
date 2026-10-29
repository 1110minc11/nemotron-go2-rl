---
name: go2-walk-train
description: Train, evaluate, record and report Go2 walking policies on a Nemotron case catalog with Isaac Lab + RSL-RL PPO in the nemotron-go2-rl repo. Use for training runs, the uniform-command ablation, per-case evaluation against a baseline checkpoint, case-tour videos and the results report.
---

# go2-walk-train — Isaac Lab PPO on the cases

Run from the nemotron-go2-rl repo root inside the Isaac Lab Python environment. Append the local
Isaac Sim kit args if the setup needs them (e.g. `--kit_args="--portable-root <workspace>/outputs/kit"`).

## Tasks

| Task id | Purpose |
|---|---|
| `Nemotron-Go2-Walk-Flat` | training: 4096 Go2s, weighted case sampling, difficulty curriculum (level 2 at iter 150, level 3 at iter 400) |
| `Nemotron-Go2-Walk-Flat-Uniform` | ablation: same everything, commands uniform over the envelope |
| `Nemotron-Go2-Walk-Flat-Eval` | env `i` runs case `i % N`, fixed command, nominal physics, no noise |
| `Nemotron-Go2-Walk-Flat-Record` | one Go2, scripted commands, following camera |

## Train (run in the background, ~1 s per iteration on an RTX 3090)

```bash
python scripts/train.py --headless --catalog configs/cases/go2_walk_flat_v1.json \
    --max_iterations 1500 --run_name nemotron_cases_v1
# ablation
python scripts/train.py --headless --task Nemotron-Go2-Walk-Flat-Uniform \
    --max_iterations 1500 --run_name uniform_v1
```

Logs: `logs/rsl_rl/nemotron_go2_walk_flat/<timestamp>_<run_name>/` with `params/cases.json`
(the exact catalog used), `exported/policy.pt|onnx`, TensorBoard events.

## Evaluate (same seed and commands for every policy)

```bash
python scripts/evaluate.py --headless \
    --policy ours=logs/rsl_rl/nemotron_go2_walk_flat/<run>/model_1499.pt \
    --policy uniform=logs/rsl_rl/nemotron_go2_walk_flat/<uniform run>/model_1499.pt \
    --policy baseline=<previous policy checkpoint> \
    --out outputs/eval/stage1.json
```

Success per env: no fall, and on every axis the 9 s average velocity is within 20 % of the command
(at least 0.05 m/s for vx, vy and 0.1 rad/s for wz), measured after a 2 s warm-up.

## Record and report

```bash
python scripts/record.py --headless --policy "Nemotron cases (ours)=<ckpt>" --out outputs/videos/ours.mp4
python scripts/record.py --headless --policy "baseline=<ckpt>" --out outputs/videos/baseline.mp4
ffmpeg -i outputs/videos/baseline.mp4 -i outputs/videos/ours.mp4 \
    -filter_complex "[0:v]scale=960:-2[a];[1:v]scale=960:-2[b];[a][b]hstack" -c:v libx264 -crf 23 \
    outputs/videos/compare.mp4
python scripts/report.py --eval outputs/eval/stage1.json --runs logs/rsl_rl/nemotron_go2_walk_flat/<run> \
    --out docs/results
```

Report what was measured. If a policy falls or misses a case, say which case and by how much.

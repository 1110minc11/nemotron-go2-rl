---
name: go2-walk
description: Run or resume the stage-1 Go2 walking pipeline of the nemotron-go2-rl repo end to end — Nemotron generates the walking cases, the validator gates them, Isaac Lab PPO trains on them in an empty flat world, then per-case evaluation, a video tour and a written report. Use when the user asks to set up, train, re-run, compare or report Go2 walking with Nemotron cases.
---

# go2-walk — stage-1 pipeline orchestrator

Work from the **nemotron-go2-rl repo root** (the directory with `nemotron_go2_rl/`, `scripts/`, `configs/`).
All Isaac Sim commands need the Python environment that has Isaac Lab 2.3 and `unitree_rl_lab` installed
(e.g. `conda activate isaac-sim`). Case generation needs only the standard library and an
OpenAI-compatible Nemotron endpoint.

## Pipeline

```text
configs/prompts/baseline_findings.txt ─┐
                                       ▼
 1. Nemotron (Ollama / NIM / build.nvidia.com)  ── /go2-walk-cases
        │  JSON-schema structured output
        ▼
 2. Validator  (clamp to Go2 envelope, category checks, coverage, repair loop, guard)
        │  configs/cases/<name>.json
        ▼
 3. Isaac Lab PPO, 4096 Go2s on an empty plane  ── /go2-walk-train
        │  logs/rsl_rl/nemotron_go2_walk_flat/<run>/model_*.pt
        ▼
 4. Per-case evaluation (same seed for every checkpoint)
 5. Video tour of every case + side-by-side comparison
 6. Report: docs/ + outputs/
```

## Steps

1. **Check the machine** before anything heavy: `nvidia-smi` (free VRAM), `ollama list` (Nemotron
   pulled?), and that no other Isaac Sim process is running. Never run Nemotron inference and
   4096-env training on the same 24 GB GPU at the same time: the generator unloads the Ollama
   model when it finishes, keep it that way.
2. **Cases**: follow the `go2-walk-cases` skill. Show the user the case table and the validator's
   auto-fixes before training. A catalog that needed `guard` cases deserves a second look.
3. **Train**: follow the `go2-walk-train` skill. Run training in the background and report
   progress from the RSL-RL log (`Mean reward`, `Mean episode length`, `curriculum_level`).
4. **Evaluate** every candidate on the same catalog and seed. Include the previous lab policy as a
   baseline when one exists; say plainly which cases the baseline was never trained on.
5. **Record** `scripts/record.py` for each policy, then compose a side-by-side with `ffmpeg hstack`.
6. **Report**: `scripts/report.py` writes charts and a markdown table; update `docs/` with the
   numbers that were actually measured. Do not round failures into successes.
7. **Feedback (optional)**: hand the evaluation back to Nemotron so it revises the catalog
   (`generate_cases.py --previous <catalog> --feedback-eval <eval.json> --feedback-policy <label>`),
   fine-tune from the last checkpoint with `--resume_path` and the hydra override
   `env.commands.base_velocity.level_up_steps=[0,0]`, and evaluate on the *original* cases.
   Keep the previous policy unless the new one is better on that same test set.

## Rules

- Commands sent to the simulator are always clamped to `ENVELOPE` in
  `nemotron_go2_rl/cases/schema.py`. Widen the envelope only on explicit user request.
- Keep the 45-dim actor observation and 12-dim action layout unchanged so policies stay
  compatible with the `unitree_rl_lab` deploy and sim2sim tooling.
- Never overwrite a committed catalog silently: write a new `--name` / `--out` and diff it.
- Add Isaac Sim kit args from the project docs (e.g. a `--portable-root` inside the workspace) when
  the local setup requires them.

## Related plugin

NVIDIA's `nemotron-customize` plugin (`/plugin install nemotron-customize@nvidia-nemotron`) builds
Nemotron SFT / RL / SDG pipelines. It is the route for a later stage that fine-tunes a dedicated
"case designer" Nemotron on accepted catalogs; it is not needed to run stage 1.

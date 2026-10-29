---
name: go2-walk-cases
description: Generate, repair and review the Go2 walking-case catalog with NVIDIA Nemotron (local Ollama or hosted endpoint) in the nemotron-go2-rl repo. Use when the user wants new or different walking cases, wants to change the prompt / known weaknesses, or asks why a case was clamped or dropped.
---

# go2-walk-cases — Nemotron writes the walking cases

Run from the nemotron-go2-rl repo root.

## Generate

Local Nemotron through Ollama (default endpoint `http://localhost:11434/v1`):

```bash
ollama pull nemotron-3.5-lightning:30b      # once; nemotron-3-nano:4b for a quick, small run
python scripts/generate_cases.py --name go2_walk_flat_v2 \
    --out configs/cases/go2_walk_flat_v2.json
```

Hosted endpoint (pass the *name* of the key variable, never the key itself):

```bash
python scripts/generate_cases.py --base-url https://integrate.api.nvidia.com/v1 \
    --model <model id on that endpoint> --api-key-env NVIDIA_API_KEY --name ... --out ...
```

Useful flags: `--num-cases` (10–24), `--max-attempts` (repair rounds), `--temperature`, `--seed`,
`--findings <file>` (known weaknesses appended to the prompt; `''` to skip), `--keep-loaded`.

Revise a catalog from an evaluation (closes the Nemotron ↔ RL loop):

```bash
python scripts/generate_cases.py --name go2_walk_flat_v2 --out configs/cases/go2_walk_flat_v2.json \
    --previous configs/cases/go2_walk_flat_v1.json \
    --feedback-eval outputs/eval/stage1.json --feedback-policy "<policy label in that JSON>"
```

Cases below 90 % success become findings (with commanded vs measured velocity); the previous
catalog is included so Nemotron edits it instead of starting over.

## What the validator does (`nemotron_go2_rl/cases/schema.py`)

- clamps every range to the Go2 envelope (vx ±1.0 m/s, vy ±0.6 m/s, wz ±1.5 rad/s, hold 1–10 s)
  and the robustness envelope (friction 0.3–1.5, added mass −1…+3 kg, push ≤ 1 m/s);
- checks each case against its category (e.g. `lateral` must have |vy| > 0.12, |vx| ≤ 0.12 and
  |wz| ≤ 0.2; forward/backward keep vy and wz near zero) and requires a slow (≤ 0.3 m/s) case
  for forward, backward and lateral;
- checks names against the body frame (+y = left, +wz = counter-clockwise), so a case called
  "left" really moves left;
- requires `stand, forward, backward, lateral, turn_in_place, arc`, a difficulty-1 moving case,
  10–24 cases, and at least 5 % standing weight;
- requires left/right symmetry (every lateral or turning case has a mirror case), each axis to
  reach 60 % of the envelope in both directions, and pushes of at least 0.2 m/s;
- sends unfixable problems back to Nemotron (repair prompt); after the last attempt, drops cases
  that contradict their category or name, and inserts `guard_*` cases (mirrors, fast-axis cases,
  templates for missing categories).

Every automatic fix is listed under `validation` in the catalog JSON and printed by the script.

## Review checklist (show this to the user)

1. The case table printed at the end (id, category, difficulty, weight, ranges, hold).
2. `validation.auto_fixes` and `validation.guard` — clamps are normal, guards mean Nemotron
   missed something.
3. Weight balance: difficulty-1 cases should carry enough weight to bootstrap a gait.
4. The `robustness` block and its rationale.

Then run `python -m pytest tests` (no Isaac Sim needed) to confirm the catalog loads.

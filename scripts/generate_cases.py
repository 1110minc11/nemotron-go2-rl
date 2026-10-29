"""Ask Nemotron for the walking cases of a flat empty world and write a validated catalog.

Runs in any Python >= 3.10 (standard library only). Examples:

    # local Nemotron through Ollama (default)
    python scripts/generate_cases.py

    # close the loop: revise a catalog using the per-case evaluation of the policy trained on it
    python scripts/generate_cases.py --name go2_walk_flat_v2 --out configs/cases/go2_walk_flat_v2.json \
        --previous configs/cases/go2_walk_flat_v1.json \
        --feedback-eval outputs/eval/stage1.json --feedback-policy nemotron_cases

    # NVIDIA API catalog
    python scripts/generate_cases.py --base-url https://integrate.api.nvidia.com/v1 \
        --model nvidia/nemotron-3.5-lightning-30b-a3b --api-key-env NVIDIA_API_KEY
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nemotron_go2_rl import DEFAULT_CATALOG, REPO_ROOT  # noqa: E402
from nemotron_go2_rl.cases.client import NemotronClient  # noqa: E402
from nemotron_go2_rl.cases.feedback import findings_from_eval  # noqa: E402
from nemotron_go2_rl.cases.generate import generate_catalog  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI-compatible endpoint")
    p.add_argument("--model", default="nemotron-3.5-lightning:30b")
    p.add_argument("--api-key-env", default=None, help="env var holding the API key (hosted endpoints)")
    p.add_argument("--name", default="go2_walk_flat_v1")
    p.add_argument("--out", default=DEFAULT_CATALOG)
    p.add_argument("--num-cases", type=int, default=16)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--findings",
        default=os.path.join(REPO_ROOT, "configs", "prompts", "baseline_findings.txt"),
        help="text file, one known weakness of the previous policy per line ('' to skip)",
    )
    p.add_argument("--keep-loaded", action="store_true", help="do not unload the Ollama model afterwards")
    p.add_argument("--previous", default=None, help="catalog to revise (closes the loop with --feedback-eval)")
    p.add_argument("--feedback-eval", default=None, help="evaluate.py JSON whose failures become findings")
    p.add_argument("--feedback-policy", default=None, help="policy label inside --feedback-eval")
    args = p.parse_args()

    findings = []
    if args.findings:
        with open(args.findings, encoding="utf-8") as f:
            findings = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    previous = None
    if args.previous:
        with open(args.previous, encoding="utf-8") as f:
            previous = json.load(f)
    if args.feedback_eval:
        if not args.feedback_policy:
            p.error("--feedback-eval needs --feedback-policy")
        findings += findings_from_eval(args.feedback_eval, args.feedback_policy)
        print("[feedback]\n  - " + "\n  - ".join(findings))

    client = NemotronClient(args.base_url, args.model, api_key_env=args.api_key_env)
    print(f"[nemotron] {args.model} @ {args.base_url}")
    try:
        catalog = generate_catalog(
            client,
            name=args.name,
            num_cases=args.num_cases,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            seed=args.seed,
            findings=findings,
            previous=previous,
        )
    finally:
        if not args.keep_loaded:
            client.unload()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\n[catalog] {args.out}")
    print(f"{'id':28s} {'cat':14s} lvl  weight  vx            vy            wz            hold")
    for c in catalog["cases"]:
        print(
            f"{c['id'][:28]:28s} {c['category']:14s} {c['difficulty']:>3d}  {c['weight']:.3f}  "
            f"{str(c['lin_vel_x']):13s} {str(c['lin_vel_y']):13s} {str(c['ang_vel_z']):13s} {c['hold_time_s']}"
        )
    print(f"robustness: {json.dumps({k: v for k, v in catalog['robustness'].items() if k != 'rationale'})}")
    v = catalog["validation"]
    print(f"auto-fixes: {len(v['auto_fixes'])}, guard: {len(v['guard'])}")


if __name__ == "__main__":
    main()

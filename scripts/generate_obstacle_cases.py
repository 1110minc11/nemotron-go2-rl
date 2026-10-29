"""Ask Nemotron for obstacle scenarios (stage 2) and write a validated catalog.

    python scripts/generate_obstacle_cases.py            # local Ollama, nemotron-3.5-lightning:30b
    python scripts/generate_obstacle_cases.py --base-url https://integrate.api.nvidia.com/v1 \
        --model <model id> --api-key-env NVIDIA_API_KEY
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nemotron_go2_rl import REPO_ROOT  # noqa: E402
from nemotron_go2_rl.cases.client import NemotronClient  # noqa: E402
from nemotron_go2_rl.obstacles.generate import generate_catalog  # noqa: E402

DEFAULT_OUT = os.path.join(REPO_ROOT, "configs", "obstacles", "go2_obstacles_v1.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="nemotron-3.5-lightning:30b")
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--name", default="go2_obstacles_v1")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--num-cases", type=int, default=16)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--findings", default=os.path.join(REPO_ROOT, "configs", "prompts", "obstacle_findings.txt"))
    p.add_argument("--keep-loaded", action="store_true")
    args = p.parse_args()
    findings = []
    if args.findings:
        with open(args.findings, encoding="utf-8") as f:
            findings = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    client = NemotronClient(args.base_url, args.model, api_key_env=args.api_key_env)
    print(f"[nemotron] {args.model} @ {args.base_url}")
    try:
        catalog = generate_catalog(client, args.name, args.num_cases, args.max_attempts, args.temperature, args.seed, findings)
    finally:
        if not args.keep_loaded:
            client.unload()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\n[catalog] {os.path.relpath(args.out, REPO_ROOT)}")
    print(f"{'id':26s} {'family':13s} lvl weight dist        bearing        along       lateral     count gap")
    for c in catalog["cases"]:
        print(f"{c['id'][:26]:26s} {c['family']:13s} {c['difficulty']:>3d} {c['weight']:.3f}  {str(c['goal_distance']):11s} "
              f"{str(c['goal_bearing_deg']):14s} {str(c['along']):11s} {str(c['lateral']):11s} {str(c['count']):6s}{c['gap']}")
    v = catalog["validation"]
    print(f"auto-fixes: {len(v['auto_fixes'])}, guard: {len(v['guard'])}")


if __name__ == "__main__":
    main()

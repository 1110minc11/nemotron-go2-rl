"""Evaluate goal-reaching policies on every Nemotron obstacle scenario (same layouts for every policy).

Environment ``i`` runs scenario ``i % C``; layouts are drawn from a seeded generator, so every policy sees exactly
the same obstacles and goals. Success = goal reached within 0.4 m before any termination.

    python scripts/evaluate_avoid.py --headless --catalog configs/obstacles/go2_obstacles_v1.json \
        --policy "ours=<run>/model_349.pt" --policy "control=<run>/model_349.pt" --out outputs/eval/stage2.json
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", action="append", required=True, help="label=path/to/model.pt (repeatable)")
parser.add_argument("--catalog", default=None)
parser.add_argument("--locomotion", default=None)
parser.add_argument("--per_case", type=int, default=32, help="environments per scenario")
parser.add_argument("--max_steps", type=int, default=150, help="5 Hz steps per episode (30 s)")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--out", default=os.path.join(REPO, "outputs", "eval", "stage2.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.catalog:
    os.environ["NEMOTRON_GO2_OBSTACLES"] = os.path.abspath(args_cli.catalog)
if args_cli.locomotion:
    os.environ["NEMOTRON_GO2_LOCOMOTION"] = os.path.abspath(args_cli.locomotion)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import nemotron_go2_rl.tasks  # noqa: E402, F401
from nemotron_go2_rl.obstacles.schema import load_catalog  # noqa: E402
from nemotron_go2_rl.tasks.field_env_cfg import resolve_catalog_path  # noqa: E402

TASK = "Nemotron-Go2-Obstacles-Eval"
FAIL_TERMS = ("obstacle_collision", "non_foot_contact", "base_contact", "bad_orientation")


@torch.inference_mode()
def run(env, policy, term, seed):
    base = env.unwrapped
    term._gen.manual_seed(seed)
    base.seed(seed)
    obs, _ = env.reset()
    n, dev = env.num_envs, base.device
    case = term.case_index.clone()
    done_ever = torch.zeros(n, dtype=torch.bool, device=dev)
    outcome = torch.full((n,), -1, dtype=torch.long, device=dev)  # 0 success, 1..4 failure kinds, 5 timeout
    steps = torch.zeros(n, device=dev)
    min_clear = torch.full((n,), 99.0, device=dev)
    path = torch.zeros(n, device=dev)
    prev = base.scene["robot"].data.root_pos_w[:, :2].clone()
    tm = base.termination_manager
    for step in range(args_cli.max_steps):
        active = ~done_ever
        d, _ = term.nearest()
        min_clear = torch.where(active, torch.minimum(min_clear, d), min_clear)
        obs, _, dones, _ = env.step(policy(obs))
        now = base.scene["robot"].data.root_pos_w[:, :2]
        path += torch.where(active, (now - prev).norm(dim=1), torch.zeros_like(path))
        prev = now.clone()
        steps += active.float()
        fin = active & dones.bool()
        reached = tm.get_term("goal_reached")
        for k, name in enumerate(FAIL_TERMS, start=1):
            hit = fin & tm.get_term(name) & ~reached
            outcome = torch.where(hit & (outcome < 0), torch.full_like(outcome, k), outcome)
        outcome = torch.where(fin & reached & (outcome < 0), torch.zeros_like(outcome), outcome)
        outcome = torch.where(fin & (outcome < 0), torch.full_like(outcome, 5), outcome)
        done_ever |= fin
        if done_ever.all():
            break
    outcome = torch.where(outcome < 0, torch.full_like(outcome, 5), outcome)
    rows = []
    for k, cid in enumerate(term.case_names):
        m = case == k
        o = outcome[m]
        succ = o == 0
        rows.append({
            "case": cid, "envs": int(m.sum()),
            "success_rate": float(succ.float().mean()),
            "collision_rate": float(((o >= 1) & (o <= 2)).float().mean()),
            "fall_rate": float(((o >= 3) & (o <= 4)).float().mean()),
            "timeout_rate": float((o == 5).float().mean()),
            "mean_steps_success": float(steps[m][succ].mean()) if succ.any() else None,
            "min_clearance_m": float(min_clear[m].mean()),
        })
    return rows


def main():
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    catalog = load_catalog(resolve_catalog_path())
    n_cases = len(catalog["cases"])
    cfg.scene.num_envs = args_cli.per_case * n_cases
    cfg.seed = args_cli.seed
    cfg.commands.goal_pose.seed = args_cli.seed
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = cfg.sim.device
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    term = env.unwrapped.command_manager.get_term("goal_pose")
    weights = {c["id"]: c["weight"] for c in catalog["cases"]}
    results = {"catalog": resolve_catalog_path(), "seed": args_cli.seed, "num_envs": env.num_envs,
               "locomotion": cfg.actions.navigation_velocity.policy_path, "max_steps": args_cli.max_steps, "policies": {}}
    for spec in args_cli.policy:
        label, path = spec.rsplit("=", 1)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(os.path.abspath(path), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        rows = run(env, policy, term, args_cli.seed)
        w = sum(weights[r["case"]] for r in rows)
        overall = {k: sum(r[k] for r in rows) / len(rows) for k in ("success_rate", "collision_rate", "fall_rate", "timeout_rate")}
        overall["success_rate_weighted"] = sum(r["success_rate"] * weights[r["case"]] for r in rows) / w
        results["policies"][label] = {"checkpoint": os.path.abspath(path), "overall": overall, "cases": rows}
        print(f"\n=== {label}: {path}")
        print(f"{'case':26s} {'succ':>5s} {'coll':>5s} {'fall':>5s} {'tout':>5s} {'min d':>6s}")
        for r in rows:
            print(f"{r['case'][:26]:26s} {r['success_rate']:5.2f} {r['collision_rate']:5.2f} {r['fall_rate']:5.2f} "
                  f"{r['timeout_rate']:5.2f} {r['min_clearance_m']:6.2f}")
        print("overall:", json.dumps({k: round(v, 3) for k, v in overall.items()}))
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

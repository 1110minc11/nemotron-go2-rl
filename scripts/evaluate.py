"""Evaluate one or more walking checkpoints on every case of a catalog.

Environment ``i`` runs case ``i % N`` with one fixed command drawn inside the case's
ranges, nominal friction/mass, no pushes and no observation noise. The same seed gives
every checkpoint exactly the same commands.

    python scripts/evaluate.py --headless \
        --policy ours=logs/rsl_rl/nemotron_go2_walk_flat/<run>/model_1499.pt \
        --policy baseline=<unitree_rl_lab>/logs/rsl_rl/unitree_go2_velocity/<run>/model_4099.pt \
        --out outputs/eval/stage1.json
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.stdout.reconfigure(line_buffering=True)  # Isaac Sim exits without flushing stdout

from isaaclab.app import AppLauncher  # noqa: E402

from nemotron_go2_rl import DEFAULT_CATALOG  # noqa: E402

parser = argparse.ArgumentParser(description="Per-case evaluation of Go2 walking policies.")
parser.add_argument("--policy", action="append", required=True, help="label=path/to/model.pt (repeatable)")
parser.add_argument("--catalog", default=DEFAULT_CATALOG)
parser.add_argument("--num_envs", type=int, default=None, help="default: 32 per case")
parser.add_argument("--warmup_steps", type=int, default=100, help="steps excluded from the metrics (2 s)")
parser.add_argument("--measure_steps", type=int, default=450, help="measured steps (9 s at 50 Hz)")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--out", default=os.path.join(REPO, "outputs", "eval", "eval.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
os.environ["NEMOTRON_GO2_CATALOG"] = os.path.abspath(args_cli.catalog)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import nemotron_go2_rl.tasks  # noqa: E402, F401
from nemotron_go2_rl.tasks.runtime import case_command_term, load_policy  # noqa: E402
from nemotron_go2_rl.tasks.walk_env_cfg import apply_catalog  # noqa: E402

TASK = "Nemotron-Go2-Walk-Flat-Eval"


def success_mask(cmd, mean_v, alive):
    """Per axis, the 9 s average velocity must be within 20 % of the command
    (at least 0.05 m/s for vx, vy and 0.1 rad/s for wz), and the robot must not fall."""
    floor = torch.tensor([0.05, 0.05, 0.1], device=cmd.device)
    tol = torch.maximum(0.2 * cmd.abs(), floor)
    return alive & ((mean_v - cmd).abs() <= tol).all(dim=1)


@torch.inference_mode()
def run_policy(env, policy, seed):
    env.unwrapped.seed(seed)
    obs, _ = env.reset()
    term = case_command_term(env)
    robot = env.unwrapped.scene["robot"]
    n = env.num_envs
    dev = env.unwrapped.device
    cmd0 = term.command.clone()
    case0 = term.case_index.clone()
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    sum_xy = torch.zeros(n, device=dev)
    sum_yaw = torch.zeros(n, device=dev)
    sum_v = torch.zeros(n, 3, device=dev)
    count = 0
    for step in range(args_cli.warmup_steps + args_cli.measure_steps):
        obs, _, dones, extras = env.step(policy(obs))
        fell = dones.bool() & ~extras["time_outs"].bool()
        alive &= ~fell
        if step >= args_cli.warmup_steps:
            v = torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=1)
            sum_xy += torch.norm(cmd0[:, :2] - v[:, :2], dim=1)
            sum_yaw += torch.abs(cmd0[:, 2] - v[:, 2])
            sum_v += v
            count += 1
    err_xy, err_yaw, mean_v = sum_xy / count, sum_yaw / count, sum_v / count
    ok = success_mask(cmd0, mean_v, alive)

    rows = []
    for k, cid in enumerate(term.case_names):
        m = case0 == k
        if not m.any():
            continue
        rows.append(
            {
                "case": cid,
                "envs": int(m.sum()),
                "success_rate": float(ok[m].float().mean()),
                "fall_rate": float((~alive[m]).float().mean()),
                "err_xy": float(err_xy[m & alive].mean()) if (m & alive).any() else None,
                "err_yaw": float(err_yaw[m & alive].mean()) if (m & alive).any() else None,
                "cmd_mean": cmd0[m].mean(0).tolist(),
                "vel_mean": mean_v[m & alive].mean(0).tolist() if (m & alive).any() else None,
            }
        )
    return rows


def main():
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    catalog = apply_catalog(env_cfg, args_cli.catalog, randomize=False)
    n_cases = len(catalog["cases"])
    env_cfg.scene.num_envs = args_cli.num_envs or 32 * n_cases
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    episode_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    assert args_cli.warmup_steps + args_cli.measure_steps < episode_steps, "evaluation must end before time-out"

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = env_cfg.sim.device
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)

    weights = {c["id"]: c["weight"] for c in catalog["cases"]}
    results = {"catalog": os.path.abspath(args_cli.catalog), "seed": args_cli.seed, "num_envs": env.num_envs,
               "warmup_steps": args_cli.warmup_steps, "measure_steps": args_cli.measure_steps, "policies": {}}
    for spec in args_cli.policy:
        label, path = spec.rsplit("=", 1)
        policy = load_policy(env, agent_cfg, os.path.abspath(path))
        rows = run_policy(env, policy, args_cli.seed)
        overall = {
            "success_rate": sum(r["success_rate"] for r in rows) / len(rows),
            "success_rate_weighted": sum(r["success_rate"] * weights[r["case"]] for r in rows)
            / sum(weights[r["case"]] for r in rows),
            "fall_rate": sum(r["fall_rate"] for r in rows) / len(rows),
        }
        results["policies"][label] = {"checkpoint": os.path.abspath(path), "overall": overall, "cases": rows}
        print(f"\n=== {label}: {path}")
        print(f"{'case':30s} {'succ':>6s} {'fall':>6s} {'err_xy':>7s} {'err_yaw':>7s}  cmd -> measured")
        for r in rows:
            fmt = lambda x: f"{x:7.3f}" if x is not None else "    n/a"  # noqa: E731
            cmd = ", ".join(f"{x:+.2f}" for x in r["cmd_mean"])
            vel = ", ".join(f"{x:+.2f}" for x in r["vel_mean"]) if r["vel_mean"] else "-"
            print(
                f"{r['case'][:30]:30s} {r['success_rate']:6.2f} {r['fall_rate']:6.2f} {fmt(r['err_xy'])} "
                f"{fmt(r['err_yaw'])}  [{cmd}] -> [{vel}]"
            )
        print(f"overall: {json.dumps({k: round(v, 3) for k, v in overall.items()})}")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

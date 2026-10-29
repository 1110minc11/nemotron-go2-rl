"""Train a Go2 flat-walking policy with RSL-RL PPO on a Nemotron walking-case catalog.

    python scripts/train.py --headless --catalog configs/cases/go2_walk_flat_v1.json \
        --max_iterations 1500 --run_name nemotron_cases_v1
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.stdout.reconfigure(line_buffering=True)  # Isaac Sim exits without flushing stdout

from isaaclab.app import AppLauncher  # noqa: E402

from nemotron_go2_rl import DEFAULT_CATALOG  # noqa: E402

parser = argparse.ArgumentParser(description="Train Go2 walking on Nemotron cases.")
parser.add_argument("--task", default="Nemotron-Go2-Walk-Flat")
parser.add_argument("--catalog", default=DEFAULT_CATALOG, help="walking-case catalog JSON")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--run_name", default=None)
parser.add_argument("--resume_path", default=None, help="checkpoint to warm-start from")
parser.add_argument("--video", action="store_true", help="record short clips during training")
parser.add_argument("--video_interval", type=int, default=24 * 250)
parser.add_argument("--video_length", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
os.environ["NEMOTRON_GO2_CATALOG"] = os.path.abspath(args_cli.catalog)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs inside Isaac Sim."""

import gymnasium as gym  # noqa: E402
import shutil  # noqa: E402
import torch  # noqa: E402
from datetime import datetime  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg  # noqa: E402

import nemotron_go2_rl.tasks  # noqa: E402, F401
from nemotron_go2_rl.tasks.walk_env_cfg import apply_catalog  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    catalog = apply_catalog(env_cfg, args_cli.catalog, randomize=True)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name, log_dir)
    print(f"[INFO] log dir: {log_dir}")
    print(f"[INFO] catalog: {args_cli.catalog} ({len(catalog['cases'])} cases, model {catalog['generator']['model']})")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.resume_path:
        print(f"[INFO] warm start from {args_cli.resume_path}")
        runner.load(args_cli.resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    shutil.copy(args_cli.catalog, os.path.join(log_dir, "params", "cases.json"))
    export_deploy_cfg(env.unwrapped, log_dir)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_dir = os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
    print(f"[INFO] exported TorchScript/ONNX to {export_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

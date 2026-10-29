"""Train goal reaching + obstacle avoidance on a Nemotron obstacle catalog (stage 2).

    python scripts/train_avoid.py --headless --catalog configs/obstacles/go2_obstacles_v1.json \
        --resume_path <unitree_rl_lab>/logs/rsl_rl/unitree_go2_unified_navigation/<run>/model_247.pt \
        --max_iterations 350 --run_name nemotron_obstacles_v1
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.stdout.reconfigure(line_buffering=True)  # Isaac Sim exits without flushing stdout

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train Go2 obstacle avoidance on Nemotron scenarios.")
parser.add_argument("--task", default="Nemotron-Go2-Obstacles")
parser.add_argument("--catalog", default=None, help="obstacle catalog JSON (default: configs/obstacles/go2_obstacles_v1.json)")
parser.add_argument("--locomotion", default=None, help="frozen 50 Hz TorchScript policy (default: stage-1 walking policy)")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--run_name", default=None)
parser.add_argument("--resume_path", default=None, help="checkpoint to warm-start from (51 -> 2 actor)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.catalog:
    os.environ["NEMOTRON_GO2_OBSTACLES"] = os.path.abspath(args_cli.catalog)
if args_cli.locomotion:
    os.environ["NEMOTRON_GO2_LOCOMOTION"] = os.path.abspath(args_cli.locomotion)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import shutil  # noqa: E402
import torch  # noqa: E402
from datetime import datetime  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import nemotron_go2_rl.tasks  # noqa: E402, F401
from nemotron_go2_rl.tasks.field_env_cfg import resolve_catalog_path  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed
    env_cfg.commands.goal_pose.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name, log_dir)
    catalog_path = env_cfg.commands.goal_pose.catalog_path
    print(f"[INFO] log dir: {log_dir}\n[INFO] catalog: {catalog_path}")
    print(f"[INFO] frozen locomotion: {env_cfg.actions.navigation_velocity.policy_path}")

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.resume_path:
        print(f"[INFO] warm start from {args_cli.resume_path}")
        runner.load(os.path.abspath(args_cli.resume_path), load_optimizer=False)
        runner.current_learning_iteration = 0  # count iterations of this run only (checkpoints start at model_0)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    shutil.copy(catalog_path, os.path.join(log_dir, "params", "obstacles.json"))

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

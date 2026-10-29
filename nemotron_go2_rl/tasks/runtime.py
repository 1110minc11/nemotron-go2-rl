"""Helpers shared by the evaluation and recording scripts (import after the app is launched)."""

from __future__ import annotations

import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def load_policy(env: RslRlVecEnvWrapper, agent_cfg, checkpoint: str):
    """Return an inference callable ``obs -> actions`` for an RSL-RL checkpoint."""
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=env.unwrapped.device)


def case_command_term(env):
    return env.unwrapped.command_manager.get_term("base_velocity")


def showcase_commands(case: dict, high: float = 0.85, low: float = 0.15) -> list[torch.Tensor]:
    """One or two representative commands for a case.

    A second, opposite command is added only when a range spans zero by a visible amount
    (e.g. ``omni`` with vy in [-0.5, 0.5]), not for small jitter like vy in [-0.05, 0.05].
    """
    axes = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    lo = torch.tensor([case[a][0] for a in axes])
    hi = torch.tensor([case[a][1] for a in axes])
    visible = torch.tensor([0.12, 0.12, 0.2])
    first = lo + high * (hi - lo)
    cmds = [first]
    if bool(((lo < -visible) & (hi > visible)).any()):
        cmds.append(lo + low * (hi - lo))
    return cmds

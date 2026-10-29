"""Velocity command term that samples from a Nemotron walking-case catalog."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from nemotron_go2_rl.cases import ENVELOPE, load_catalog

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class CaseVelocityCommand(UniformVelocityCommand):
    """Body-frame ``[vx, vy, wz]`` command drawn from a walking-case catalog.

    On every resample an environment picks one case (by catalog weight, among the cases
    unlocked by the difficulty curriculum), draws a command uniformly inside that case's
    ranges and holds it for a duration drawn from the case's ``hold_time_s``.

    Assignment modes:
        ``weighted``: training. Weighted sampling with a difficulty curriculum.
        ``round_robin``: evaluation. Environment ``i`` always runs case ``i % num_cases``.
        ``external``: scripted playback. The command is set with :meth:`set_command`.
        ``uniform``: ablation without cases. Uniform over the whole envelope, with
            ``rel_standing_envs`` standing; the catalog is ignored for sampling.
    """

    cfg: CaseVelocityCommandCfg

    def __init__(self, cfg: CaseVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        catalog = load_catalog(cfg.catalog_path)
        cases = catalog["cases"]
        self.catalog = catalog
        self.case_names = [c["id"] for c in cases]
        axes = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        self._lo = torch.tensor([[c[a][0] for a in axes] for c in cases], device=self.device)
        self._hi = torch.tensor([[c[a][1] for a in axes] for c in cases], device=self.device)
        self._hold = torch.tensor([c["hold_time_s"] for c in cases], device=self.device)
        self._weight = torch.tensor([c["weight"] for c in cases], device=self.device)
        self._difficulty = torch.tensor([c["difficulty"] for c in cases], device=self.device)
        self._env_lo = torch.tensor([ENVELOPE[a][0] for a in axes], device=self.device)
        self._env_hi = torch.tensor([ENVELOPE[a][1] for a in axes], device=self.device)
        self.case_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.level = 1
        self.metrics["curriculum_level"] = torch.ones(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"CaseVelocityCommand: {len(self.case_names)} cases from {self.cfg.catalog_path}\n"
            f"\tassignment: {self.cfg.assignment}, level-up steps: {self.cfg.level_up_steps}"
        )

    @property
    def num_cases(self) -> int:
        return len(self.case_names)

    def set_command(self, command: torch.Tensor, case_index: int | torch.Tensor | None = None, env_ids=None):
        """Overwrite the command (``external`` mode playback and tests)."""
        env_ids = slice(None) if env_ids is None else env_ids
        self.vel_command_b[env_ids] = command.to(self.device)
        if case_index is not None:
            self.case_index[env_ids] = case_index

    def _current_level(self) -> int:
        step = self._env.common_step_counter
        return 1 + sum(int(step >= s) for s in self.cfg.level_up_steps)

    def _update_metrics(self):
        steps = max(self._env.max_episode_length, 1)
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / steps
        )
        self.metrics["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / steps
        self.metrics["curriculum_level"][:] = float(self.level)

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        if n == 0:
            return
        if self.cfg.assignment == "external":
            self.time_left[ids] = 1.0e9
            return
        if self.cfg.assignment == "uniform":
            u = torch.rand(n, 3, device=self.device)
            self.vel_command_b[ids] = self._env_lo + (self._env_hi - self._env_lo) * u
            hold_lo, hold_hi = ENVELOPE["hold_time_s"]
            self.time_left[ids] = hold_lo + (hold_hi - hold_lo) * torch.rand(n, device=self.device)
            self.is_standing_env[ids] = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
            return

        if self.cfg.assignment == "round_robin":
            idx = ids % self.num_cases
        else:
            self.level = self._current_level()
            probs = self._weight * (self._difficulty <= self.level).float()
            idx = torch.multinomial(probs, n, replacement=True)
        self.case_index[ids] = idx

        u = torch.rand(n, 3, device=self.device)
        self.vel_command_b[ids] = self._lo[idx] + (self._hi[idx] - self._lo[idx]) * u
        if self.cfg.hold_forever:
            self.time_left[ids] = 1.0e9
        else:
            hold = self._hold[idx]
            self.time_left[ids] = hold[:, 0] + (hold[:, 1] - hold[:, 0]) * torch.rand(n, device=self.device)
        # standing is expressed by "stand" cases, not by a random standing fraction
        self.is_standing_env[ids] = False


@configclass
class CaseVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = CaseVelocityCommand

    catalog_path: str = MISSING
    """Catalog JSON written by ``scripts/generate_cases.py``."""

    assignment: str = "weighted"
    """``weighted`` | ``round_robin`` | ``external`` | ``uniform``."""

    level_up_steps: tuple[int, ...] = (150 * 24, 400 * 24)
    """Env steps (``common_step_counter``) at which difficulty 2 and 3 are unlocked."""

    hold_forever: bool = False
    """Never resample within an episode (evaluation)."""

    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (10.0, 10.0)  # unused: hold times come from the catalog
    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=ENVELOPE["lin_vel_x"], lin_vel_y=ENVELOPE["lin_vel_y"], ang_vel_z=ENVELOPE["ang_vel_z"]
    )

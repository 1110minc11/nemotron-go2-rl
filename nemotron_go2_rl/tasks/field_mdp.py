"""MDP terms for goal reaching among Nemotron-designed obstacle scenarios (import after the app is launched)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from unitree_rl_lab.tasks.locomotion.mdp.navigation import GoalCommand, GoalCommandCfg

from nemotron_go2_rl.obstacles.geometry import INF, point_distances, ray_ranges
from nemotron_go2_rl.obstacles.layout import NUM_SLOTS, SLOT_HEIGHT, CaseTable, goal_world, sample_layouts
from nemotron_go2_rl.obstacles.schema import load_catalog

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

LIDAR_RAYS = 36
LIDAR_FOV_DEG = 180.0
LIDAR_MAX_RANGE = 4.0
PARK_Z = -5.0  # inactive obstacle slots wait below the floor


def _yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    q = quat_wxyz
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


class FieldGoalCommand(GoalCommand):
    """Samples a Nemotron scenario, a goal, and places the obstacle slots for each reset environment.

    Assignment modes: ``weighted`` (training, difficulty curriculum), ``round_robin`` (env ``i`` always
    runs scenario ``i % C``; evaluation), ``cycle`` (every reset advances to the next scenario; videos).
    """

    cfg: FieldGoalCommandCfg

    def __init__(self, cfg: FieldGoalCommandCfg, env):
        super().__init__(cfg, env)
        self.catalog = load_catalog(cfg.catalog_path)
        self.table = CaseTable(self.catalog["cases"], device=self.device)
        self.case_names = self.table.ids
        self._slots = [env.scene[f"slot{i}"] for i in range(NUM_SLOTS)]
        n, s = self.num_envs, NUM_SLOTS
        self.slot_xy_w = torch.zeros(n, s, 2, device=self.device)
        self.slot_yaw = torch.zeros(n, s, device=self.device)
        self.active = torch.zeros(n, s, dtype=torch.bool, device=self.device)
        self.slot_half = self.table.slot_half
        self.case_index = torch.zeros(n, dtype=torch.long, device=self.device)
        self.start_w = torch.zeros(n, 2, device=self.device)
        self.path_dir = torch.zeros(n, 2, device=self.device)
        self.guide_w = torch.zeros(n, 2, device=self.device)
        self.guide_on = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.guide_s = torch.zeros(n, device=self.device)
        self.episodes = torch.zeros(n, dtype=torch.long, device=self.device)
        self.level = 1
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(cfg.seed))
        self.metrics["layout_failed"] = torch.zeros(n, device=self.device)
        self.metrics["curriculum_level"] = torch.ones(n, device=self.device)
        self.metrics["active_obstacles"] = torch.zeros(n, device=self.device)

    def _current_level(self) -> int:
        step = self._env.common_step_counter
        return 1 + sum(int(step >= s) for s in self.cfg.level_up_steps)

    def _choose_cases(self, env_ids: torch.Tensor) -> torch.Tensor:
        count = len(self.table)
        if self.cfg.assignment == "round_robin":
            return (env_ids + self.cfg.case_offset) % count
        if self.cfg.assignment == "cycle":
            idx = (self.episodes[env_ids] + env_ids + self.cfg.case_offset) % count
            self.episodes[env_ids] += 1
            return idx
        self.level = self._current_level()
        probs = self.table.weight * (self.table.difficulty <= self.level).float()
        return torch.multinomial(probs, len(env_ids), replacement=True, generator=self._gen)

    def _resample_command(self, env_ids: torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids)
        idx = self._choose_cases(env_ids)
        lay = sample_layouts(self.table, idx, self._gen)
        start = self._robot.data.root_pos_w[env_ids, :2]
        y0 = _yaw(self._robot.data.root_quat_w[env_ids])
        c, s = torch.cos(y0), torch.sin(y0)

        def rot(v):  # (n, ..., 2) rotate by the robot heading
            shape = (n,) + (1,) * (v.dim() - 2)
            cc, ss = c.reshape(shape), s.reshape(shape)
            return torch.stack((cc * v[..., 0] - ss * v[..., 1], ss * v[..., 0] + cc * v[..., 1]), dim=-1)

        self.case_index[env_ids] = idx
        self.start_w[env_ids] = start
        self.goal_pos_w[env_ids] = start + rot(goal_world(lay))
        self.path_dir[env_ids] = rot(torch.stack((torch.cos(lay.bearing), torch.sin(lay.bearing)), dim=-1))
        self.slot_xy_w[env_ids] = start.unsqueeze(1) + rot(lay.slot_xy)
        self.slot_yaw[env_ids] = lay.slot_yaw + y0.unsqueeze(1)
        self.active[env_ids] = lay.active
        self.guide_w[env_ids] = start + rot(lay.guide_xy)
        self.guide_on[env_ids] = lay.guide_on
        self.guide_s[env_ids] = lay.guide_s
        self.metrics["layout_failed"][env_ids] = lay.failed.float()
        self.metrics["active_obstacles"][env_ids] = lay.active.sum(1).float()

        origins = self._env.scene.env_origins[env_ids]
        for i, slot in enumerate(self._slots):
            pose = slot.data.default_root_state[env_ids, :7].clone()
            on = self.active[env_ids, i]
            pose[:, 0] = torch.where(on, self.slot_xy_w[env_ids, i, 0], origins[:, 0])
            pose[:, 1] = torch.where(on, self.slot_xy_w[env_ids, i, 1], origins[:, 1])
            pose[:, 2] = torch.where(on, origins[:, 2] + SLOT_HEIGHT / 2, torch.full_like(origins[:, 2], PARK_Z))
            yaw = self.slot_yaw[env_ids, i]
            pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6] = torch.cos(yaw / 2), 0.0, 0.0, torch.sin(yaw / 2)
            slot.write_root_pose_to_sim(pose, env_ids=env_ids)
            slot.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        self._compute_command()

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["curriculum_level"][:] = float(self.level)

    # -- geometry queries used by observations, rewards and terminations ---------------------------------------

    def lidar(self) -> torch.Tensor:
        """36-ray front LiDAR ranges in metres, (N, 36)."""
        root = self._robot.data.root_pos_w[:, :2]
        yaw = _yaw(self._robot.data.root_quat_w)
        half = math.radians(LIDAR_FOV_DEG) / 2
        ang = torch.linspace(-half, half, LIDAR_RAYS, device=self.device)[None, :] + yaw[:, None]
        dirs = torch.stack((ang.cos(), ang.sin()), dim=-1)
        return ray_ranges(root, dirs, self.slot_xy_w, self.slot_yaw, self.slot_half, self.active, LIDAR_MAX_RANGE)

    def nearest(self):
        """Distance from the robot root to the nearest obstacle surface (N,) and the closest point (N, 2)."""
        dist, closest = point_distances(
            self._robot.data.root_pos_w[:, :2], self.slot_xy_w, self.slot_yaw, self.slot_half, self.active
        )
        d, k = dist.min(dim=1)
        return d, closest[torch.arange(self.num_envs, device=self.device), k]


@configclass
class FieldGoalCommandCfg(GoalCommandCfg):
    class_type: type = FieldGoalCommand
    catalog_path: str = MISSING
    assignment: str = "weighted"
    case_offset: int = 0
    seed: int = 0
    level_up_steps: tuple[int, ...] = (80 * 32, 200 * 32)
    """Environment steps (``common_step_counter``) that unlock difficulty 2 and 3 (32 steps per PPO iteration)."""


def _term(env) -> FieldGoalCommand:
    return env.command_manager.get_term("goal_pose")


def field_lidar(env) -> torch.Tensor:
    return _term(env).lidar() / LIDAR_MAX_RANGE


def field_privileged(env) -> torch.Tensor:
    """Critic-only geometry: vector to the nearest surface, its distance, front clearance, obstacle count."""
    term = _term(env)
    dist, point = term.nearest()
    root = term._robot.data.root_pos_w
    delta = torch.zeros(env.num_envs, 3, device=env.device)
    delta[:, :2] = torch.where((dist < INF / 2).unsqueeze(1), point - root[:, :2], torch.zeros_like(point))
    body = quat_apply_inverse(yaw_quat(term._robot.data.root_quat_w), delta)
    near = torch.where(dist < INF / 2, dist, torch.full_like(dist, LIDAR_MAX_RANGE))
    front = term.lidar()[:, 14:22].amin(dim=1)
    return torch.stack(
        (
            (body[:, 0] / LIDAR_MAX_RANGE).clamp(-1, 1),
            (body[:, 1] / LIDAR_MAX_RANGE).clamp(-1, 1),
            (near / LIDAR_MAX_RANGE).clamp(0, 1),
            front / LIDAR_MAX_RANGE,
            term.active.sum(1).float() / NUM_SLOTS,
        ),
        dim=1,
    )


def field_clearance_penalty(env, safe_distance_m: float) -> torch.Tensor:
    dist, _ = _term(env).nearest()
    return ((safe_distance_m - dist).clamp(min=0.0) / safe_distance_m).square()


def field_collision(env, robot_radius_m: float) -> torch.Tensor:
    dist, _ = _term(env).nearest()
    return dist < robot_radius_m


def guided_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Speed toward the doorway until it is passed (gate / wall_gap), then toward the goal.

    The doorway position is privileged shaping only: the actor never sees it.
    """
    term = _term(env)
    robot = term._robot
    xy = robot.data.root_pos_w[:, :2]
    s = ((xy - term.start_w) * term.path_dir).sum(dim=1)
    guiding = term.guide_on & (s < term.guide_s + 0.35)
    target = torch.where(guiding.unsqueeze(1), term.guide_w, term.goal_pos_w)
    delta = torch.zeros(env.num_envs, 3, device=env.device)
    delta[:, :2] = target - xy
    body = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), delta)[:, :2]
    unit = body / body.norm(dim=1, keepdim=True).clamp(min=1e-3)
    return (robot.data.root_lin_vel_b[:, :2] * unit).sum(dim=1)

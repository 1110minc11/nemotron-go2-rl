"""Go2 walking on an empty flat plane, with commands drawn from a Nemotron case catalog.

Observations (45 actor / 60 critic), actions (12 joint targets), rewards and terminations
are inherited unchanged from ``unitree_rl_lab``'s ``Unitree-Go2-Velocity`` task, so
policies trained here drop into the existing deploy / sim2sim tooling, and existing
checkpoints (e.g. robust ``model_4099``) can be evaluated on the same cases.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg

from nemotron_go2_rl import DEFAULT_CATALOG
from nemotron_go2_rl.cases import load_catalog

from .case_command import CaseVelocityCommandCfg

CATALOG_ENV_VAR = "NEMOTRON_GO2_CATALOG"


def resolve_catalog_path(path: str | None = None) -> str:
    return os.path.abspath(path or os.environ.get(CATALOG_ENV_VAR) or DEFAULT_CATALOG)


def apply_catalog(env_cfg: RobotEnvCfg, path: str | None = None, randomize: bool = True) -> dict:
    """Point the command term at ``path`` and copy the catalog's robustness block into the events."""
    path = resolve_catalog_path(path)
    catalog = load_catalog(path)
    env_cfg.commands.base_velocity.catalog_path = path
    rob = catalog["robustness"]
    ev = env_cfg.events
    if randomize:
        friction = tuple(rob["friction"])
        ev.physics_material.params["static_friction_range"] = friction
        ev.physics_material.params["dynamic_friction_range"] = (friction[0] * 0.9, friction[1] * 0.9)
        ev.add_base_mass.params["mass_distribution_params"] = tuple(rob["added_base_mass_kg"])
        push = rob["push_velocity_mps"]
        ev.push_robot.params["velocity_range"] = {"x": (-push, push), "y": (-push, push)}
        ev.push_robot.interval_range_s = tuple(rob["push_interval_s"])
    else:
        ev.physics_material.params["static_friction_range"] = (1.0, 1.0)
        ev.physics_material.params["dynamic_friction_range"] = (0.9, 0.9)
        ev.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        ev.push_robot = None
    return catalog


@configclass
class Go2WalkFlatEnvCfg(RobotEnvCfg):
    """Training: 4096 Go2s on a flat plane, weighted case sampling + difficulty curriculum."""

    def __post_init__(self):
        super().__post_init__()
        # empty world: an infinite flat plane, nothing else
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None  # replaced by the case difficulty curriculum

        self.commands.base_velocity = CaseVelocityCommandCfg(debug_vis=False, catalog_path=resolve_catalog_path())
        apply_catalog(self, randomize=True)


@configclass
class Go2WalkFlatUniformEnvCfg(Go2WalkFlatEnvCfg):
    """Ablation: same world, robustness and PPO, but commands uniform over the envelope (no cases)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.assignment = "uniform"
        self.commands.base_velocity.rel_standing_envs = 0.1


@configclass
class Go2WalkFlatEvalEnvCfg(Go2WalkFlatEnvCfg):
    """Evaluation: env ``i`` runs case ``i % N`` with one fixed command, nominal physics, no noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.scene.env_spacing = 3.0
        self.episode_length_s = 12.0
        self.commands.base_velocity.assignment = "round_robin"
        self.commands.base_velocity.hold_forever = True
        self.observations.policy.enable_corruption = False
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)}
        self.events.base_external_force_torque = None
        apply_catalog(self, randomize=False)


@configclass
class Go2WalkFlatRecordEnvCfg(Go2WalkFlatEvalEnvCfg):
    """Video: one Go2, scripted command sequence, camera following the robot."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0
        self.commands.base_velocity.assignment = "external"
        self.commands.base_velocity.debug_vis = True
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (1.8, -2.6, 1.1)
        self.viewer.lookat = (0.0, 0.0, 0.25)
        self.viewer.resolution = (1280, 720)

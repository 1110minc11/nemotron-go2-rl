"""Go2 goal reaching on a flat plane among Nemotron-designed obstacle scenarios.

Observations (policy 51 / critic 56), the 5 Hz [forward, yaw] action and the PPO layout are the same as
unitree_rl_lab's ``Unitree-Go2-Unified-Navigation`` so its model_247 can warm-start training. The frozen 50 Hz
locomotion policy is this repository's stage-1 walking policy by default.
"""

from __future__ import annotations

import hashlib
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.sensor_avoidance_env_cfg import Go2SensorAvoidanceEnvCfg

from nemotron_go2_rl import REPO_ROOT
from nemotron_go2_rl.obstacles.layout import SLOT_HALF, SLOT_HEIGHT, SLOT_KIND, NUM_SLOTS
from nemotron_go2_rl.obstacles.schema import load_catalog

from . import field_mdp as fmdp

CATALOG_ENV_VAR = "NEMOTRON_GO2_OBSTACLES"
LOCOMOTION_ENV_VAR = "NEMOTRON_GO2_LOCOMOTION"
DEFAULT_OBSTACLE_CATALOG = os.path.join(REPO_ROOT, "configs", "obstacles", "go2_obstacles_v1.json")
STAGE1_POLICY = os.path.join(REPO_ROOT, "policies", "stage1_walk", "policy.pt")

COLORS = {"pillar": (0.9, 0.55, 0.1), "box": (0.8, 0.1, 0.1), "wall": (0.25, 0.4, 0.8)}
ROBOT_RADIUS_M = 0.30
SAFE_DISTANCE_M = 0.65


def resolve_catalog_path(path: str | None = None) -> str:
    return os.path.abspath(path or os.environ.get(CATALOG_ENV_VAR) or DEFAULT_OBSTACLE_CATALOG)


def use_locomotion(cfg, path: str | None = None) -> str:
    """Freeze ``path`` (default: the stage-1 walking policy) as the 50 Hz locomotion controller."""
    path = os.path.abspath(path or os.environ.get(LOCOMOTION_ENV_VAR) or STAGE1_POLICY)
    cfg.actions.navigation_velocity.policy_path = path
    with open(path, "rb") as f:
        cfg.actions.navigation_velocity.expected_sha256 = hashlib.sha256(f.read()).hexdigest()
    return path


def _slot(i: int) -> RigidObjectCfg:
    hx, hy = SLOT_HALF[i]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Slot{i}",
        spawn=sim_utils.CuboidCfg(
            size=(2 * hx, 2 * hy, SLOT_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[SLOT_KIND[i]]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, fmdp.PARK_Z)),
    )


@configclass
class Go2ObstacleFieldEnvCfg(Go2SensorAvoidanceEnvCfg):
    """Training: 512 Go2s, each with its own randomly drawn Nemotron scenario."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.obstacle = None
        for i in range(NUM_SLOTS):
            setattr(self.scene, f"slot{i}", _slot(i))
        self.scene.contact_forces.history_length = 40
        self.episode_length_s = 30.0
        self.commands.goal_pose = fmdp.FieldGoalCommandCfg(
            asset_name="robot", resampling_time_range=(1.0e9, 1.0e9), debug_vis=False,
            catalog_path=resolve_catalog_path(),
        )
        for group in (self.observations.policy, self.observations.critic):
            group.lidar.func = fmdp.field_lidar
            group.lidar.params = {}
        self.observations.critic.privileged_obstacle.func = fmdp.field_privileged
        self.observations.critic.privileged_obstacle.params = {}
        self.rewards.move_toward_goal = RewTerm(func=fmdp.guided_progress, weight=3.0)
        self.rewards.heading_error.weight = -0.08
        self.rewards.clearance = RewTerm(func=fmdp.field_clearance_penalty, weight=-6.0,
                                         params={"safe_distance_m": SAFE_DISTANCE_M})
        self.rewards.collision = RewTerm(
            func=mdp.safety_termination_penalty, weight=-80.0,
            params={"term_names": ["obstacle_collision", "non_foot_contact", "base_contact", "bad_orientation"]})
        self.rewards.time.weight = -0.05
        self.terminations.obstacle_collision = DoneTerm(func=fmdp.field_collision, params={"robot_radius_m": ROBOT_RADIUS_M})
        self.terminations.non_foot_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces",
                                                 body_names=["base", "Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
                    "threshold": 3.0})
        self.events.reset_obstacle = None
        use_locomotion(self)


@configclass
class Go2ObstacleFieldUniformEnvCfg(Go2ObstacleFieldEnvCfg):
    """Control: the original single-box distribution (one 0.6 m box on the path 80 % of the time), no Nemotron scenarios."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.goal_pose.catalog_path = os.path.join(REPO_ROOT, "configs", "obstacles", "baseline_single_block.json")


@configclass
class Go2ObstacleFieldEvalEnvCfg(Go2ObstacleFieldEnvCfg):
    """Evaluation: environment ``i`` always runs scenario ``i % C``; no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.scene.env_spacing = 14.0
        self.commands.goal_pose.assignment = "round_robin"
        self.observations.policy.enable_corruption = False


@configclass
class Go2ObstacleFieldRecordEnvCfg(Go2ObstacleFieldEvalEnvCfg):
    """Video: one robot, a new scenario at every reset, top-down camera over the env origin."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.goal_pose.assignment = "cycle"
        self.commands.goal_pose.debug_vis = True
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (0.0, -0.01, 9.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (1280, 720)

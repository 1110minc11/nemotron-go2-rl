"""PPO settings: unitree_rl_lab's Go2 locomotion PPO with a finite budget for flat walking."""

from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class Go2WalkFlatPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "nemotron_go2_walk_flat"
    max_iterations = 1500
    save_interval = 100

"""Gym registration. Import only after the Isaac Sim app is running."""

import gymnasium as gym

_ENV = f"{__name__}.walk_env_cfg"
_PPO = f"{__name__}.agents:Go2WalkFlatPPORunnerCfg"

gym.register(
    id="Nemotron-Go2-Walk-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:Go2WalkFlatEnvCfg",
        "play_env_cfg_entry_point": f"{_ENV}:Go2WalkFlatRecordEnvCfg",
        "rsl_rl_cfg_entry_point": _PPO,
    },
)

gym.register(
    id="Nemotron-Go2-Walk-Flat-Uniform",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_ENV}:Go2WalkFlatUniformEnvCfg", "rsl_rl_cfg_entry_point": _PPO},
)

gym.register(
    id="Nemotron-Go2-Walk-Flat-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_ENV}:Go2WalkFlatEvalEnvCfg", "rsl_rl_cfg_entry_point": _PPO},
)

gym.register(
    id="Nemotron-Go2-Walk-Flat-Record",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_ENV}:Go2WalkFlatRecordEnvCfg", "rsl_rl_cfg_entry_point": _PPO},
)

_FIELD = f"{__name__}.field_env_cfg"
_FIELD_PPO = f"{__name__}.agents:Go2ObstacleFieldPPORunnerCfg"
for _id, _cfg, _play in (
    ("Nemotron-Go2-Obstacles", "Go2ObstacleFieldEnvCfg", "Go2ObstacleFieldRecordEnvCfg"),
    ("Nemotron-Go2-Obstacles-Control", "Go2ObstacleFieldUniformEnvCfg", None),
    ("Nemotron-Go2-Obstacles-Eval", "Go2ObstacleFieldEvalEnvCfg", None),
    ("Nemotron-Go2-Obstacles-Record", "Go2ObstacleFieldRecordEnvCfg", None),
):
    _kwargs = {"env_cfg_entry_point": f"{_FIELD}:{_cfg}", "rsl_rl_cfg_entry_point": _FIELD_PPO}
    if _play:
        _kwargs["play_env_cfg_entry_point"] = f"{_FIELD}:{_play}"
    gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True, kwargs=_kwargs)

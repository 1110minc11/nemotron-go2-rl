"""nemotron-go2-rl: Nemotron-designed walking cases + Isaac Lab PPO for the Unitree Go2.

Importing this package does not import Isaac Sim. Gym tasks are registered by
``import nemotron_go2_rl.tasks`` after the simulation app has been launched.
"""

import os

__version__ = "0.1.0"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CATALOG = os.path.join(REPO_ROOT, "configs", "cases", "go2_walk_flat_v1.json")

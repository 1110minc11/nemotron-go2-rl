"""Prompt asking Nemotron to design obstacle scenarios for goal-reaching on a flat plane."""

from __future__ import annotations

import json

from .schema import ENVELOPE, FAMILIES, FAMILY_HELP, MAX_CASES, MIN_CASES, MIN_DOORWAY, REQUIRED_FAMILIES

SYSTEM_PROMPT = """You are a curriculum designer for reinforcement learning of robot navigation.
You design the obstacle scenarios a quadruped must learn to handle, and you answer with JSON that
matches the given schema exactly. No prose outside the JSON."""


def build_user_prompt(num_cases: int, findings: list[str] | None = None) -> str:
    envelope = {k: list(v) for k, v in ENVELOPE.items()}
    lines = [
        "Robot: Unitree Go2 quadruped, 0.31 m wide, walking at up to 0.4 m/s and turning at up to 0.5 rad/s.",
        "World: a flat floor. Each episode places the robot at the origin facing +x and a goal point at the given",
        "distance and bearing. Obstacles are 0.8 m high boxes; the robot must reach the goal (within 0.4 m)",
        "without touching anything.",
        "Policy input: goal distance and bearing, a 36-ray LiDAR over the front 180 degrees (4 m range), and body",
        "velocities. It cannot see behind itself and has no map. Policy output: forward speed and yaw rate.",
        "",
        "Task: design the scenarios the training curriculum should sample. A scenario is a layout family plus",
        "parameter ranges; every episode draws a random layout from one scenario (random side, random values",
        "inside the ranges). The obstacle pool is: 3 pillars 0.3 m, 3 boxes 0.6 m, 2 wall pieces 0.2 x 2.0 m.",
        f"Return between {MIN_CASES} and {MAX_CASES} scenarios (aim for about {num_cases}).",
        "",
        "Layout families:",
    ]
    lines += [f"- {name}: {FAMILY_HELP[name]}" for name in FAMILIES]
    lines += [
        "",
        "For every scenario give:",
        f"- id (unique snake_case), family (one of {list(FAMILIES)}), title (<= 6 English words), title_ko (Korean, <= 15 chars)",
        "- difficulty: 1 = learned first, 2 = after basics, 3 = hardest",
        "- weight: relative sampling frequency (positive number)",
        "- ranges as [min, max]: goal_distance (m), goal_bearing_deg (goal direction relative to the robot's heading),",
        "  along (fraction of the way to the goal where obstacles sit), lateral (magnitude in m, a random side is",
        "  chosen), count (integer), gap (m). Fields a family does not use must still be present; any valid values.",
        "- rationale: one sentence on what skill the scenario trains",
        "",
        f"Hard envelope (values outside are clipped): {json.dumps(envelope)}",
        "",
        "Requirements the validator enforces:",
        f"- at least one scenario for each of {list(REQUIRED_FAMILIES)}, and difficulty levels 1, 2 and 3 all used;",
        "- difficulty 1 includes an 'open' or 'single_block' scenario;",
        f"- gate and wall_gap openings must be at least {MIN_DOORWAY} m wide, and at least one must be able to be",
        "  1.3 m or narrower (tight doorways);",
        "- at least one corridor whose width can be 1.3 m or less, and at least one scatter whose count can reach 4 or 5;",
        "- side_clutter needs |lateral| >= 1.0 (obstacles must not block the path); slalom needs count >= 2 and",
        "  goal_distance up to at least 3.5; corridor needs goal_distance up to at least 4.0 and |lateral| <= 0.4;",
        "  gate |lateral| <= 0.8; wall_gap |lateral| <= 1.2; side_clutter count <= 3;",
        "- some scenarios must put the goal behind the robot (bearing range reaching beyond +-150 degrees) and some",
        "  must cover the frontal sector (-30 to 30 degrees);",
        "- cover each family at several difficulties where it makes sense: wider doorways and fewer obstacles first,",
        "  tight doorways, several obstacles and goals behind the robot last.",
    ]
    if findings:
        lines += ["", "Known weaknesses of the current policy (give them dedicated scenarios):"]
        lines += [f"- {f}" for f in findings]
    return "\n".join(lines)


def build_repair_prompt(problems: list[str]) -> str:
    return (
        "Your previous catalog failed validation. Fix every problem below and return the complete corrected "
        "JSON (all scenarios), not a diff:\n- " + "\n- ".join(problems)
    )

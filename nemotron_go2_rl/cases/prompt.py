"""Prompt that asks Nemotron to enumerate the walking cases for a flat, empty world."""

from __future__ import annotations

import json

from .schema import (
    CATEGORIES,
    ENVELOPE,
    MAX_CASES,
    MIN_CASES,
    MIN_PUSH_VELOCITY,
    REQUIRED_CATEGORIES,
    ROBUSTNESS_ENVELOPE,
)

SYSTEM_PROMPT = """You are a locomotion curriculum designer for reinforcement learning.
You enumerate the distinct walking situations ("cases") a quadruped controller must master,
and you answer with JSON that matches the given schema exactly. No prose outside the JSON."""


def build_user_prompt(
    num_cases: int, prior_findings: list[str] | None = None, previous_cases: list[dict] | None = None
) -> str:
    findings = prior_findings or []
    envelope = {k: list(v) for k, v in ENVELOPE.items()}
    robustness = {k: list(v) for k, v in ROBUSTNESS_ENVELOPE.items()}
    lines = [
        "Robot: Unitree Go2 quadruped (15 kg, 12 actuated joints, max ~23.7 Nm per joint).",
        "World: a flat, empty ground plane. No obstacles, slopes, stairs or terrain. Only walking matters.",
        "Policy: PPO actor at 50 Hz. Input = IMU angular velocity, projected gravity, joint positions and",
        "velocities, previous action, and a body-frame velocity command [lin_vel_x, lin_vel_y, ang_vel_z].",
        "Output = 12 joint position targets. One command is held for hold_time_s seconds, then resampled.",
        "",
        "Task: list the walking cases a training curriculum should sample velocity commands from.",
        f"Return between {MIN_CASES} and {MAX_CASES} cases (aim for about {num_cases}).",
        "",
        "For every case give:",
        "- id: short snake_case name, unique",
        f"- category: one of {list(CATEGORIES)}",
        "- title: <= 6 English words; title_ko: the same title in natural Korean (<= 15 characters)",
        "- difficulty: 1 = learned first, 2 = after basic gait works, 3 = hardest / most dynamic",
        "- weight: relative sampling frequency (positive number)",
        "- lin_vel_x, lin_vel_y, ang_vel_z: [min, max] command range. A command is sampled uniformly in it.",
        "- hold_time_s: [min, max] seconds the sampled command is held (short = frequent transitions)",
        "- rationale: one sentence on what skill this case trains",
        "",
        f"Hard safety envelope (values outside are clipped): {json.dumps(envelope)}",
        "",
        "Category rules:",
        "- stand: all three ranges [0, 0] (learn to stand still and to stop).",
        "- forward / backward: lin_vel_x strictly positive / negative, lin_vel_y and ang_vel_z near 0.",
        "- lateral: lin_vel_y non-zero, |lin_vel_x| <= 0.1.",
        "- diagonal: both lin_vel_x and lin_vel_y non-zero.",
        "- turn_in_place: |ang_vel_z| > 0.2 with |lin_vel_x|, |lin_vel_y| <= 0.1.",
        "- arc: forward or backward walking while turning (lin_vel_x and ang_vel_z non-zero).",
        "- omni: all three components may be non-zero at once.",
        f"Must include at least one case of each category in {list(REQUIRED_CATEGORIES)}.",
        "Flat-ground walking is left/right symmetric: for every case that has a lateral or turning",
        "direction, also include its mirror case (same lin_vel_x, lin_vel_y and ang_vel_z sign-flipped).",
        "",
        "Cover the space deliberately: slow precise speeds (0.1-0.3 m/s, where gaits often stall),",
        "nominal speeds, fast speeds near the envelope, both signs of every axis, quick stop-and-go",
        "transitions (short hold_time_s), and long steady holds.",
        "",
        "Also return a robustness block (domain randomisation applied to every case):",
        f"friction [min, max], added_base_mass_kg [min, max], push_velocity_mps (max push, at least"
        f" {MIN_PUSH_VELOCITY} so the gait learns to recover), push_interval_s [min, max], rationale."
        f" Envelope: {json.dumps(robustness)}",
    ]
    if findings:
        lines += ["", "Known weaknesses of the previous walking policy (address them with dedicated cases):"]
        lines += [f"- {f}" for f in findings]
    if previous_cases:
        keys = ("id", "category", "title", "title_ko", "difficulty", "weight", "lin_vel_x", "lin_vel_y", "ang_vel_z",
                "hold_time_s")
        compact = [{k: c[k] for k in keys} for c in previous_cases]
        lines += [
            "",
            "The policy above was trained on this catalog. Revise it rather than starting over: keep cases that",
            "already pass, and change weights, ranges, hold times or difficulty, or add cases, so that training",
            "spends more effort on the failing cases. All rules above still apply.",
            json.dumps(compact, ensure_ascii=False),
        ]
    return "\n".join(lines)


def build_repair_prompt(problems: list[str]) -> str:
    return (
        "Your previous catalog failed validation. Fix every problem below and return the complete corrected "
        "JSON (all cases plus the robustness block), not a diff:\n- " + "\n- ".join(problems)
    )

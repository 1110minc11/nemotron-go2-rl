"""Nemotron -> validated walking-case catalog.

Loop: ask Nemotron for a catalog, validate it, send the remaining problems back as a
repair prompt, and stop when the catalog is clean or the attempt budget runs out. If
problems remain after the last attempt, offending cases are dropped and minimal
"guard" cases fill missing categories, so training never starts from an
incomplete catalog. Every fix is recorded in the output file.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from datetime import datetime, timezone

from .client import NemotronClient, parse_json_reply
from .prompt import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from .schema import (
    CATEGORIES,
    ENVELOPE,
    LLM_RESPONSE_SCHEMA,
    MIN_PUSH_VELOCITY,
    MIN_REACH,
    ROBUSTNESS_ENVELOPE,
    SCHEMA_VERSION,
    _category_problems,
    _direction_name_problems,
    _reach_problems,
    _slow_problems,
    _symmetry_problems,
    validate_catalog,
)

GUARD_CASES = {
    "stand": dict(lin_vel_x=[0, 0], lin_vel_y=[0, 0], ang_vel_z=[0, 0], difficulty=1, title_ko="정지 유지"),
    "forward": dict(lin_vel_x=[0.3, 0.8], lin_vel_y=[0, 0], ang_vel_z=[0, 0], difficulty=1, title_ko="전진"),
    "backward": dict(lin_vel_x=[-0.6, -0.2], lin_vel_y=[0, 0], ang_vel_z=[0, 0], difficulty=1, title_ko="후진"),
    "lateral": dict(lin_vel_x=[0, 0], lin_vel_y=[-0.4, 0.4], ang_vel_z=[0, 0], difficulty=2, title_ko="옆걸음"),
    "turn_in_place": dict(lin_vel_x=[0, 0], lin_vel_y=[0, 0], ang_vel_z=[-1.0, 1.0], difficulty=1, title_ko="제자리 회전"),
    "arc": dict(lin_vel_x=[0.3, 0.8], lin_vel_y=[0, 0], ang_vel_z=[-0.8, 0.8], difficulty=2, title_ko="곡선 보행"),
}


# axis -> (category, Korean title) for the positive and the negative direction
REACH_GUARD = {
    "lin_vel_x": (("forward", "빠른 전진"), ("backward", "빠른 후진")),
    "lin_vel_y": (("lateral", "빠른 왼쪽 옆걸음"), ("lateral", "빠른 오른쪽 옆걸음")),
    "ang_vel_z": (("turn_in_place", "빠른 왼쪽 회전"), ("turn_in_place", "빠른 오른쪽 회전")),
}


_LR_SWAP = {"left": "right", "right": "left", "ccw": "cw", "cw": "ccw", "Left": "Right", "Right": "Left",
            "CCW": "CW", "CW": "CCW", "왼쪽": "오른쪽", "오른쪽": "왼쪽", "좌": "우", "우": "좌"}


def _swap_left_right(text: str) -> str:
    """Mirror the direction words in a case name (left <-> right, ccw <-> cw)."""
    return re.sub("|".join(sorted(map(re.escape, _LR_SWAP), key=len, reverse=True)), lambda m: _LR_SWAP[m.group(0)], text)


SLOW_GUARD = {
    "forward": [("", {"lin_vel_x": [0.1, 0.3]}, "느린 전진")],
    "backward": [("", {"lin_vel_x": [-0.3, -0.1]}, "느린 후진")],
    "lateral": [("_left", {"lin_vel_y": [0.1, 0.3]}, "느린 왼쪽 옆걸음"),
                ("_right", {"lin_vel_y": [-0.3, -0.1]}, "느린 오른쪽 옆걸음")],
}


def _guard(raw: dict) -> tuple[dict, list[str]]:
    """Drop cases that contradict their category and add templates for missing categories."""
    notes = []
    probe = validate_catalog(raw)
    kept = []
    for case in probe.cases:
        if case["category"] not in CATEGORIES:
            bad = [f"unknown category {case['category']!r}"]
        else:
            bad = _category_problems(case) + _direction_name_problems(case)
        if bad:
            notes.append(f"guard dropped {case['id']}: {bad[0]}")
        else:
            kept.append(case)
    for problem in _reach_problems(kept):
        axis = next(a for a in REACH_GUARD if a in problem)
        sign = 1.0 if ">=" in problem else -1.0
        category, title_ko = REACH_GUARD[axis][0 if sign > 0 else 1]
        lo, hi = sorted(sign * f * abs(ENVELOPE[axis][0 if sign < 0 else 1]) for f in (MIN_REACH, 0.8))
        case = dict(
            id=f"guard_reach_{axis}_{'pos' if sign > 0 else 'neg'}",
            category=category,
            title=f"Guard fast {axis} {'+' if sign > 0 else '-'}",
            title_ko=title_ko,
            difficulty=3,
            weight=0.04,
            lin_vel_x=[0.0, 0.0],
            lin_vel_y=[0.0, 0.0],
            ang_vel_z=[0.0, 0.0],
            hold_time_s=[2.0, 5.0],
            rationale="Inserted by the validator so every axis reaches 60 % of the envelope.",
            source="guard",
        )
        case[axis] = [round(lo, 3), round(hi, 3)]
        kept.append(case)
        notes.append(f"guard added {case['id']}")
    for problem in _slow_problems(kept):
        category = next(c for c in SLOW_GUARD if f"'{c}'" in problem)
        for suffix, ranges, title_ko in SLOW_GUARD[category]:
            case = dict(id=f"guard_slow_{category}{suffix}", category=category, title=f"Guard slow {category}",
                        title_ko=title_ko, difficulty=1, weight=0.05, lin_vel_x=[0.0, 0.0], lin_vel_y=[0.0, 0.0],
                        ang_vel_z=[0.0, 0.0], hold_time_s=[3.0, 6.0], source="guard",
                        rationale="Inserted by the validator: slow commands are where gaits stall.")
            case.update(ranges)
            kept.append(case)
            notes.append(f"guard added {case['id']}")
    for problem in _symmetry_problems(kept):
        case_id = problem.split(":", 1)[0]
        source = next(c for c in kept if c["id"] == case_id)
        mirror = copy.deepcopy(source)
        mirror.update(
            id=f"guard_mirror_{_swap_left_right(case_id)}",
            lin_vel_y=[-source["lin_vel_y"][1], -source["lin_vel_y"][0]],
            ang_vel_z=[-source["ang_vel_z"][1], -source["ang_vel_z"][0]],
            title=_swap_left_right(source["title"]),
            title_ko=_swap_left_right(source["title_ko"]),
            rationale="Mirror case inserted by the validator to keep left/right symmetry.",
            source="guard",
        )
        kept.append(mirror)
        notes.append(f"guard added {mirror['id']}")
    robustness = dict(raw.get("robustness") or {})
    if float(robustness.get("push_velocity_mps", 0.0) or 0.0) < MIN_PUSH_VELOCITY:
        robustness["push_velocity_mps"] = MIN_PUSH_VELOCITY
        notes.append(f"guard raised push_velocity_mps to {MIN_PUSH_VELOCITY}")
    present = {c["category"] for c in kept}
    for category, template in GUARD_CASES.items():
        if category not in present:
            case = dict(
                id=f"guard_{category}",
                category=category,
                title=f"Guard {category.replace('_', ' ')}",
                weight=0.06,
                hold_time_s=[3.0, 8.0],
                rationale="Inserted by the validator because Nemotron did not provide a valid case.",
                source="guard",
                **copy.deepcopy(template),
            )
            kept.append(case)
            notes.append(f"guard added {case['id']}")
    return {"cases": kept, "robustness": robustness}, notes


def generate_catalog(
    client: NemotronClient,
    name: str,
    num_cases: int = 16,
    max_attempts: int = 3,
    temperature: float = 0.6,
    seed: int | None = 7,
    findings: list[str] | None = None,
    previous: dict | None = None,
    log=print,
) -> dict:
    user_prompt = build_user_prompt(num_cases, findings, previous["cases"] if previous else None)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    attempts = []
    raw, result = None, None

    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        reply = client.chat_json(messages, LLM_RESPONSE_SCHEMA, temperature=temperature, seed=seed)
        elapsed = time.time() - t0
        record = {"attempt": attempt, "seconds": round(elapsed, 1), "usage": reply.usage}
        try:
            raw = parse_json_reply(reply.content)
        except (ValueError, json.JSONDecodeError) as e:
            record["parse_error"] = str(e)
            attempts.append(record)
            log(f"[attempt {attempt}] reply was not JSON ({e}); asking again")
            messages += [
                {"role": "assistant", "content": reply.content[:2000]},
                {"role": "user", "content": "That was not valid JSON. Return only the JSON object."},
            ]
            continue
        result = validate_catalog(raw)
        record.update(num_cases=len(result.cases), fixes=len(result.fixes), problems=result.problems)
        attempts.append(record)
        log(
            f"[attempt {attempt}] {elapsed:.0f}s, {len(result.cases)} cases, "
            f"{len(result.fixes)} auto-fixes, {len(result.problems)} problems"
        )
        if result.ok:
            break
        for p in result.problems:
            log(f"    - {p}")
        messages += [
            {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
            {"role": "user", "content": build_repair_prompt(result.problems)},
        ]

    if raw is None:
        raise RuntimeError("Nemotron never returned parseable JSON")

    guard_notes: list[str] = []
    if not result.ok:
        guarded, guard_notes = _guard(raw)
        result = validate_catalog(guarded)
        for note in guard_notes:
            log(f"    {note}")
        if not result.ok:
            raise RuntimeError("catalog still invalid after guard: " + "; ".join(result.problems))

    prompt_hash = hashlib.sha256((SYSTEM_PROMPT + "\n" + user_prompt).encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "robot": "unitree_go2",
        "environment": "flat_empty_plane",
        "generator": {
            "model": client.model,
            "base_url": client.base_url,
            "temperature": temperature,
            "seed": seed,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prompt_sha256": prompt_hash,
            "attempts": attempts,
        },
        "envelope": {k: list(v) for k, v in ENVELOPE.items()},
        "robustness_envelope": {k: list(v) for k, v in ROBUSTNESS_ENVELOPE.items()},
        "parent": previous["name"] if previous else None,
        "findings": findings or [],
        "cases": result.cases,
        "robustness": result.robustness,
        "validation": {"auto_fixes": result.fixes, "guard": guard_notes},
    }

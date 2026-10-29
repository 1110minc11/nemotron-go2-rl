"""Nemotron -> validated obstacle-scenario catalog (same generate / validate / repair / guard loop as the walking cases)."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone

from ..cases.client import NemotronClient, parse_json_reply
from .prompt import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from .schema import ENVELOPE, FAMILIES, LLM_RESPONSE_SCHEMA, REQUIRED_FAMILIES, SCHEMA_VERSION, _family_problems, validate_catalog

_BASE = dict(goal_distance=[2.6, 4.2], goal_bearing_deg=[-180, 180], along=[0.4, 0.65], lateral=[0.0, 0.4],
             count=[1, 2], gap=[1.2, 1.6], weight=0.05, source="guard",
             rationale="Inserted by the validator because Nemotron did not provide a valid scenario.")
GUARD = {
    "single_block": dict(title="Guard single block", title_ko="상자 하나", difficulty=1),
    "scatter": dict(title="Guard scatter", title_ko="흩어진 장애물", difficulty=2, lateral=[0.3, 1.3], count=[2, 3]),
    "slalom": dict(title="Guard slalom", title_ko="지그재그", difficulty=2, goal_distance=[3.8, 5.0], lateral=[0.4, 0.7],
                   count=[2, 3]),
    "gate": dict(title="Guard gate", title_ko="좁은 문", difficulty=2, lateral=[0.0, 0.5], gap=[1.1, 1.6]),
    "wall_gap": dict(title="Guard doorway", title_ko="벽의 문", difficulty=3, lateral=[0.0, 0.8], gap=[1.0, 1.4]),
    "corridor": dict(title="Guard corridor", title_ko="좁은 복도", difficulty=3, goal_distance=[4.2, 5.0], along=[0.45, 0.55],
                     lateral=[0.0, 0.3], gap=[0.9, 1.3]),
    "side_clutter": dict(title="Guard side clutter", title_ko="옆 장애물", difficulty=1, lateral=[1.1, 1.7], count=[1, 3]),
}


def _guard(raw: dict) -> tuple[dict, list[str]]:
    """Drop scenarios that violate their family rules; add templates for missing families and coverage."""
    notes, kept = [], []
    for case in validate_catalog(raw).cases:
        bad = [f"unknown family {case['family']!r}"] if case["family"] not in FAMILIES else _family_problems(case)
        if bad:
            notes.append(f"guard dropped {case['id']}: {bad[0]}")
        else:
            kept.append(case)
    present = {c["family"] for c in kept}
    for fam in REQUIRED_FAMILIES:
        if fam not in present:
            kept.append({"id": f"guard_{fam}", "family": fam, **copy.deepcopy(_BASE), **copy.deepcopy(GUARD[fam])})
            notes.append(f"guard added guard_{fam}")
    for level, fam in ((1, "single_block"), (2, "gate"), (3, "wall_gap")):
        if not any(c["difficulty"] == level for c in kept):
            kept.append({"id": f"guard_level{level}", "family": fam, **copy.deepcopy(_BASE), **copy.deepcopy(GUARD[fam]),
                         "difficulty": level})
            notes.append(f"guard added guard_level{level}")
    if not any(c["difficulty"] == 1 and c["family"] in ("open", "single_block") for c in kept):
        kept.append({"id": "guard_start", "family": "single_block", **copy.deepcopy(_BASE), **copy.deepcopy(GUARD["single_block"])})
        notes.append("guard added guard_start")
    if not any(c["goal_bearing_deg"][1] >= 150 or c["goal_bearing_deg"][0] <= -150 for c in kept):
        kept.append({"id": "guard_behind", "family": "single_block", **copy.deepcopy(_BASE), **copy.deepcopy(GUARD["single_block"]),
                     "goal_bearing_deg": [150, 180]})
        notes.append("guard added guard_behind")
    if not any(c["goal_bearing_deg"][0] <= -30 and c["goal_bearing_deg"][1] >= 30 for c in kept):
        kept.append({"id": "guard_front", "family": "single_block", **copy.deepcopy(_BASE), **copy.deepcopy(GUARD["single_block"]),
                     "goal_bearing_deg": [-40, 40]})
        notes.append("guard added guard_front")
    if not any(c["family"] == "corridor" and c["gap"][1] <= 1.3 for c in kept):
        kept.append({"id": "guard_tight_corridor", "family": "corridor", **copy.deepcopy(_BASE), **copy.deepcopy(GUARD["corridor"])})
        notes.append("guard added guard_tight_corridor")
    if not any(c["family"] == "scatter" and c["count"][1] >= 4 for c in kept):
        kept.append({"id": "guard_dense_scatter", "family": "scatter", **copy.deepcopy(_BASE), **copy.deepcopy(GUARD["scatter"]),
                     "difficulty": 3, "goal_distance": [3.5, 5.0], "along": [0.3, 0.8], "lateral": [0.3, 1.5], "count": [4, 5]})
        notes.append("guard added guard_dense_scatter")
    if not any(c["family"] in ("gate", "wall_gap") and c["gap"][1] <= 1.3 for c in kept):
        kept.append({"id": "guard_tight_gate", "family": "gate", **copy.deepcopy(_BASE), **copy.deepcopy(GUARD["gate"]),
                     "difficulty": 3, "gap": [0.9, 1.2]})
        notes.append("guard added guard_tight_gate")
    return {"cases": kept}, notes


def generate_catalog(client: NemotronClient, name: str, num_cases: int = 16, max_attempts: int = 4,
                     temperature: float = 0.6, seed: int | None = 7, findings: list[str] | None = None, log=print) -> dict:
    user_prompt = build_user_prompt(num_cases, findings)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    attempts, raw, result = [], None, None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        reply = client.chat_json(messages, LLM_RESPONSE_SCHEMA, temperature=temperature, seed=seed)
        record = {"attempt": attempt, "seconds": round(time.time() - t0, 1), "usage": reply.usage}
        try:
            raw = parse_json_reply(reply.content)
        except (ValueError, json.JSONDecodeError) as e:
            record["parse_error"] = str(e)
            attempts.append(record)
            log(f"[attempt {attempt}] reply was not JSON ({e}); asking again")
            messages += [{"role": "assistant", "content": reply.content[:2000]},
                         {"role": "user", "content": "That was not valid JSON. Return only the JSON object."}]
            continue
        result = validate_catalog(raw)
        record.update(num_cases=len(result.cases), fixes=len(result.fixes), problems=result.problems)
        attempts.append(record)
        log(f"[attempt {attempt}] {record['seconds']:.0f}s, {len(result.cases)} cases, {len(result.fixes)} auto-fixes, "
            f"{len(result.problems)} problems")
        if result.ok:
            break
        for p in result.problems:
            log(f"    - {p}")
        messages += [{"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
                     {"role": "user", "content": build_repair_prompt(result.problems)}]
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
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "robot": "unitree_go2",
        "environment": "flat_plane_with_obstacles",
        "generator": {"model": client.model, "base_url": client.base_url, "temperature": temperature, "seed": seed,
                      "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "prompt_sha256": hashlib.sha256((SYSTEM_PROMPT + "\n" + user_prompt).encode()).hexdigest(),
                      "attempts": attempts},
        "envelope": {k: list(v) for k, v in ENVELOPE.items()},
        "findings": findings or [],
        "cases": result.cases,
        "validation": {"auto_fixes": result.fixes, "guard": guard_notes},
    }

"""Unit tests for the case validator (no Isaac Sim needed): python -m pytest tests"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nemotron_go2_rl import DEFAULT_CATALOG  # noqa: E402
from nemotron_go2_rl.cases.client import parse_json_reply  # noqa: E402
from nemotron_go2_rl.cases.generate import _guard  # noqa: E402
from nemotron_go2_rl.cases.schema import ENVELOPE, MIN_STAND_WEIGHT, load_catalog, validate_catalog  # noqa: E402


def case(cid, category, vx=(0, 0), vy=(0, 0), wz=(0, 0), difficulty=1, weight=1.0):
    return {
        "id": cid,
        "category": category,
        "title": cid,
        "title_ko": cid,
        "difficulty": difficulty,
        "weight": weight,
        "lin_vel_x": list(vx),
        "lin_vel_y": list(vy),
        "ang_vel_z": list(wz),
        "hold_time_s": [2, 6],
        "rationale": "",
    }


BASE = {
    "cases": [
        case("stand", "stand"),
        case("fwd_slow", "forward", vx=(0.1, 0.3)),
        case("fwd_fast", "forward", vx=(0.6, 1.0), difficulty=2),
        case("back", "backward", vx=(-0.6, -0.2)),
        case("side", "lateral", vy=(-0.4, 0.4), difficulty=2),
        case("diag_l", "diagonal", vx=(0.2, 0.5), vy=(0.2, 0.4), difficulty=2),
        case("diag_r", "diagonal", vx=(0.2, 0.5), vy=(-0.4, -0.2), difficulty=2),
        case("turn", "turn_in_place", wz=(-1.0, 1.0)),
        case("arc_l", "arc", vx=(0.3, 0.7), wz=(0.3, 0.8), difficulty=2),
        case("arc_r", "arc", vx=(0.3, 0.7), wz=(-0.8, -0.3), difficulty=2),
        case("omni", "omni", vx=(-0.5, 0.5), vy=(-0.3, 0.3), wz=(-1, 1), difficulty=3),
    ],
    "robustness": {
        "friction": [0.5, 1.2],
        "added_base_mass_kg": [-1, 2],
        "push_velocity_mps": 0.5,
        "push_interval_s": [5, 10],
        "rationale": "",
    },
}


def test_valid_catalog_passes_and_weights_normalise():
    result = validate_catalog(BASE)
    assert result.ok, result.problems
    assert abs(sum(c["weight"] for c in result.cases) - 1.0) < 1e-3


def test_ranges_are_clamped_and_swapped():
    raw = copy.deepcopy(BASE)
    raw["cases"][2]["lin_vel_x"] = [3.0, 0.6]  # swapped and above the envelope
    result = validate_catalog(raw)
    assert result.cases[2]["lin_vel_x"] == [0.6, ENVELOPE["lin_vel_x"][1]]
    assert any("swapped" in f for f in result.fixes) and any("clamped" in f for f in result.fixes)


def test_category_mismatch_is_a_problem():
    raw = copy.deepcopy(BASE)
    raw["cases"][4]["lin_vel_x"] = [0.5, 0.8]  # "lateral" that walks forward
    result = validate_catalog(raw)
    assert not result.ok and any("lateral" in p for p in result.problems)


def test_stand_is_forced_to_zero_and_keeps_min_weight():
    raw = copy.deepcopy(BASE)
    raw["cases"][0]["lin_vel_x"] = [0.1, 0.2]
    raw["cases"][0]["weight"] = 0.001
    result = validate_catalog(raw)
    stand = result.cases[0]
    assert stand["lin_vel_x"] == [0.0, 0.0]
    assert stand["weight"] >= MIN_STAND_WEIGHT - 1e-3


def test_missing_category_detected_and_guard_repairs_it():
    raw = copy.deepcopy(BASE)
    raw["cases"] = [c for c in raw["cases"] if c["category"] != "backward"]
    raw["cases"].append(case("bad_lateral", "lateral", vx=(0.5, 0.9)))
    assert not validate_catalog(raw).ok
    guarded, notes = _guard(raw)
    result = validate_catalog(guarded)
    assert result.ok, result.problems
    assert any(c["category"] == "backward" and c["source"] == "guard" for c in result.cases)
    assert all(c["id"] != "bad_lateral" for c in result.cases)


def test_duplicate_ids_are_renamed():
    raw = copy.deepcopy(BASE)
    raw["cases"].append(case("turn", "turn_in_place", wz=(0.5, 1.2), difficulty=2))
    ids = [c["id"] for c in validate_catalog(raw).cases]
    assert len(ids) == len(set(ids))


def test_robustness_clamped():
    raw = copy.deepcopy(BASE)
    raw["robustness"]["push_velocity_mps"] = 5.0
    raw["robustness"]["friction"] = [0.05, 3.0]
    rob = validate_catalog(raw).robustness
    assert rob["push_velocity_mps"] == 1.0
    assert rob["friction"] == [0.3, 1.5]


def test_parse_reply_with_think_and_fence():
    text = '<think>planning...</think>\n```json\n{"cases": [], "robustness": {}}\n```'
    assert parse_json_reply(text) == {"cases": [], "robustness": {}}


def test_committed_catalog_is_valid():
    if not os.path.exists(DEFAULT_CATALOG):
        return
    catalog = load_catalog(DEFAULT_CATALOG)
    assert catalog["generator"]["model"]
    json.dumps(catalog)


def test_missing_mirror_case_is_a_problem_and_guard_mirrors_it():
    raw = copy.deepcopy(BASE)
    raw["cases"] = [c for c in raw["cases"] if c["id"] != "arc_r"]
    result = validate_catalog(raw)
    assert any(p.startswith("arc_l: no left/right mirror") for p in result.problems)
    guarded, _ = _guard(raw)
    fixed = validate_catalog(guarded)
    assert fixed.ok, fixed.problems
    mirror = next(c for c in fixed.cases if c["id"] == "guard_mirror_arc_l")
    assert mirror["ang_vel_z"] == [-0.8, -0.3]


def test_zero_push_is_a_problem():
    raw = copy.deepcopy(BASE)
    raw["robustness"]["push_velocity_mps"] = 0.0
    assert any("push_velocity_mps" in p for p in validate_catalog(raw).problems)
    guarded, _ = _guard(raw)
    assert validate_catalog(guarded).ok


def test_direction_names_follow_the_body_frame():
    raw = copy.deepcopy(BASE)
    raw["cases"].append(case("side_left_fast", "lateral", vy=(-0.5, -0.3), difficulty=3))
    raw["cases"].append(case("side_right_fast", "lateral", vy=(0.3, 0.5), difficulty=3))
    problems = validate_catalog(raw).problems
    assert any(p.startswith("side_left_fast: named left") for p in problems)
    assert any(p.startswith("side_right_fast: named right") for p in problems)


def test_each_axis_must_reach_most_of_the_envelope():
    raw = copy.deepcopy(BASE)
    for c in raw["cases"]:
        c["ang_vel_z"] = [max(c["ang_vel_z"][0], -0.5), min(c["ang_vel_z"][1], 0.5)]
    problems = validate_catalog(raw).problems
    assert any("ang_vel_z >= 0.9" in p for p in problems) and any("ang_vel_z <= -0.9" in p for p in problems)
    guarded, notes = _guard(raw)
    assert validate_catalog(guarded).ok, validate_catalog(guarded).problems
    assert any("guard_reach_ang_vel_z_pos" in n for n in notes)


def test_lateral_with_fast_yaw_is_not_lateral():
    raw = copy.deepcopy(BASE)
    raw["cases"][4]["ang_vel_z"] = [0.8, 1.2]  # "side" stepping while spinning
    assert any("'lateral' needs |ang_vel_z|" in p for p in validate_catalog(raw).problems)


def test_slow_lateral_required_and_guarded():
    raw = copy.deepcopy(BASE)
    raw["cases"].append(case("side_fast_left", "lateral", vy=(0.4, 0.6), difficulty=3))
    raw["cases"] = [c for c in raw["cases"] if c["id"] != "side"] + [case("side_fast_right", "lateral", vy=(-0.6, -0.4))]
    assert any("no slow 'lateral'" in p for p in validate_catalog(raw).problems)
    guarded, _ = _guard(raw)
    result = validate_catalog(guarded)
    assert result.ok, result.problems
    assert {"guard_slow_lateral_left", "guard_slow_lateral_right"} <= {c["id"] for c in result.cases}


def test_feedback_lists_failing_cases(tmp_path):
    from nemotron_go2_rl.cases.feedback import findings_from_eval
    from nemotron_go2_rl.cases.prompt import build_user_prompt

    ev = {"policies": {"p": {"cases": [
        {"case": "side", "success_rate": 0.05, "fall_rate": 0.0, "cmd_mean": [0, 0.2, 0], "vel_mean": [0, 0.13, 0]},
        {"case": "fwd_slow", "success_rate": 1.0, "fall_rate": 0.0, "cmd_mean": [0.2, 0, 0], "vel_mean": [0.2, 0, 0]},
    ]}}}
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(ev))
    findings = findings_from_eval(str(path), "p")
    assert findings[0].startswith("Case side: only 5%") and "fwd_slow" in findings[-1]
    prompt = build_user_prompt(16, findings, validate_catalog(BASE).cases)
    assert "Revise it rather than starting over" in prompt and '"id": "side"' in prompt

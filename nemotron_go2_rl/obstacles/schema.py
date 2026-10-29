"""Obstacle-scenario catalog: families, parameter envelope and deterministic validation.

Nemotron designs *scenarios* (a layout family plus parameter ranges, difficulty and weight).
The layout sampler (``layout.py``) turns a scenario into concrete obstacle poses; nothing here
trusts the LLM output. Problems are plain sentences that can be sent back to the model.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

SCHEMA_VERSION = 1

FAMILIES = ("open", "single_block", "scatter", "slalom", "gate", "wall_gap", "corridor", "side_clutter")
REQUIRED_FAMILIES = ("single_block", "scatter", "slalom", "gate", "wall_gap", "corridor", "side_clutter")

FAMILY_HELP = {
    "open": "no obstacles; the robot only walks to the goal",
    "single_block": "one 0.6 m box at fraction `along` of the path, lateral offset `lateral` (|y| small = blocking)",
    "scatter": "`count` (1-5) pillars (0.3 m) or boxes (0.6 m) scattered in the band between start and goal, "
    "|lateral| = half-width of the band (objects keep >= 1.5 m spacing; if the requested count does not fit, "
    "fewer objects are placed)",
    "slalom": "`count` (2-3) boxes alternating left/right of the path, spread evenly between start and goal "
    "(`along` and `gap` are ignored); |lateral| = side offset; needs goal_distance up to at least 3.5 m",
    "gate": "two 0.6 m boxes with a free opening of width `gap` at fraction `along`; opening centre offset `lateral`",
    "wall_gap": "two 2 m wall pieces across the path leaving a doorway of width `gap` at fraction `along`; "
    "doorway centre offset `lateral`",
    "corridor": "two 2 m walls parallel to the path forming a corridor of inner width `gap` that the robot must "
    "walk through, centred at fraction `along`, offset `lateral` (<= 0.4); needs goal_distance up to at least 4.0 m",
    "side_clutter": "`count` (1-3) obstacles beside the path, never blocking (|lateral| >= 1.0)",
}

# Hard envelope for every range. Values outside are clamped.
ENVELOPE = {
    "goal_distance": (2.2, 5.0),  # m
    "goal_bearing_deg": (-180.0, 180.0),
    "along": (0.3, 0.8),  # fraction of goal distance
    "lateral": (0.0, 1.8),  # m, magnitude; a random side is chosen at reset
    "count": (0.0, 5.0),
    "gap": (0.9, 2.0),  # m; doorway width or slalom spacing
}

RANGE_FIELDS = tuple(ENVELOPE)
MIN_CASES = 10
MAX_CASES = 24
MIN_DOORWAY = 0.9  # the Go2 (0.3 m wide) plus a 0.3 m safety margin on each side

_RANGE = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}

LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "cases": {
            "type": "array",
            "minItems": MIN_CASES,
            "maxItems": MAX_CASES,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "family": {"type": "string", "enum": list(FAMILIES)},
                    "title": {"type": "string"},
                    "title_ko": {"type": "string"},
                    "difficulty": {"type": "integer", "minimum": 1, "maximum": 3},
                    "weight": {"type": "number"},
                    **{name: _RANGE for name in RANGE_FIELDS},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "family", "title", "title_ko", "difficulty", "weight", *RANGE_FIELDS, "rationale"],
            },
        }
    },
    "required": ["cases"],
}


@dataclass
class ValidationResult:
    cases: list[dict]
    fixes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _slug(text) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "case"


def _range(value, name, cid, fixes):
    if isinstance(value, (int, float)):
        value = [value, value]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        fixes.append(f"{cid}: {name} malformed ({value!r}), reset to the envelope midpoint")
        lo, hi = ENVELOPE[name]
        return [(lo + hi) / 2, (lo + hi) / 2]
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        fixes.append(f"{cid}: {name} swapped [{lo:g}, {hi:g}]")
        lo, hi = hi, lo
    e_lo, e_hi = ENVELOPE[name]
    c_lo, c_hi = max(lo, e_lo), min(hi, e_hi)
    if c_lo > c_hi:
        c_lo = c_hi = e_lo if hi < e_lo else e_hi
    if (c_lo, c_hi) != (lo, hi):
        fixes.append(f"{cid}: {name} [{lo:g}, {hi:g}] clamped to [{c_lo:g}, {c_hi:g}]")
    return [round(c_lo, 3), round(c_hi, 3)]


def _family_problems(case: dict) -> list[str]:
    cid, fam = case["id"], case["family"]
    lat, cnt, gap = case["lateral"], case["count"], case["gap"]
    out = []
    if fam in ("gate", "wall_gap", "corridor") and gap[0] < MIN_DOORWAY - 1e-9:
        out.append(f"{cid}: {fam} gap must be at least {MIN_DOORWAY} m so the robot can physically pass")
    if fam == "corridor" and lat[1] > 0.4:
        out.append(f"{cid}: corridor offset |lateral| must be <= 0.4 m")
    if fam == "corridor" and case["goal_distance"][1] < 4.0 - 1e-9:
        out.append(f"{cid}: corridor needs goal_distance up to at least 4.0 m so the 2 m walls fit between start and goal")
    if fam == "side_clutter" and cnt[1] > 3:
        out.append(f"{cid}: side_clutter count must be <= 3")
    if fam == "gate" and lat[1] > 0.8:
        out.append(f"{cid}: gate opening offset |lateral| must be <= 0.8 m")
    if fam == "wall_gap" and lat[1] > 1.2:
        out.append(f"{cid}: wall_gap doorway offset |lateral| must be <= 1.2 m (the wall pieces are 2 m long)")
    if fam == "side_clutter" and lat[0] < 1.0 - 1e-9:
        out.append(f"{cid}: side_clutter |lateral| must be >= 1.0 m so it never blocks the path")
    if fam == "slalom" and cnt[0] < 2 - 1e-9:
        out.append(f"{cid}: slalom needs count >= 2")
    if fam in ("scatter", "side_clutter") and cnt[0] < 1 - 1e-9:
        out.append(f"{cid}: {fam} needs count >= 1")
    if fam == "slalom" and case["goal_distance"][1] < 3.5 - 1e-9:
        out.append(f"{cid}: slalom needs goal_distance up to at least 3.5 m so two boxes fit with clearance")
    if fam == "slalom" and not (0.25 <= lat[0] and lat[1] <= 1.0):
        out.append(f"{cid}: slalom |lateral| must stay within 0.25-1.0 m (offset from the path)")
    return out


def validate_catalog(raw: dict) -> ValidationResult:
    fixes: list[str] = []
    problems: list[str] = []
    cases: list[dict] = []
    seen: set[str] = set()
    for i, src in enumerate(raw.get("cases") or []):
        case = copy.deepcopy(src)
        cid = _slug(case.get("id") or case.get("title") or f"case_{i}")
        base, n = cid, 2
        while cid in seen:
            cid = f"{base}_{n}"
            n += 1
        if cid != case.get("id"):
            fixes.append(f"case #{i}: id {case.get('id')!r} normalised to {cid!r}")
        seen.add(cid)
        case["id"] = cid
        family = str(case.get("family", "")).strip().lower()
        if family not in FAMILIES:
            problems.append(f"{cid}: unknown family {family!r}; use one of {list(FAMILIES)}")
        case["family"] = family
        for name in RANGE_FIELDS:
            case[name] = _range(case.get(name, ENVELOPE[name]), name, cid, fixes)
        case["count"] = [int(round(case["count"][0])), int(round(case["count"][1]))]
        try:
            difficulty = int(case.get("difficulty", 1))
        except (TypeError, ValueError):
            difficulty = 1
        if difficulty not in (1, 2, 3):
            fixes.append(f"{cid}: difficulty {case.get('difficulty')!r} clipped into 1..3")
            difficulty = min(3, max(1, difficulty))
        case["difficulty"] = difficulty
        try:
            weight = float(case.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if weight <= 0:
            weight = 0.01
            fixes.append(f"{cid}: non-positive weight replaced by 0.01")
        case["weight"] = weight
        case["title"] = str(case.get("title") or cid)
        case["title_ko"] = str(case.get("title_ko") or case["title"])
        case["rationale"] = str(case.get("rationale") or "")
        case.setdefault("source", "nemotron")
        if family in FAMILIES:
            problems.extend(_family_problems(case))
        cases.append(case)

    if not (MIN_CASES <= len(cases) <= MAX_CASES):
        problems.append(f"catalog has {len(cases)} cases; need between {MIN_CASES} and {MAX_CASES}")
    present = {c["family"] for c in cases}
    for fam in REQUIRED_FAMILIES:
        if fam not in present:
            problems.append(f"missing a case with family '{fam}'")
    for level in (1, 2, 3):
        if not any(c["difficulty"] == level for c in cases):
            problems.append(f"no case with difficulty {level}; the curriculum needs all three levels")
    if not any(c["difficulty"] == 1 and c["family"] in ("open", "single_block") for c in cases):
        problems.append("difficulty 1 needs an 'open' or 'single_block' case to start the curriculum")
    if not any(c["goal_bearing_deg"][1] >= 150 or c["goal_bearing_deg"][0] <= -150 for c in cases):
        problems.append("no case sends the goal behind the robot (|bearing| >= 150 deg)")
    if not any(c["goal_bearing_deg"][0] <= -30 and c["goal_bearing_deg"][1] >= 30 for c in cases):
        problems.append("no case spans the frontal sector (-30..30 deg)")
    if not any(c["family"] in ("gate", "wall_gap") and c["gap"][1] <= 1.3 for c in cases):
        problems.append("no tight doorway case (a gate or wall_gap whose gap can be 1.3 m or less)")
    if not any(c["family"] == "corridor" and c["gap"][1] <= 1.3 for c in cases):
        problems.append("no tight corridor case (a corridor whose width can be 1.3 m or less)")
    if not any(c["family"] == "scatter" and c["count"][1] >= 4 for c in cases):
        problems.append("no dense scatter case (a scatter whose count can reach 4 or more)")

    total = sum(c["weight"] for c in cases) or 1.0
    for c in cases:
        c["weight"] = round(c["weight"] / total, 4)
    return ValidationResult(cases=cases, fixes=fixes, problems=problems)


def load_catalog(path: str) -> dict:
    import json

    with open(path, encoding="utf-8") as f:
        catalog = json.load(f)
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {catalog.get('schema_version')} != {SCHEMA_VERSION}")
    result = validate_catalog(catalog)
    if not result.ok and not catalog.get("control"):
        raise ValueError(f"{path}: catalog failed validation:\n  - " + "\n  - ".join(result.problems))
    catalog["cases"] = result.cases
    return catalog

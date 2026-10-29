"""Walking-case catalog: JSON schema, safety envelope and deterministic validation.

Nemotron *proposes* walking cases. Nothing in this module trusts that proposal:
every range is clamped to the Go2 safety envelope, every case is checked against
its category, and the catalog as a whole must cover the motions a flat-ground
walking policy needs. Problems are returned as plain sentences so they can be
fed back to Nemotron for a repair round.

This module is pure Python (no Isaac Sim / torch) so it can run anywhere.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

SCHEMA_VERSION = 1

# Body-frame command limits for the Unitree Go2 on flat ground. These are the
# hard limits the simulator will ever see, regardless of what the LLM writes.
ENVELOPE = {
    "lin_vel_x": (-1.0, 1.0),  # m/s
    "lin_vel_y": (-0.6, 0.6),  # m/s
    "ang_vel_z": (-1.5, 1.5),  # rad/s
    "hold_time_s": (1.0, 10.0),  # s, how long one command is held before resampling
}

ROBUSTNESS_ENVELOPE = {
    "friction": (0.3, 1.5),  # static friction coefficient of the feet/ground pair
    "added_base_mass_kg": (-1.0, 3.0),
    "push_velocity_mps": (0.0, 1.0),
    "push_interval_s": (3.0, 15.0),
}

CATEGORIES = (
    "stand",
    "forward",
    "backward",
    "lateral",
    "diagonal",
    "turn_in_place",
    "arc",
    "omni",
)

# Categories a flat-ground walking catalog must contain.
REQUIRED_CATEGORIES = ("stand", "forward", "backward", "lateral", "turn_in_place", "arc")

# A command component with |value| below this is treated as "zero" when
# checking that a case matches its category.
NEAR_ZERO = 0.12
# Yaw rate allowed in a "pure" forward / backward / lateral case.
PURE_YAW = 0.2
# Each translational category needs a slow case: gaits stall at low commanded speed.
SLOW_SPEED = 0.3
SLOW_AXES = {"forward": "lin_vel_x", "backward": "lin_vel_x", "lateral": "lin_vel_y"}
MIN_CASES = 10
MAX_CASES = 24
MIN_STAND_WEIGHT = 0.05
# Pushes are what teach recovery; a catalog without them trains a fragile gait.
MIN_PUSH_VELOCITY = 0.2

DEFAULT_ROBUSTNESS = {
    "friction": [0.5, 1.25],
    "added_base_mass_kg": [-1.0, 2.0],
    "push_velocity_mps": 0.5,
    "push_interval_s": [5.0, 10.0],
}

_RANGE = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}

# JSON schema handed to the LLM server as `response_format` (structured output).
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
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "title": {"type": "string"},
                    "title_ko": {"type": "string"},
                    "difficulty": {"type": "integer", "minimum": 1, "maximum": 3},
                    "weight": {"type": "number"},
                    "lin_vel_x": _RANGE,
                    "lin_vel_y": _RANGE,
                    "ang_vel_z": _RANGE,
                    "hold_time_s": _RANGE,
                    "rationale": {"type": "string"},
                },
                "required": [
                    "id",
                    "category",
                    "title",
                    "title_ko",
                    "difficulty",
                    "weight",
                    "lin_vel_x",
                    "lin_vel_y",
                    "ang_vel_z",
                    "hold_time_s",
                    "rationale",
                ],
            },
        },
        "robustness": {
            "type": "object",
            "properties": {
                "friction": _RANGE,
                "added_base_mass_kg": _RANGE,
                "push_velocity_mps": {"type": "number"},
                "push_interval_s": _RANGE,
                "rationale": {"type": "string"},
            },
            "required": ["friction", "added_base_mass_kg", "push_velocity_mps", "push_interval_s", "rationale"],
        },
    },
    "required": ["cases", "robustness"],
}


@dataclass
class ValidationResult:
    """Outcome of :func:`validate_catalog`."""

    cases: list[dict]
    robustness: dict
    fixes: list[str] = field(default_factory=list)
    """Automatic corrections that were applied (clamps, swaps, renames)."""
    problems: list[str] = field(default_factory=list)
    """Issues the validator could not fix by itself; sent back to Nemotron."""

    @property
    def ok(self) -> bool:
        return not self.problems


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug or "case"


def _as_range(value, name: str, fixes: list[str], case_id: str) -> list[float]:
    if isinstance(value, (int, float)):
        fixes.append(f"{case_id}: {name} was a scalar, expanded to [{value}, {value}]")
        value = [value, value]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        fixes.append(f"{case_id}: {name} malformed ({value!r}), reset to [0, 0]")
        return [0.0, 0.0]
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        fixes.append(f"{case_id}: {name} swapped [{lo}, {hi}]")
        lo, hi = hi, lo
    return [lo, hi]


def _clamp_range(rng: list[float], limits: tuple[float, float], name: str, fixes: list[str], case_id: str):
    lo, hi = rng
    c_lo, c_hi = max(lo, limits[0]), min(hi, limits[1])
    if c_lo > c_hi:  # entirely outside the envelope: collapse onto the nearest limit
        c_lo = c_hi = limits[0] if hi < limits[0] else limits[1]
    if (c_lo, c_hi) != (lo, hi):
        fixes.append(f"{case_id}: {name} [{lo:g}, {hi:g}] clamped to [{c_lo:g}, {c_hi:g}]")
    return [round(c_lo, 3), round(c_hi, 3)]


def _max_abs(rng: list[float]) -> float:
    return max(abs(rng[0]), abs(rng[1]))


def _category_problems(case: dict) -> list[str]:
    """Check that the command ranges actually express the declared category."""
    cid, cat = case["id"], case["category"]
    vx, vy, wz = case["lin_vel_x"], case["lin_vel_y"], case["ang_vel_z"]
    problems = []
    if cat in ("forward", "backward") and not (_max_abs(vy) <= NEAR_ZERO and _max_abs(wz) <= PURE_YAW):
        problems.append(f"{cid}: category '{cat}' needs |lin_vel_y| <= {NEAR_ZERO} and |ang_vel_z| <= {PURE_YAW}")
    if cat == "lateral" and _max_abs(wz) > PURE_YAW:
        problems.append(f"{cid}: category 'lateral' needs |ang_vel_z| <= {PURE_YAW}; turning while stepping is 'omni'")
    if cat == "forward" and not (vx[0] >= 0.0 and vx[1] > NEAR_ZERO):
        problems.append(f"{cid}: category 'forward' needs lin_vel_x >= 0 with a max above {NEAR_ZERO}")
    if cat == "backward" and not (vx[1] <= 0.0 and vx[0] < -NEAR_ZERO):
        problems.append(f"{cid}: category 'backward' needs lin_vel_x <= 0 with a min below -{NEAR_ZERO}")
    if cat == "lateral" and not (_max_abs(vy) > NEAR_ZERO and _max_abs(vx) <= NEAR_ZERO):
        problems.append(f"{cid}: category 'lateral' needs |lin_vel_y| > {NEAR_ZERO} and |lin_vel_x| <= {NEAR_ZERO}")
    if cat == "diagonal" and not (_max_abs(vx) > NEAR_ZERO and _max_abs(vy) > NEAR_ZERO):
        problems.append(f"{cid}: category 'diagonal' needs both |lin_vel_x| and |lin_vel_y| > {NEAR_ZERO}")
    if cat == "turn_in_place" and not (
        _max_abs(wz) > 0.2 and _max_abs(vx) <= NEAR_ZERO and _max_abs(vy) <= NEAR_ZERO
    ):
        problems.append(f"{cid}: category 'turn_in_place' needs |ang_vel_z| > 0.2 and near-zero linear velocity")
    if cat == "arc" and not (_max_abs(vx) > NEAR_ZERO and _max_abs(wz) > 0.2):
        problems.append(f"{cid}: category 'arc' needs |lin_vel_x| > {NEAR_ZERO} and |ang_vel_z| > 0.2")
    return problems


def _overlaps(a: list[float], b: list[float], eps: float = 0.02) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1]) + eps


def _symmetry_problems(cases: list[dict]) -> list[str]:
    """Flat-ground walking is left/right symmetric: every case needs a mirror case.

    The mirror of a case flips the sign of lin_vel_y and ang_vel_z and keeps lin_vel_x.
    A case whose ranges are already symmetric about zero is its own mirror.
    """
    problems = []
    for c in cases:
        mvy = [-c["lin_vel_y"][1], -c["lin_vel_y"][0]]
        mwz = [-c["ang_vel_z"][1], -c["ang_vel_z"][0]]
        mirrored = any(
            o["category"] == c["category"]
            and _overlaps(o["lin_vel_x"], c["lin_vel_x"])
            and _overlaps(o["lin_vel_y"], mvy)
            and _overlaps(o["ang_vel_z"], mwz)
            for o in cases
        )
        if not mirrored:
            problems.append(
                f"{c['id']}: no left/right mirror case; add a '{c['category']}' case with lin_vel_x "
                f"{c['lin_vel_x']}, lin_vel_y {mvy}, ang_vel_z {mwz}"
            )
    return problems


def _direction_name_problems(case: dict) -> list[str]:
    """Names must agree with the body frame: +y is left, +wz is counter-clockwise (a left turn)."""
    text = f"{case['id']} {case['title']}".lower().replace("-", "_")
    text_ko = case.get("title_ko", "")
    vy, wz = case["lin_vel_y"], case["ang_vel_z"]
    pos = (vy[0] >= 0 and vy[1] > 0) or (wz[0] >= 0 and wz[1] > 0)
    neg = (vy[1] <= 0 and vy[0] < 0) or (wz[1] <= 0 and wz[0] < 0)
    says_left = bool(re.search(r"(^|_| )(left|ccw)", text)) or "왼" in text_ko or "좌" in text_ko
    says_right = bool(re.search(r"(^|_| )(right|cw)", text)) or "오른" in text_ko or "우측" in text_ko
    if says_left and not says_right and not pos:
        return [f"{case['id']}: named left/ccw but its lin_vel_y or ang_vel_z is not positive (+y = left, +wz = ccw)"]
    if says_right and not says_left and not neg:
        return [f"{case['id']}: named right/cw but its lin_vel_y or ang_vel_z is not negative (-y = right, -wz = cw)"]
    return []


def _slow_problems(cases: list[dict]) -> list[str]:
    problems = []
    for category, axis in SLOW_AXES.items():
        slowest = [min(abs(c[axis][0]), abs(c[axis][1])) if c[axis][0] * c[axis][1] > 0 else 0.0
                   for c in cases if c["category"] == category]
        if slowest and min(slowest) > SLOW_SPEED:
            problems.append(f"no slow '{category}' case: add one whose {axis} range reaches |v| <= {SLOW_SPEED} m/s")
    return problems


# Each axis must be exercised up to this fraction of the envelope, in both directions.
MIN_REACH = 0.6


def _reach_problems(cases: list[dict]) -> list[str]:
    problems = []
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        lo_env, hi_env = ENVELOPE[axis]
        top = max((c[axis][1] for c in cases), default=0.0)
        bottom = min((c[axis][0] for c in cases), default=0.0)
        if top < MIN_REACH * hi_env:
            problems.append(f"no case reaches {axis} >= {MIN_REACH * hi_env:g} (max is {top:g}); cover faster motions")
        if bottom > MIN_REACH * lo_env:
            problems.append(f"no case reaches {axis} <= {MIN_REACH * lo_env:g} (min is {bottom:g}); cover faster motions")
    return problems


def validate_catalog(raw: dict) -> ValidationResult:
    """Normalise an LLM proposal into a simulator-safe catalog.

    Args:
        raw: Parsed JSON with ``cases`` and ``robustness`` keys (see :data:`LLM_RESPONSE_SCHEMA`).

    Returns:
        The cleaned cases and robustness block, the automatic fixes applied, and the
        remaining problems. A catalog is only usable for training when ``result.ok``.
    """
    fixes: list[str] = []
    problems: list[str] = []
    cases: list[dict] = []
    seen_ids: set[str] = set()

    raw_cases = raw.get("cases") or []
    for i, src in enumerate(raw_cases):
        case = copy.deepcopy(src)
        cid = _slug(case.get("id") or case.get("title") or f"case_{i}")
        base, n = cid, 2
        while cid in seen_ids:
            cid = f"{base}_{n}"
            n += 1
        if cid != case.get("id"):
            fixes.append(f"case #{i}: id {case.get('id')!r} normalised to {cid!r}")
        seen_ids.add(cid)
        case["id"] = cid

        category = str(case.get("category", "")).strip().lower()
        if category not in CATEGORIES:
            problems.append(f"{cid}: unknown category {category!r}; use one of {list(CATEGORIES)}")
        case["category"] = category

        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "hold_time_s"):
            rng = _as_range(case.get(name, [0.0, 0.0]), name, fixes, cid)
            case[name] = _clamp_range(rng, ENVELOPE[name], name, fixes, cid)

        if category == "stand":
            for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                if case[name] != [0.0, 0.0]:
                    fixes.append(f"{cid}: stand case forces {name} to [0, 0]")
                    case[name] = [0.0, 0.0]

        difficulty = case.get("difficulty", 1)
        try:
            difficulty = int(difficulty)
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
        if weight <= 0.0:
            fixes.append(f"{cid}: non-positive weight {weight:g} replaced by 0.01")
            weight = 0.01
        case["weight"] = weight

        case["title"] = str(case.get("title") or cid)
        case["title_ko"] = str(case.get("title_ko") or case["title"])
        case["rationale"] = str(case.get("rationale") or "")
        case.setdefault("source", "nemotron")

        if category in CATEGORIES:
            problems.extend(_category_problems(case))
        problems.extend(_direction_name_problems(case))
        cases.append(case)

    if not (MIN_CASES <= len(cases) <= MAX_CASES):
        problems.append(f"catalog has {len(cases)} cases; need between {MIN_CASES} and {MAX_CASES}")

    present = {c["category"] for c in cases}
    for category in REQUIRED_CATEGORIES:
        if category not in present:
            problems.append(f"missing a case with category '{category}'")
    if not any(c["difficulty"] == 1 and c["category"] != "stand" for c in cases):
        problems.append("no moving case with difficulty 1; the curriculum would start with nothing to learn")
    problems.extend(_symmetry_problems([c for c in cases if c["category"] in CATEGORIES]))
    problems.extend(_reach_problems(cases))
    problems.extend(_slow_problems(cases))

    # Normalise weights to a probability distribution and keep a minimum share of standing.
    total = sum(c["weight"] for c in cases) or 1.0
    for c in cases:
        c["weight"] = c["weight"] / total
    stand_weight = sum(c["weight"] for c in cases if c["category"] == "stand")
    if cases and 0.0 < stand_weight < MIN_STAND_WEIGHT:
        # give standing exactly MIN_STAND_WEIGHT and shrink the moving cases to fill the rest
        stand_scale = MIN_STAND_WEIGHT / stand_weight
        move_scale = (1.0 - MIN_STAND_WEIGHT) / (1.0 - stand_weight)
        for c in cases:
            c["weight"] *= stand_scale if c["category"] == "stand" else move_scale
        fixes.append(f"stand weight raised from {stand_weight:.3f} to {MIN_STAND_WEIGHT}")
    for c in cases:
        c["weight"] = round(c["weight"], 4)

    robustness = _validate_robustness(raw.get("robustness") or {}, fixes)
    if robustness["push_velocity_mps"] < MIN_PUSH_VELOCITY:
        problems.append(
            f"robustness.push_velocity_mps is {robustness['push_velocity_mps']:g}; use at least "
            f"{MIN_PUSH_VELOCITY} m/s so the policy learns to recover from pushes"
        )
    return ValidationResult(cases=cases, robustness=robustness, fixes=fixes, problems=problems)


def _validate_robustness(raw: dict, fixes: list[str]) -> dict:
    out = {"rationale": str(raw.get("rationale") or "")}
    for name in ("friction", "added_base_mass_kg", "push_interval_s"):
        rng = _as_range(raw.get(name, DEFAULT_ROBUSTNESS[name]), name, fixes, "robustness")
        out[name] = _clamp_range(rng, ROBUSTNESS_ENVELOPE[name], name, fixes, "robustness")
    try:
        push = float(raw.get("push_velocity_mps", DEFAULT_ROBUSTNESS["push_velocity_mps"]))
    except (TypeError, ValueError):
        push = DEFAULT_ROBUSTNESS["push_velocity_mps"]
    lo, hi = ROBUSTNESS_ENVELOPE["push_velocity_mps"]
    clamped = min(max(push, lo), hi)
    if clamped != push:
        fixes.append(f"robustness: push_velocity_mps {push:g} clamped to {clamped:g}")
    out["push_velocity_mps"] = round(clamped, 3)
    return out


def load_catalog(path: str) -> dict:
    """Read a catalog written by ``scripts/generate_cases.py`` and re-check it."""
    import json

    with open(path, encoding="utf-8") as f:
        catalog = json.load(f)
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {catalog.get('schema_version')} != {SCHEMA_VERSION}")
    result = validate_catalog(catalog)
    if not result.ok:
        raise ValueError(f"{path}: catalog failed validation:\n  - " + "\n  - ".join(result.problems))
    catalog["cases"] = result.cases
    catalog["robustness"] = result.robustness
    return catalog

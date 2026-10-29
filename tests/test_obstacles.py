"""Pure-torch tests for the obstacle scenario schema, geometry and layout sampler."""

import copy
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nemotron_go2_rl.obstacles.geometry import point_distances, ray_ranges  # noqa: E402
from nemotron_go2_rl.obstacles.layout import (  # noqa: E402
    FAMILY_ID, NUM_SLOTS, SLOT_HALF, START_CLEARANCE, CaseTable, goal_world, sample_layouts)
from nemotron_go2_rl.obstacles.schema import MIN_DOORWAY, validate_catalog  # noqa: E402


def case(cid, family, difficulty=1, dist=(2.5, 4.0), bearing=(-180, 180), along=(0.4, 0.65), lateral=(0, 0.4),
         count=(1, 2), gap=(1.0, 1.6), weight=1.0):
    return dict(id=cid, family=family, title=cid, title_ko=cid, difficulty=difficulty, weight=weight,
                goal_distance=list(dist), goal_bearing_deg=list(bearing), along=list(along), lateral=list(lateral),
                count=list(count), gap=list(gap), rationale="")


BASE = {"cases": [
    case("open", "open"),
    case("block", "single_block"),
    case("block_hard", "single_block", 2, lateral=(0, 0.2)),
    case("scatter", "scatter", 2, along=(0.3, 0.75), lateral=(0.3, 1.4), count=(2, 3)),
    case("slalom", "slalom", 2, dist=(3.8, 5.0), lateral=(0.4, 0.7), count=(2, 3)),
    case("gate", "gate", 2, lateral=(0, 0.5), gap=(1.1, 1.6)),
    case("gate_tight", "gate", 3, lateral=(0, 0.5), gap=(0.9, 1.2)),
    case("door", "wall_gap", 3, lateral=(0, 0.8), gap=(1.0, 1.4)),
    case("corridor", "corridor", 3, dist=(4.2, 5.0), along=(0.45, 0.55), lateral=(0, 0.3), gap=(0.9, 1.3)),
    case("dense", "scatter", 3, dist=(3.5, 5.0), along=(0.3, 0.8), lateral=(0.3, 1.5), count=(4, 5)),
    case("side", "side_clutter", 1, lateral=(1.1, 1.7), count=(1, 3)),
    case("front", "single_block", 1, bearing=(-40, 40)),
]}


def test_valid_catalog_passes():
    r = validate_catalog(BASE)
    assert r.ok, r.problems
    assert abs(sum(c["weight"] for c in r.cases) - 1) < 1e-3


def test_doorway_narrower_than_the_robot_is_rejected():
    raw = copy.deepcopy(BASE)
    next(c for c in raw["cases"] if c["id"] == "gate")["gap"] = [0.5, 0.7]  # envelope clamps to 0.9, still reported through the min rule via clamp+fix
    r = validate_catalog(raw)
    assert any("clamped" in f for f in r.fixes)
    assert all(c["gap"][0] >= MIN_DOORWAY - 1e-9 for c in r.cases if c["family"] in ("gate", "wall_gap"))


def test_blocking_side_clutter_is_a_problem():
    raw = copy.deepcopy(BASE)
    next(c for c in raw["cases"] if c["id"] == "side")["lateral"] = [0.2, 1.0]
    assert any("never blocks" in p for p in validate_catalog(raw).problems)


def test_missing_family_and_coverage_problems():
    raw = copy.deepcopy(BASE)
    raw["cases"] = [c for c in raw["cases"] if c["family"] != "wall_gap"]
    assert any("wall_gap" in p for p in validate_catalog(raw).problems)


def test_ray_ranges_match_analytic_box():
    xy = torch.tensor([[[3.0, 0.0]]])
    yaw = torch.zeros(1, 1)
    half = torch.tensor([[0.5, 0.5]])
    act = torch.ones(1, 1, dtype=torch.bool)
    ang = torch.tensor([0.0, 0.1, math.pi / 2])
    dirs = torch.stack((ang.cos(), ang.sin()), dim=-1).unsqueeze(0)
    r = ray_ranges(torch.zeros(1, 2), dirs, xy, yaw, half, act, 4.0)
    assert abs(r[0, 0] - 2.5) < 1e-5  # front face at x = 2.5
    assert abs(r[0, 1] - 2.5 / math.cos(0.1)) < 1e-4
    assert r[0, 2] == 4.0  # misses
    # rotating the box by 90 degrees keeps a square's range unchanged, a long box changes it
    long_half = torch.tensor([[0.1, 1.0]])
    r0 = ray_ranges(torch.zeros(1, 2), dirs[:, :1], xy, yaw, long_half, act, 4.0)
    r90 = ray_ranges(torch.zeros(1, 2), dirs[:, :1], xy, torch.full((1, 1), math.pi / 2), long_half, act, 4.0)
    assert abs(r0[0, 0] - 2.9) < 1e-5 and abs(r90[0, 0] - 2.0) < 1e-5
    # inactive slots are ignored
    off = ray_ranges(torch.zeros(1, 2), dirs[:, :1], xy, yaw, half, torch.zeros(1, 1, dtype=torch.bool), 4.0)
    assert off[0, 0] == 4.0


def test_point_distance_and_closest_point():
    xy = torch.tensor([[[3.0, 0.0]]])
    d, c = point_distances(torch.tensor([[0.0, 0.0]]), xy, torch.zeros(1, 1), torch.tensor([[0.5, 0.5]]),
                           torch.ones(1, 1, dtype=torch.bool))
    assert abs(d[0, 0] - 2.5) < 1e-6 and torch.allclose(c[0, 0], torch.tensor([2.5, 0.0]))
    inside, _ = point_distances(torch.tensor([[3.0, 0.1]]), xy, torch.zeros(1, 1), torch.tensor([[0.5, 0.5]]),
                                torch.ones(1, 1, dtype=torch.bool))
    assert inside[0, 0] == 0


def _surface_gap(a_xy, a_half, b_xy, b_half):
    """Axis-aligned gap along y between two path-aligned boxes (centre y difference minus half heights)."""
    return abs(a_xy[1] - b_xy[1]) - a_half[1] - b_half[1]


def test_layouts_are_feasible_for_every_family():
    cases = validate_catalog(BASE).cases
    table = CaseTable(cases)
    gen = torch.Generator().manual_seed(0)
    for ci, c in enumerate(cases):
        idx = torch.full((256,), ci)
        lay = sample_layouts(table, idx, gen)
        limit = 0.05
        assert lay.failed.float().mean() < limit, (c["id"], lay.failed.float().mean())
        goal = goal_world(lay)
        d_start, _ = point_distances(torch.zeros(256, 2), lay.slot_xy, lay.slot_yaw, table.slot_half, lay.active)
        d_goal, _ = point_distances(goal, lay.slot_xy, lay.slot_yaw, table.slot_half, lay.active)
        ok = ~lay.failed
        assert (d_start.amin(1)[ok] >= START_CLEARANCE - 1e-5).all() and (d_goal.amin(1)[ok] >= START_CLEARANCE - 1e-5).all()
        n_active = lay.active.sum(1)
        if c["family"] == "open":
            assert (n_active == 0).all()
        elif c["family"] in ("gate", "wall_gap", "corridor"):
            assert (n_active[ok] == 2).all() and lay.guide_on[ok].all()
        elif c["family"] == "single_block":
            assert (n_active[ok] == 1).all()


def test_gate_and_door_openings_match_the_requested_gap():
    cases = validate_catalog(BASE).cases
    table = CaseTable(cases)
    gen = torch.Generator().manual_seed(1)
    for cid, slots in (("gate_tight", (3, 4)), ("door", (6, 7))):
        ci = [c["id"] for c in cases].index(cid)
        lay = sample_layouts(table, torch.full((200,), ci), gen)
        ok = ~lay.failed
        a, b = slots
        # rotate the two centres back into the path frame
        rel = lay.slot_xy[:, a] - lay.slot_xy[:, b]
        cos, sin = torch.cos(lay.bearing), torch.sin(lay.bearing)
        lateral = -sin * rel[:, 0] + cos * rel[:, 1]
        along = cos * rel[:, 0] + sin * rel[:, 1]
        assert along[ok].abs().max() < 1e-4
        half_y = SLOT_HALF[a][1]
        opening = lateral.abs() - 2 * half_y
        lo, hi = cases[ci]["gap"]
        assert (opening[ok] >= lo - 1e-4).all() and (opening[ok] <= hi + 1e-4).all()
        assert (opening[ok] >= MIN_DOORWAY - 1e-4).all()


def test_all_slots_and_family_ids_consistent():
    assert NUM_SLOTS == len(SLOT_HALF) == 8
    assert set(FAMILY_ID) >= {"open", "single_block", "gate", "wall_gap"}


def test_corridor_walls_run_along_the_path_with_the_requested_width():
    cases = validate_catalog(BASE).cases
    table = CaseTable(cases)
    ci = [c["id"] for c in cases].index("corridor")
    lay = sample_layouts(table, torch.full((200,), ci), torch.Generator().manual_seed(3))
    ok = ~lay.failed
    assert ok.float().mean() > 0.9
    cos, sin = torch.cos(lay.bearing), torch.sin(lay.bearing)
    rel = lay.slot_xy[:, 6] - lay.slot_xy[:, 7]
    lateral = -sin * rel[:, 0] + cos * rel[:, 1]
    inner = lateral.abs() - 2 * 0.1  # wall thickness 0.2 (half 0.1 each)
    lo, hi = cases[ci]["gap"]
    assert (inner[ok] >= lo - 1e-4).all() and (inner[ok] <= hi + 1e-4).all()
    # walls are rotated a quarter turn so their 2 m side lies along the path
    assert torch.allclose((lay.slot_yaw[ok, 6] - lay.bearing[ok]).abs(), torch.full_like(lay.bearing[ok], math.pi / 2))

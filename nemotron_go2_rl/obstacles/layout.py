"""Turn Nemotron scenario ranges into concrete obstacle poses (batched torch, no Isaac Sim).

A fixed pool of eight oriented boxes ("slots") exists in every environment:
slots 0-2 pillars (0.3 x 0.3 m), 3-5 boxes (0.6 x 0.6 m), 6-7 wall pieces (0.2 x 2.0 m); all 0.8 m high.
A scenario activates some slots and places them in the *path frame* (x from start to goal, y to the left);
the result is rotated by the goal bearing into the robot-start-relative world frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .geometry import point_distances
from .schema import FAMILIES

SLOT_HALF = [(0.15, 0.15)] * 3 + [(0.30, 0.30)] * 3 + [(0.10, 1.00)] * 2
SLOT_KIND = ["pillar"] * 3 + ["box"] * 3 + ["wall"] * 2
SLOT_HEIGHT = 0.8
NUM_SLOTS = len(SLOT_HALF)
FAMILY_ID = {name: i for i, name in enumerate(FAMILIES)}

START_CLEARANCE = 0.9  # m from start / goal to any obstacle surface
MIN_SPACING = {"scatter": 1.5, "side_clutter": 1.2, "slalom": 0.95}


class CaseTable:
    """Scenario ranges as tensors, indexed by case."""

    def __init__(self, cases: list[dict], device="cpu"):
        t = lambda key: torch.tensor([c[key] for c in cases], dtype=torch.float32, device=device)  # noqa: E731
        self.ids = [c["id"] for c in cases]
        self.family = torch.tensor([FAMILY_ID[c["family"]] for c in cases], device=device)
        self.difficulty = torch.tensor([c["difficulty"] for c in cases], device=device)
        self.weight = torch.tensor([c["weight"] for c in cases], dtype=torch.float32, device=device)
        self.distance = t("goal_distance")
        self.bearing = torch.deg2rad(t("goal_bearing_deg"))
        self.along = t("along")
        self.lateral = t("lateral")
        self.count = t("count")
        self.gap = t("gap")
        self.slot_half = torch.tensor(SLOT_HALF, dtype=torch.float32, device=device)
        self.device = device

    def __len__(self):
        return len(self.ids)


@dataclass
class Layout:
    distance: torch.Tensor  # (n,)
    bearing: torch.Tensor  # (n,) rad, goal direction relative to the (zero-yaw) robot
    slot_xy: torch.Tensor  # (n, S, 2) relative to the robot start, world axes
    slot_yaw: torch.Tensor  # (n, S)
    active: torch.Tensor  # (n, S) bool
    guide_xy: torch.Tensor  # (n, 2) doorway centre (world axes, relative to start)
    guide_on: torch.Tensor  # (n,) bool
    guide_s: torch.Tensor  # (n,) path coordinate of the doorway
    failed: torch.Tensor  # (n,) bool: no valid layout found, obstacles removed


def _uni(r: torch.Tensor, n: int, gen) -> torch.Tensor:
    return r[:, 0] + (r[:, 1] - r[:, 0]) * torch.rand(n, device=r.device, generator=gen)


def _sign(n: int, device, gen) -> torch.Tensor:
    return torch.where(torch.rand(n, device=device, generator=gen) < 0.5, -1.0, 1.0)


def _count(r: torch.Tensor, n: int, gen, lo_min=0, hi_max=3) -> torch.Tensor:
    lo = r[:, 0].round().clamp(lo_min, hi_max)
    hi = r[:, 1].round().clamp(lo_min, hi_max).maximum(lo)
    return (lo + torch.floor(torch.rand(n, device=r.device, generator=gen) * (hi - lo + 1))).minimum(hi).long()


def _draw_path_frame(table: CaseTable, case_idx: torch.Tensor, d: torch.Tensor, gen, count_cap: int = 5):
    """One random layout per env in the path frame. Returns px, py, yaw_off, active, guide_xy, guide_on."""
    n, dev = len(case_idx), table.device
    px = torch.zeros(n, NUM_SLOTS, device=dev)
    py = torch.zeros(n, NUM_SLOTS, device=dev)
    yaw_off = torch.zeros(n, NUM_SLOTS, device=dev)
    active = torch.zeros(n, NUM_SLOTS, dtype=torch.bool, device=dev)
    guide = torch.zeros(n, 2, device=dev)
    guide_on = torch.zeros(n, dtype=torch.bool, device=dev)
    fam = table.family[case_idx]

    for name, fid in FAMILY_ID.items():
        m = (fam == fid).nonzero().flatten()
        if len(m) == 0 or name == "open":
            continue
        k = len(m)
        ci = case_idx[m]
        dd = d[m]
        along = _uni(table.along[ci], k, gen)
        lat = _sign(k, dev, gen) * _uni(table.lateral[ci], k, gen)
        if name == "single_block":
            active[m, 3] = True
            px[m, 3] = dd * along
            py[m, 3] = lat
        elif name in ("scatter", "side_clutter"):
            cnt = _count(table.count[ci], k, gen, hi_max=min(count_cap, 5 if name == "scatter" else 3))
            order = torch.rand(k, 6, device=dev, generator=gen).argsort(dim=1)  # random subset of the 6 pillar/box slots
            for r in range(5):
                on = r < cnt
                slot = order[:, r]
                x_j = dd * _uni(table.along[ci], k, gen)
                y_j = _sign(k, dev, gen) * _uni(table.lateral[ci], k, gen)
                rows = m[on]
                active[rows, slot[on]] = True
                px[rows, slot[on]] = x_j[on]
                py[rows, slot[on]] = y_j[on]
        elif name == "slalom":
            # evenly spread between 1.25 m after the start and 1.25 m before the goal; fewer boxes if too short
            cnt = _count(table.count[ci], k, gen, lo_min=2)
            room = (dd - 2.5).clamp(min=0.0)
            cnt = torch.minimum(cnt, torch.floor(room / MIN_SPACING["slalom"]).long() + 1)
            side0 = _sign(k, dev, gen)
            for j in range(3):
                on = (j < cnt) & (cnt >= 2)
                rows = m[on]
                frac = j / (cnt - 1).clamp(min=1).float()
                active[rows, 3 + j] = True
                px[rows, 3 + j] = (1.25 + room * frac)[on]
                py[rows, 3 + j] = (side0 * (-1.0) ** j * lat.abs())[on]
        elif name == "gate":
            g = _uni(table.gap[ci], k, gen)
            x = dd * along
            active[m, 3] = active[m, 4] = True
            px[m, 3] = px[m, 4] = x
            py[m, 3] = lat + g / 2 + 0.3
            py[m, 4] = lat - g / 2 - 0.3
            guide[m, 0], guide[m, 1], guide_on[m] = x, lat, True
        elif name == "corridor":
            g = _uni(table.gap[ci], k, gen)
            x = dd * along
            active[m, 6] = active[m, 7] = True
            px[m, 6] = px[m, 7] = x
            py[m, 6] = lat + g / 2 + 0.1
            py[m, 7] = lat - g / 2 - 0.1
            yaw_off[m, 6] = yaw_off[m, 7] = math.pi / 2  # long axis along the path
            guide[m, 0], guide[m, 1], guide_on[m] = x, lat, True
        elif name == "wall_gap":
            g = _uni(table.gap[ci], k, gen)
            x = dd * along
            active[m, 6] = active[m, 7] = True
            px[m, 6] = px[m, 7] = x
            py[m, 6] = lat + g / 2 + 1.0
            py[m, 7] = lat - g / 2 - 1.0
            guide[m, 0], guide[m, 1], guide_on[m] = x, lat, True
    return px, py, yaw_off, active, guide, guide_on


def _valid(table: CaseTable, case_idx, d, px, py, yaw_off, active) -> torch.Tensor:
    """Start/goal clearance and minimum spacing between obstacles of scattered families."""
    n = len(case_idx)
    xy = torch.stack((px, py), dim=-1)
    yaw = yaw_off
    zero = torch.zeros(n, 2, device=table.device)
    goal = torch.stack((d, torch.zeros_like(d)), dim=-1)
    ok = point_distances(zero, xy, yaw, table.slot_half, active)[0].amin(dim=1) >= START_CLEARANCE
    ok &= point_distances(goal, xy, yaw, table.slot_half, active)[0].amin(dim=1) >= START_CLEARANCE
    fam = table.family[case_idx]
    min_gap = torch.zeros(n, device=table.device)
    for name, spacing in MIN_SPACING.items():
        min_gap = torch.where(fam == FAMILY_ID[name], spacing, min_gap)
    pair = torch.linalg.norm(xy.unsqueeze(1) - xy.unsqueeze(2), dim=-1)  # (n, S, S)
    both = active.unsqueeze(1) & active.unsqueeze(2) & ~torch.eye(NUM_SLOTS, dtype=torch.bool, device=table.device)
    too_close = both & (pair < min_gap.view(-1, 1, 1))
    need = torch.zeros(n, dtype=torch.long, device=table.device)
    need = torch.where(fam == FAMILY_ID["slalom"], 2, need)
    return ok & ~too_close.flatten(1).any(dim=1) & (active.sum(1) >= need)


def sample_layouts(table: CaseTable, case_idx: torch.Tensor, generator: torch.Generator | None = None, attempts: int = 32) -> Layout:
    """Sample one valid layout per entry of ``case_idx`` (rejection sampling, then give up with no obstacles)."""
    n, dev = len(case_idx), table.device
    d = torch.zeros(n, device=dev)
    b = torch.zeros(n, device=dev)
    px = torch.zeros(n, NUM_SLOTS, device=dev)
    py = torch.zeros_like(px)
    yoff = torch.zeros_like(px)
    active = torch.zeros(n, NUM_SLOTS, dtype=torch.bool, device=dev)
    guide = torch.zeros(n, 2, device=dev)
    guide_on = torch.zeros(n, dtype=torch.bool, device=dev)
    todo = torch.arange(n, device=dev)
    for attempt in range(attempts):
        if len(todo) == 0:
            break
        ci = case_idx[todo]
        cd = _uni(table.distance[ci], len(todo), generator)  # goal distance / bearing are redrawn on every attempt
        cb = _uni(table.bearing[ci], len(todo), generator)
        # a scatter that cannot be arranged is retried with one object fewer every 6 attempts (never fewer than 1)
        cpx, cpy, cyaw, cact, cg, cgon = _draw_path_frame(table, ci, cd, generator, count_cap=max(1, 5 - attempt // 6))
        good = _valid(table, ci, cd, cpx, cpy, cyaw, cact)
        rows = todo[good]
        d[rows], b[rows] = cd[good], cb[good]
        px[rows], py[rows], yoff[rows] = cpx[good], cpy[good], cyaw[good]
        active[rows], guide[rows], guide_on[rows] = cact[good], cg[good], cgon[good]
        todo = todo[~good]
    if len(todo):  # still unsolved: keep a goal but no obstacles
        d[todo] = _uni(table.distance[case_idx[todo]], len(todo), generator)
        b[todo] = _uni(table.bearing[case_idx[todo]], len(todo), generator)
    failed = torch.zeros(n, dtype=torch.bool, device=dev)
    failed[todo] = True

    c, s = torch.cos(b), torch.sin(b)
    slot_xy = torch.stack((px * c[:, None] - py * s[:, None], px * s[:, None] + py * c[:, None]), dim=-1)
    guide_xy = torch.stack((guide[:, 0] * c - guide[:, 1] * s, guide[:, 0] * s + guide[:, 1] * c), dim=-1)
    return Layout(
        distance=d,
        bearing=b,
        slot_xy=slot_xy,
        slot_yaw=b[:, None] + yoff,
        active=active,
        guide_xy=guide_xy,
        guide_on=guide_on,
        guide_s=guide[:, 0].clone(),
        failed=failed,
    )


def goal_world(layout: Layout) -> torch.Tensor:
    """Goal position relative to the start, world axes."""
    return torch.stack((layout.distance * torch.cos(layout.bearing), layout.distance * torch.sin(layout.bearing)), dim=-1)

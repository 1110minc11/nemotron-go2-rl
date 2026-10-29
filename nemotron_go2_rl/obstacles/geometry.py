"""Planar geometry for oriented boxes ("slots"): ray ranges and point distances (torch, batched).

Shapes: N environments, S obstacle slots, R rays. Slot poses are given in one common 2D frame
(world frame relative to the env origin); inactive slots are ignored.
"""

from __future__ import annotations

import torch

INF = 1.0e6


def _to_local(points: torch.Tensor, slot_xy: torch.Tensor, slot_yaw: torch.Tensor) -> torch.Tensor:
    """``points`` (N, ..., 2) minus slot centres (N, S, 2) rotated into each slot frame -> (N, ..., S, 2)."""
    c, s = torch.cos(slot_yaw), torch.sin(slot_yaw)  # (N, S)
    extra = points.dim() - 2  # number of dims between N and the coordinate axis
    shape = (points.shape[0],) + (1,) * extra + (slot_xy.shape[1], 2)
    rel = points.unsqueeze(-2) - slot_xy.reshape(shape)  # (N, ..., S, 2)
    c = c.reshape((c.shape[0],) + (1,) * extra + (c.shape[1],))
    s = s.reshape_as(c)
    # rotate by -yaw: x' = c x + s y ; y' = -s x + c y
    return torch.stack((c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]), dim=-1)


def ray_ranges(
    origins: torch.Tensor,
    directions: torch.Tensor,
    slot_xy: torch.Tensor,
    slot_yaw: torch.Tensor,
    slot_half: torch.Tensor,
    active: torch.Tensor,
    max_range: float,
) -> torch.Tensor:
    """Distance to the first oriented box hit by each ray, ``max_range`` if none. Returns (N, R).

    Args:
        origins: (N, 2) ray origins.
        directions: (N, R, 2) unit directions.
        slot_xy: (N, S, 2) box centres. slot_yaw: (N, S). slot_half: (S, 2) half extents. active: (N, S) bool.
    """
    o = _to_local(origins, slot_xy, slot_yaw)  # (N, S, 2)
    c, s = torch.cos(slot_yaw), torch.sin(slot_yaw)  # (N, S)
    dx = directions[..., 0].unsqueeze(-1)  # (N, R, 1)
    dy = directions[..., 1].unsqueeze(-1)
    d_loc = torch.stack((c.unsqueeze(1) * dx + s.unsqueeze(1) * dy, -s.unsqueeze(1) * dx + c.unsqueeze(1) * dy), dim=-1)
    # (N, R, S, 2)
    half = slot_half.reshape(1, 1, -1, 2)
    o = o.unsqueeze(1)  # (N, 1, S, 2)
    safe = torch.where(d_loc.abs() > 1e-7, d_loc, torch.full_like(d_loc, 1e-7))
    t0 = (-half - o) / safe
    t1 = (half - o) / safe
    t_near = torch.minimum(t0, t1).amax(dim=-1)  # (N, R, S)
    t_far = torch.maximum(t0, t1).amin(dim=-1)
    hit = (t_far >= t_near.clamp(min=0.0)) & (t_near >= 0.0) & (t_near <= max_range) & active.unsqueeze(1)
    t = torch.where(hit, t_near, torch.full_like(t_near, max_range))
    return t.amin(dim=-1)


def point_distances(
    points: torch.Tensor, slot_xy: torch.Tensor, slot_yaw: torch.Tensor, slot_half: torch.Tensor, active: torch.Tensor
):
    """Euclidean distance from each point (N, 2) to every box surface -> (N, S), plus closest point (N, S, 2).

    Inactive slots return ``INF``. A point inside a box has distance 0.
    """
    p = _to_local(points, slot_xy, slot_yaw)  # (N, S, 2) in each slot frame
    half = slot_half.reshape(1, -1, 2)
    outside = (p.abs() - half).clamp(min=0.0)
    dist = torch.linalg.norm(outside, dim=-1)
    dist = torch.where(active, dist, torch.full_like(dist, INF))
    closest_local = torch.maximum(torch.minimum(p, half), -half)
    c, s = torch.cos(slot_yaw), torch.sin(slot_yaw)
    closest = torch.stack(
        (c * closest_local[..., 0] - s * closest_local[..., 1], s * closest_local[..., 0] + c * closest_local[..., 1]),
        dim=-1,
    ) + slot_xy
    return dist, closest

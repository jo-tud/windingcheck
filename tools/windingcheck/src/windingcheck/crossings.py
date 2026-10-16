"""Read the organisers' ``winding_inference`` ray crossings (villa ``export_spiral_supervision``).

Format: per shard, rays with ``ray_origin_zyx`` and ``ray_step_zyx``; crossing ``k`` of ray ``r``
sits at ``origin + crossing_t[k] * step`` and carries an integer ``crossing_level``, relative
inside the ray. ``crossing_offsets`` (length rays + 1) delimits each ray's crossings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chains import Chains


@dataclass
class RayStore:
    """All crossings of all rays, in file order (each ray's crossings contiguous, by ``t``)."""
    xyz: np.ndarray        # (N, 3) float64, voxels, x y z
    level: np.ndarray      # (N,) int64, relative inside the ray
    t: np.ndarray          # (N,) float32, position along the ray, samples
    ray: np.ndarray        # (N,) int64, global ray id
    ray_dir: np.ndarray    # (R, 3) float64, unit ray direction, x y z
    ray_length: int        # samples per ray

    def __len__(self) -> int:
        return len(self.level)

    def chains(self) -> Chains:
        return Chains(self.xyz, self.level.astype(np.float64), self.ray,
                      [f"ray{u}" for u in range(len(self.ray_dir))])


def load_ray_store(root, *, max_shards=None) -> RayStore:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    xyz, level, t, ray, dirs = [], [], [], [], []
    base = 0
    for i, shard in enumerate(manifest["shards"]):
        if max_shards is not None and i >= max_shards:
            break
        d = root / shard.get("name", f"shard_{i}")
        origin = np.load(d / "ray_origin_zyx.npy")
        step = np.load(d / "ray_step_zyx.npy")
        ti = np.load(d / "crossing_t.npy")
        offsets = np.load(d / "crossing_offsets.npy").astype(np.int64)
        n_rays = len(origin)
        r = np.repeat(np.arange(n_rays, dtype=np.int64), np.diff(offsets[: n_rays + 1]))
        zyx = origin[r].astype(np.float64) + ti[:, None].astype(np.float64) * step[r]
        xyz.append(zyx[:, ::-1])
        level.append(np.load(d / "crossing_level.npy").astype(np.int64))
        t.append(ti)
        ray.append(r + base)
        sd = step[:, ::-1].astype(np.float64)
        dirs.append(sd / np.linalg.norm(sd, axis=1, keepdims=True))
        base += n_rays
    length = int(manifest.get("anchor_sample", 191)) * 2 + 1
    return RayStore(np.concatenate(xyz), np.concatenate(level), np.concatenate(t),
                    np.concatenate(ray), np.concatenate(dirs), length)


def load_crossings(root, *, bbox_xyz=None, max_shards=None) -> Chains:
    """Load all shards as Chains (one chain per ray), coordinates converted to xyz.

    ``bbox_xyz`` = ((x0, y0, z0), (x1, y1, z1)) keeps only crossings inside the box.
    """
    store = load_ray_store(root, max_shards=max_shards)
    keep = np.ones(len(store), bool)
    if bbox_xyz is not None:
        lo, hi = np.asarray(bbox_xyz[0]), np.asarray(bbox_xyz[1])
        keep = ((store.xyz >= lo) & (store.xyz <= hi)).all(1)
    uniq, inv = np.unique(store.ray[keep], return_inverse=True)
    return Chains(store.xyz[keep], store.level[keep].astype(np.float64), inv.astype(np.int64),
                  [f"ray{u}" for u in uniq])

"""Fetch the public human winding annotations (small JSON files, a few MB)."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

BASE = "https://dl.ash2txt.org/datasets/spiral_datasets"
FILES = ("relative_windings.json", "same_windings.json", "abs_winding.json", "umbilicus.json")
CROSSING_FILES = ("manifest.json", "crossing_level.npy", "crossing_offsets.npy", "crossing_t.npy",
                  "ray_origin_zyx.npy", "ray_step_zyx.npy", "seed_winding.npy")


def default_root() -> Path:
    env = os.environ.get("WINDINGCHECK_DATA")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "tools" / "windingcheck").is_dir():
            return parent / "data" / "spiral_datasets"
    return Path.cwd() / "data" / "spiral_datasets"


def _get(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return
        except OSError:
            if attempt == 4:
                raise


def fetch(scroll="PHercParis4", root=None, *, crossings=False) -> Path:
    root = Path(root) if root else default_root()
    d = root / scroll
    for f in FILES:
        _get(f"{BASE}/{scroll}/{f}", d / f)
    if crossings:
        import json
        wi = d / "winding_inference"
        _get(f"{BASE}/{scroll}/winding_inference/manifest.json", wi / "manifest.json")
        for sh in json.loads((wi / "manifest.json").read_text())["shards"]:
            name = sh.get("name")
            for f in CROSSING_FILES:
                _get(f"{BASE}/{scroll}/winding_inference/{name}/{f}", wi / name / f)
    return d

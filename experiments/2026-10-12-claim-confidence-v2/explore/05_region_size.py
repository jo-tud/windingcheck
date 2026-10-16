import sys, numpy as np
sys.path.insert(0, "tools/windingcheck/src")
from windingcheck import data
from windingcheck.chains import load_pcl
from windingcheck.crossings import load_ray_store
d = data.fetch("PHercParis4", crossings=True)
h = load_pcl(d / "relative_windings.json")
print("clicks", len(h), "chains", h.n_chains)
lo, hi = h.xyz.min(0), h.xyz.max(0); print("bbox xyz", lo.round(), hi.round())
print("z pct", np.percentile(h.xyz[:,2],[0,10,50,90,100]).round())
store = load_ray_store(d / "winding_inference")
m = ((store.xyz >= lo-50) & (store.xyz <= hi+50)).all(1)
print("crossings in bbox", m.sum(), "of", len(store))
# near clicks: crossings within 60 vox of any click
from scipy.spatial import cKDTree
t = cKDTree(h.xyz); dd,_ = t.query(store.xyz, distance_upper_bound=60)
near = np.isfinite(dd); rays = np.unique(store.ray[near])
inr = np.isin(store.ray, rays)
print("crossings within 60 of a click", near.sum(), "rays", len(rays), "crossings on those rays", inr.sum())
# chunks touched by steps on those rays at level 2
c = np.unique((store.xyz[inr][:, ::-1] // 128).astype(int), axis=0); print("chunks (crossing positions only)", len(c), "~GB", round(len(c)*2/1024,1))
call = np.unique((store.xyz[:, ::-1] // 128).astype(int), axis=0); print("chunks full set", len(call), "~GB", round(len(call)*2/1024,1))

import sys, time, numpy as np
sys.path.insert(0, "tools/windingcheck/src")
from windingcheck import data
from windingcheck.crossings import load_ray_store
from windingcheck.ct import CTSampler
d = data.fetch("PHercParis4", crossings=True)
store = load_ray_store(d / "winding_inference")
k = np.flatnonzero(store.ray[1:]==store.ray[:-1])[:1000]
print("step norm / dt:", np.median(np.linalg.norm(store.xyz[k+1]-store.xyz[k],axis=1)/(store.t[k+1]-store.t[k])))
z = np.load(sys.argv[1]); ca, cb, pd = z["ca"], z["cb"], z["pd"]
lo, hi = np.minimum(ca,cb), np.maximum(ca,cb)
rng = np.random.default_rng(0)
sel = np.r_[np.flatnonzero(pd==2), rng.choice(np.flatnonzero(pd==1), 300, replace=False)]
ct = CTSampler()
T = np.linspace(-0.25, 1.25, 61)  # fraction of the span lo..hi
pts, owner = [], []
for i in sel:
    a, b = store.xyz[lo[i]], store.xyz[hi[i]]
    for f in T: pts.append(a + f*(b-a))
pts = np.array(pts)[:, ::-1]
t0=time.time(); v = ct.sample(pts).reshape(len(sel), len(T)); print("sampled", len(pts), "in", round(time.time()-t0,1),"s", ct.stats())
# crossing densities: max within +-1.5 voxels along ray
def cross_ct(c):
    p = np.concatenate([store.xyz[c] + o*store.ray_dir[store.ray[c]] for o in (-1.5,-0.75,0,0.75,1.5)])
    return ct.sample(p[:, ::-1]).reshape(5, len(c)).max(0)
m2 = pd[sel]==2
ends = cross_ct(np.r_[lo[sel], hi[sel]])
mid = cross_ct(lo[sel][m2]+1)
print("CT at matched ends: pct", np.percentile(ends,[10,25,50,75,90]).round(0))
print("CT at middle crossing (pred=2):", np.percentile(mid,[10,25,50,75,90]).round(0))
# min along the span interior
inner = (T>0.15)&(T<0.85)
print("min CT inside span, pred=1:", np.percentile(v[~m2][:,inner].min(1),[10,25,50,75,90]).round(0))
print("min CT inside span, pred=2:", np.percentile(v[m2][:,inner].min(1),[10,25,50,75,90]).round(0))
print("CT at span midpoint, pred=1:", np.percentile(v[~m2][:,30],[10,25,50,75,90]).round(0))
print("CT at span midpoint, pred=2:", np.percentile(v[m2][:,30],[10,25,50,75,90]).round(0))
np.savez(sys.argv[2], v=v, T=T, m2=m2, mid=mid, ends=ends)

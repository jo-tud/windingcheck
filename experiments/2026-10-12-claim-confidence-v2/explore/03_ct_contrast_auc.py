import sys, time, numpy as np
from scipy.stats import rankdata
sys.path.insert(0, "tools/windingcheck/src")
from windingcheck import data
from windingcheck.chains import load_pcl
from windingcheck.crossings import load_ray_store
from windingcheck.score import build_truth, score_chains
from windingcheck.confidence import pair_steps, StepModel, step_features, step_index
from windingcheck.prior import Umbilicus
from windingcheck.ct import CTSampler
def auc(s, y):
    y=np.asarray(y,bool); r=rankdata(s); return float((r[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
d = data.fetch("PHercParis4", crossings=True)
human = load_pcl(d / "relative_windings.json")
ids = np.arange(build_truth(human).chains.n_chains)
truth = build_truth(human.subset(ids[ids % 2 == 0]))
store = load_ray_store(d / "winding_inference")
res, t = score_chains(truth, store.chains(), return_tests=True)
kc = np.abs(t["cb"]-t["ca"])  # claimed steps
exact = t["pred_dw"]==t["true_dw"]
print("even tests", len(kc), "claimed k dist", np.bincount(np.minimum(kc,8)))
owner, ks = pair_steps(t["ca"], t["cb"])
uk = np.unique(ks)
print("unique steps", len(uk))
ct = CTSampler()
F = np.linspace(0.15, 0.85, 8)
A, B = store.xyz[uk], store.xyz[uk+1]
pts = (A[:,None,:] + F[None,:,None]*(B-A)[:,None,:]).reshape(-1,3)
t0=time.time()
inner = ct.sample(pts[:, ::-1]).reshape(len(uk), len(F))
def cross_ct(c):
    p = np.concatenate([store.xyz[c] + o*store.ray_dir[store.ray[c]] for o in (-1.5,-0.75,0,0.75,1.5)])
    return ct.sample(p[:, ::-1]).reshape(5, len(c)).max(0)
ca_ct, cb_ct = cross_ct(uk), cross_ct(uk+1)
print("CT sample secs", round(time.time()-t0,1), ct.stats())
valley = inner.min(1); ends = 0.5*(ca_ct+cb_ct)
contrast = (np.minimum(ca_ct,cb_ct) - valley)/np.maximum(np.minimum(ca_ct,cb_ct),1)
pos = np.searchsorted(uk, ks)
n = len(kc)
cmin = np.full(n, np.inf); np.minimum.at(cmin, owner, contrast[pos])
vmax = np.full(n, -np.inf); np.maximum.at(vmax, owner, valley[pos])
# intermediate crossing densities: steps beyond the first have a start crossing that is intermediate
# v1 confidence
umb = Umbilicus(d / "umbilicus.json")
m = StepModel.load("tools/windingcheck/src/windingcheck/models/paris4_even_chains.json")
X, s = step_features(store, umb, uk)
p = m.prob(X); prod = np.exp(np.bincount(owner, weights=np.log(p[pos]), minlength=n))
for lo, hi in ((2,7),(2,3),(4,7)):
    sel = (kc>=lo)&(kc<=hi)
    print(f"claims k={lo}-{hi}: n={sel.sum()} exact={exact[sel].mean():.3f}  AUC v1 prod={auc(prod[sel],exact[sel]):.3f}  min contrast={auc(cmin[sel],exact[sel]):.3f}  -max valley={auc(-vmax[sel],exact[sel]):.3f}")
sel = kc==1
print("k=1: n", sel.sum(), "exact", exact[sel].mean())
np.savez(sys.argv[1], uk=uk, inner=inner, ca_ct=ca_ct, cb_ct=cb_ct)

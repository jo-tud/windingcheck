import sys, time, numpy as np
from scipy.stats import rankdata
from scipy.spatial import cKDTree
sys.path.insert(0, "tools/windingcheck/src")
from windingcheck import data
from windingcheck.chains import load_pcl
from windingcheck.crossings import load_ray_store
from windingcheck.score import build_truth, score_chains
from windingcheck.confidence import pair_steps, StepModel, step_features
from windingcheck.prior import Umbilicus
def auc(s, y):
    y=np.asarray(y,bool); r=rankdata(s); return float((r[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
d = data.fetch("PHercParis4", crossings=True)
human = load_pcl(d / "relative_windings.json")
ids = np.arange(build_truth(human).chains.n_chains)
truth = build_truth(human.subset(ids[ids % 2 == 0]))
store = load_ray_store(d / "winding_inference")
res, t = score_chains(truth, store.chains(), return_tests=True)
ca, cb = t["ca"], t["cb"]; kc = np.abs(cb-ca); exact = t["pred_dw"]==t["true_dw"]
t0=time.time(); tree = cKDTree(store.xyz); print("tree", round(time.time()-t0,1))
umb = Umbilicus(d / "umbilicus.json")
for R, TN in ((25, 4.0), (40, 5.0)):
    agree = np.zeros(len(ca)); supp = np.zeros(len(ca)); mean_dev = np.zeros(len(ca))
    na = tree.query_ball_point(store.xyz[ca], R); nb = tree.query_ball_point(store.xyz[cb], R)
    for i in range(len(ca)):
        r0 = store.ray[ca[i]]; dvec = store.ray_dir[r0]
        def near(lst, c):
            j = np.array(lst); j = j[store.ray[j] != r0]
            dn = np.abs((store.xyz[j]-store.xyz[c]) @ dvec)
            j, dn = j[dn < TN], dn[dn < TN]
            # one per ray: nearest along normal
            if len(j) == 0: return {}
            o = np.lexsort((dn, store.ray[j])); j = j[o]; r = store.ray[j]
            f = np.r_[True, r[1:]!=r[:-1]]; return dict(zip(r[f], j[f]))
        A, B = near(na[i], ca[i]), near(nb[i], cb[i])
        common = [r for r in A if r in B]
        if not common: agree[i]=0.5; continue
        dl = np.array([abs(store.level[B[r]]-store.level[A[r]]) for r in common])
        supp[i] = len(dl); agree[i] = np.mean(dl == kc[i])
    for lo, hi in ((1,1),(2,7)):
        s = (kc>=lo)&(kc<=hi)
        print(f"R={R} TN={TN} k={lo}-{hi}: n={s.sum()} supported={np.mean(supp[s]>0):.2f} median support={np.median(supp[s]):.0f}  AUC agree={auc(agree[s]+1e-3*supp[s],exact[s]) if 0<exact[s].mean()<1 else None}")
    np.save(f"{sys.argv[1]}_R{R}.npy", np.c_[agree, supp])
s = (kc>=2)&(kc<=7)
m = StepModel.load("tools/windingcheck/src/windingcheck/models/paris4_even_chains.json")
owner, ks = pair_steps(ca, cb); uk = np.unique(ks); pos = np.searchsorted(uk, ks)
X, _ = step_features(store, umb, uk); p = m.prob(X)
prod = np.exp(np.bincount(owner, weights=np.log(p[pos]), minlength=len(ca)))
print("v1 prod AUC k2-7", auc(prod[s], exact[s]))
ag = np.load(f"{sys.argv[1]}_R25.npy")[:,0]
print("combined rank-sum AUC", auc(rankdata(prod[s])+rankdata(ag[s]), exact[s]))

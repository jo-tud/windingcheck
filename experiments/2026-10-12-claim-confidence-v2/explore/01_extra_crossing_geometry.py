import sys, numpy as np
sys.path.insert(0, "tools/windingcheck/src")
from windingcheck import data
from windingcheck.chains import load_pcl
from windingcheck.crossings import load_ray_store
from windingcheck.score import build_truth, score_chains
from windingcheck.confidence import step_index, _ray_median_step
d = data.fetch("PHercParis4", crossings=True)
human = load_pcl(d / "relative_windings.json")
ids = np.arange(build_truth(human).chains.n_chains)
truth = build_truth(human.subset(ids[ids % 2 == 0]))
store = load_ray_store(d / "winding_inference")
res, t = score_chains(truth, store.chains(), return_tests=True)
ch = truth.chains.chain
adj = (t["b"] == t["a"] + 1) & (ch[t["a"]] == ch[t["b"]]) & (t["true_dw"] == 1)
k = step_index(store); s = np.linalg.norm(store.xyz[k+1]-store.xyz[k],axis=1); med = _ray_median_step(store,k,s)
medc = np.zeros(len(store)); medc[k] = med
ca, cb = t["ca"][adj], t["cb"][adj]; pd = np.abs(t["pred_dw"][adj])
span = np.linalg.norm(store.xyz[cb]-store.xyz[ca],axis=1)
lo = np.minimum(ca,cb)
human_gap = np.linalg.norm(truth.chains.xyz[t["b"][adj]]-truth.chains.xyz[t["a"][adj]],axis=1)
for v in (0,1,2,3):
    m = pd==v
    if m.sum()==0: continue
    print(f"pred={v}: n={m.sum()}  span/ray_median: {np.percentile(span[m]/medc[lo[m]],[10,25,50,75,90]).round(2)}  human click gap: {np.percentile(human_gap[m],[25,50,75]).round(1)} ray median: {np.median(medc[lo[m]]).round(1)}")
# for pred=2: ratio of the two sub-steps
m = pd==2
s1 = np.linalg.norm(store.xyz[lo[m]+1]-store.xyz[lo[m]],axis=1); s2=np.linalg.norm(store.xyz[lo[m]+2]-store.xyz[lo[m]+1],axis=1)
print("pred=2 substep min/max ratio pct:", np.percentile(np.minimum(s1,s2)/np.maximum(s1,s2),[10,25,50,75,90]).round(2))
np.savez(sys.argv[1], ca=ca, cb=cb, pd=pd, a=t["a"][adj], b=t["b"][adj])

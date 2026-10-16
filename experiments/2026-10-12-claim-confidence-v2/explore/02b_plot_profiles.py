import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
z = np.load(sys.argv[1]); v, T, m2 = z["v"], z["T"], z["m2"]
fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for a, m, title in ((ax[0], ~m2, "ray counts 1 (agrees with human)"), (ax[1], m2, "ray counts 2 where human counts 1")):
    vv = v[m]
    for row in vv[:25]: a.plot(T, row, color="0.75", lw=0.6)
    a.plot(T, np.median(vv, 0), color="#2a78d6", lw=2.5, label="median")
    a.axvline(0, color="k", ls="--"); a.axvline(1, color="k", ls="--"); a.set_title(f"{title}, n={len(vv)}"); a.set_ylabel("CT")
ax[1].set_xlabel("position along ray, 0 and 1 = the two crossings matched to human clicks")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110)

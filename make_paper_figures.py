#!/usr/bin/env python3
"""Regenerate the manuscript's result figures (Figures 2 and 3) from results/.

  python make_paper_figures.py            # reads results/, writes figures/
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
MODELS = ["TF-IDF", "General SBERT", "SBERT-EduAdapt"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]

# Figure 2: similarity by gold label (pooled test set)
t = pd.read_csv(os.path.join(RES, "all", "test_predictions_notext.csv"))
fig, ax = plt.subplots(figsize=(3.3, 2.6))
labels, w = ["C1", "C2", "C3", "C4"], 0.24
for j, (m, c) in enumerate(zip(MODELS, COLORS)):
    data = [t.loc[t.label == l, f"sim_{m}"] for l in labels]
    pos = np.arange(4) + (j - 1) * (w + 0.03)
    b = ax.boxplot(data, positions=pos, widths=w, showfliers=False, patch_artist=True,
                   medianprops=dict(color="white", lw=1.2),
                   whiskerprops=dict(color=c, lw=1), capprops=dict(color=c, lw=1))
    for p in b["boxes"]:
        p.set(facecolor=c, edgecolor=c)
    ax.plot([], [], "s", color=c, label=m)
ax.set_xticks(range(4), ["C1\ncorrect", "C2\npartial", "C3\nmisconc.", "C4\nirrelevant"])
ax.set_ylabel("Reference–student cosine similarity")
ax.grid(axis="y", color="#e5e5e5", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
          handletextpad=0.3, columnspacing=0.8)
fig.savefig(os.path.join(OUT, "fig2_similarity_by_label.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 3: macro-F1 and QWK by split with 95% CIs
splits = [("test_ua", "UA\n(n=540)"), ("test_uq", "UQ\n(n=733)"), ("test_ud", "UD\n(n=4,562)")]
fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.9), sharex=True)
for ax, (metric, title) in zip(axes, [("bin_macro_f1", "Binary macro-F1"), ("ord_qwk", "Ordinal QWK")]):
    for j, (m, c, mk) in enumerate(zip(MODELS, COLORS, MARKERS)):
        v, lo, hi = [], [], []
        for d, _ in splits:
            r = pd.read_csv(os.path.join(RES, d, "metrics_with_ci.csv"))
            r = r[(r.model == m) & (r.metric == metric)].iloc[0]
            v.append(r.value); lo.append(r.value - r.ci_low); hi.append(r.ci_high - r.value)
        ax.errorbar(np.arange(3) + (j - 1) * 0.18, v, yerr=[lo, hi], fmt=mk, color=c, ms=5,
                    capsize=2, lw=1, label=m)
    ax.set_ylabel(title)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
axes[1].set_xticks(range(3), [s for _, s in splits])
axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3,
               handletextpad=0.2, columnspacing=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_by_split.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Figures written to", OUT)

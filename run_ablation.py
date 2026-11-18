#!/usr/bin/env python3
"""
SEAR Paper 1 — held-out ablation: TF-IDF vs general SBERT vs SBERT-EduAdapt.

What it does
------------
1. Loads the semantic-adaptation corpus (reference answer, student answer,
   normalized label C1-C4, split).
2. Checks for train/test leakage against the SBERT training-pairs file.
3. Computes reference-student cosine similarity on the TEST split with:
     - TF-IDF (vectorizer fitted on non-test text only)
     - general Sentence-BERT (default: sentence-transformers/all-mpnet-base-v2)
     - SBERT-EduAdapt (your fine-tuned model folder)
4. Converts similarity into labels for two tasks:
     - Binary:  C1+C2 (acceptable) vs C3+C4 (not acceptable)
     - Ordinal: C1 / C2 / C3+C4 (3 levels, matching the 1.0 / 0.7 / 0.0 targets)
   Thresholds are tuned WITHOUT looking at the instances they are applied to
   (default: 5-fold cross-fitting inside the test split), so no model gets an
   optimistic threshold and the fine-tuned model is not favoured by having
   seen the training split.
5. Reports accuracy, macro-F1, Cohen's kappa, QWK (ordinal), ROC-AUC and
   Spearman rho, each with a 95% bootstrap CI; paired McNemar tests and paired
   bootstrap differences between models; confusion matrices; figures; and a
   ready-to-paste Markdown/CSV version of Table 3.

Note: the relation/direction validation stage ("Full SEAR") needs
concept-map triples and a same-domain expert graph. Short-answer pairs have no
relation slot, so Full SEAR is NOT evaluated here and must not be reported
from this script.

Usage (on the computer that has the data and the model)
------------------------------------------------------
  pip install sentence-transformers scikit-learn scipy pandas matplotlib
  python run_ablation.py \
      --data semantic_corpus.csv \
      --eduadapt ./SBERT-EduAdapt-v1 \
      --training-pairs SBERT_training_pairs.csv \
      --out results/

Run `python run_ablation.py --help` for column-name options.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

SEED = 42

# --------------------------------------------------------------------------
# Label handling
# --------------------------------------------------------------------------
LABEL_ALIASES = {
    "C1": "C1", "CORRECT": "C1",
    "C2": "C2", "PARTIAL": "C2", "PARTIALLY_CORRECT": "C2",
    "PARTIALLY_CORRECT_INCOMPLETE": "C2",
    "C3": "C3", "MISCONCEPTION": "C3", "CONTRADICTORY": "C3", "INCORRECT": "C3",
    "C4": "C4", "IRRELEVANT": "C4", "NON_DOMAIN": "C4",
}


def normalize_label(x):
    s = str(x).strip().upper().replace(" ", "_").replace("-", "_")
    for key in (s, s.split("_")[0]):
        if key in LABEL_ALIASES:
            return LABEL_ALIASES[key]
    raise ValueError(f"Unrecognised label: {x!r}. Edit LABEL_ALIASES in the script.")


BINARY = {"C1": 1, "C2": 1, "C3": 0, "C4": 0}       # 1 = acceptable
ORDINAL = {"C1": 2, "C2": 1, "C3": 0, "C4": 0}      # 2 correct, 1 partial, 0 wrong/irrelevant


def clean(t):
    return " ".join(str(t).lower().split())


# --------------------------------------------------------------------------
# Similarity back-ends
# --------------------------------------------------------------------------
def tfidf_similarity(fit_texts, refs, studs):
    vec = TfidfVectorizer(lowercase=True)
    vec.fit(fit_texts)
    a = vec.transform(refs)
    b = vec.transform(studs)
    # rows are L2-normalised by default -> dot product = cosine
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def sbert_similarity(model_spec, refs, studs, cache_dir, batch_size):
    """model_spec: HF name or local folder. 'mock:<noise>' = fake encoder for testing."""
    uniq = sorted(set(refs) | set(studs))
    idx = {t: i for i, t in enumerate(uniq)}
    key = hashlib.sha1((model_spec + "\n".join(uniq)).encode()).hexdigest()[:16]
    cache = os.path.join(cache_dir, f"emb_{key}.npy")
    if os.path.exists(cache):
        emb = np.load(cache)
    elif model_spec.startswith("mock:"):
        emb = _mock_encode(uniq, float(model_spec.split(":", 1)[1]))
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_spec)
        emb = model.encode(uniq, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    np.save(cache, emb)
    a = emb[[idx[t] for t in refs]]
    b = emb[[idx[t] for t in studs]]
    return np.sum(a * b, axis=1)


def _mock_encode(texts, noise):
    """Deterministic stand-in encoder (char n-gram TF-IDF + SVD + noise). Testing only."""
    from sklearn.decomposition import TruncatedSVD
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit_transform(texts)
    k = max(2, min(64, v.shape[1] - 1, len(texts) - 1))
    z = TruncatedSVD(k, random_state=SEED).fit_transform(v)
    z = z + np.random.default_rng(SEED).normal(0, noise, z.shape)
    return z / np.linalg.norm(z, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Threshold tuning
# --------------------------------------------------------------------------
def tune_binary(sim, y):
    cands = np.unique(np.quantile(sim, np.linspace(0.01, 0.99, 197)))
    best = max(cands, key=lambda t: f1_score(y, (sim >= t).astype(int),
                                             average="macro", zero_division=0))
    return float(best)


def tune_ordinal(sim, y):
    cands = np.unique(np.quantile(sim, np.linspace(0.02, 0.98, 49)))
    best, best_s = (cands[0], cands[-1]), -np.inf
    for i, lo in enumerate(cands):
        for hi in cands[i:]:
            p = np.digitize(sim, [lo, hi])
            s = cohen_kappa_score(y, p, weights="quadratic")
            if s > best_s:
                best, best_s = (lo, hi), s
    return float(best[0]), float(best[1])


def predict_with_thresholds(sim_test, y_bin, y_ord, mode, sim_tune=None,
                            yb_tune=None, yo_tune=None, folds=5):
    """Returns binary preds, ordinal preds, and the thresholds used."""
    if mode in ("train", "val"):
        tb = tune_binary(sim_tune, yb_tune)
        lo, hi = tune_ordinal(sim_tune, yo_tune)
        return ((sim_test >= tb).astype(int), np.digitize(sim_test, [lo, hi]),
                {"binary": [tb], "ordinal": [[lo, hi]]})
    # cross-fitting inside the test split
    pb = np.zeros(len(sim_test), int)
    po = np.zeros(len(sim_test), int)
    used = {"binary": [], "ordinal": []}
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    for tr, te in skf.split(sim_test, y_ord):
        tb = tune_binary(sim_test[tr], y_bin[tr])
        lo, hi = tune_ordinal(sim_test[tr], y_ord[tr])
        pb[te] = (sim_test[te] >= tb).astype(int)
        po[te] = np.digitize(sim_test[te], [lo, hi])
        used["binary"].append(tb)
        used["ordinal"].append([lo, hi])
    return pb, po, used


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metric_set(sim, yb, yo, pb, po):
    m = {
        "bin_accuracy": accuracy_score(yb, pb),
        "bin_macro_f1": f1_score(yb, pb, average="macro", zero_division=0),
        "bin_kappa": cohen_kappa_score(yb, pb),
        "ord_accuracy": accuracy_score(yo, po),
        "ord_macro_f1": f1_score(yo, po, average="macro", zero_division=0),
        "ord_kappa": cohen_kappa_score(yo, po),
        "ord_qwk": cohen_kappa_score(yo, po, weights="quadratic"),
        "spearman_rho": stats.spearmanr(sim, yo).correlation,
    }
    m["roc_auc"] = roc_auc_score(yb, sim) if len(np.unique(yb)) > 1 else np.nan
    return m


def bootstrap_ci(sim, yb, yo, pb, po, n_boot, rng):
    n = len(sim)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(yb[i])) < 2:
            continue
        draws.append(metric_set(sim[i], yb[i], yo[i], pb[i], po[i]))
    df = pd.DataFrame(draws)
    return df.quantile(0.025), df.quantile(0.975)


def mcnemar(y, p1, p2):
    a = (p1 == y) & (p2 != y)
    b = (p1 != y) & (p2 == y)
    n01, n10 = int(a.sum()), int(b.sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue  # exact
    return n01, n10, float(p)


def paired_boot_diff(y, p1, p2, n_boot, rng, metric):
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(metric(y[i], p2[i]) - metric(y[i], p1[i]))
    d = np.array(d)
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(test, sims, preds, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(sims)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    order = ["C1", "C2", "C3", "C4"]
    for ax, name in zip(axes, names):
        data = [sims[name][test["label"].values == c] for c in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Gold label")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Cosine similarity")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_similarity_by_label.png"), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.3))
    axes = np.atleast_1d(axes)
    yo = test["y_ord"].values
    for ax, name in zip(axes, names):
        cm = confusion_matrix(yo, preds[name]["ord"], labels=[2, 1, 0])
        ax.imshow(cm, cmap="Blues")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, cm[r, c], ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=9)
        ticks = ["C1", "C2", "C3/C4"]
        ax.set_xticks(range(3), ticks)
        ax.set_yticks(range(3), ticks)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_confusion_ordinal.png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with reference, student, label, split")
    ap.add_argument("--ref-col", default="reference_answer")
    ap.add_argument("--stud-col", default="student_answer")
    ap.add_argument("--label-col", default="normalized_label")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--test-splits", default="test",
                    help="Comma-separated split values treated as test "
                         "(e.g. test_ua,test_uq,test_ud)")
    ap.add_argument("--val-splits", default="", help="Split values used for threshold tuning "
                                                     "when --threshold-source val")
    ap.add_argument("--base-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--eduadapt", required=True, help="Folder of the fine-tuned model")
    ap.add_argument("--training-pairs", default="",
                    help="CSV used to fine-tune EduAdapt (for the leakage check)")
    ap.add_argument("--tp-ref-col", default="reference_clean")
    ap.add_argument("--tp-stud-col", default="student_clean")
    ap.add_argument("--threshold-source", choices=["cv-test", "train", "val"], default="cv-test")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="results")
    ap.add_argument("--allow-leakage", action="store_true",
                    help="Continue even if test pairs appear in the training pairs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache_dir = os.path.join(args.out, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # ---- load
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    df = pd.read_csv(args.data)
    missing = [c for c in (args.ref_col, args.stud_col, args.label_col, args.split_col)
               if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns {missing}. Available: {list(df.columns)}")
    df = df.dropna(subset=[args.ref_col, args.stud_col, args.label_col]).copy()
    df["ref"] = df[args.ref_col].map(clean)
    df["stud"] = df[args.stud_col].map(clean)
    df["label"] = df[args.label_col].map(normalize_label)
    df["y_bin"] = df["label"].map(BINARY)
    df["y_ord"] = df["label"].map(ORDINAL)
    df["split_n"] = df[args.split_col].astype(str).str.strip().str.lower()

    test_vals = [s.strip().lower() for s in args.test_splits.split(",") if s.strip()]
    val_vals = [s.strip().lower() for s in args.val_splits.split(",") if s.strip()]
    is_test = df["split_n"].isin(test_vals)
    if is_test.sum() == 0:
        sys.exit(f"No rows with split in {test_vals}. Split values present: "
                 f"{sorted(df['split_n'].unique())}")
    test = df[is_test].reset_index(drop=True)
    nontest = df[~is_test]
    print(f"Rows: total={len(df)}  test={len(test)}  non-test={len(nontest)}")
    print("Test label counts:", test["label"].value_counts().sort_index().to_dict())

    # ---- leakage check
    leak = {"checked": False}
    if args.training_pairs:
        tp = pd.read_csv(args.training_pairs)
        if args.tp_ref_col in tp.columns and args.tp_stud_col in tp.columns:
            tp_pairs = set(zip(tp[args.tp_ref_col].map(clean), tp[args.tp_stud_col].map(clean)))
            overlap = sum((r, s) in tp_pairs for r, s in zip(test["ref"], test["stud"]))
            leak = {"checked": True, "training_pairs": len(tp_pairs),
                    "test_pairs_found_in_training": int(overlap),
                    "share": overlap / len(test)}
            print(f"Leakage check: {overlap}/{len(test)} test pairs appear in training pairs")
            if overlap > 0 and not args.allow_leakage:
                sys.exit("STOP: test pairs were used to fine-tune EduAdapt. Retrain EduAdapt on "
                         "the non-test split only (or rerun with --allow-leakage to inspect, "
                         "but do NOT report those numbers as held-out results).")
        else:
            print(f"Leakage check skipped: columns {args.tp_ref_col}/{args.tp_stud_col} not "
                  f"found in {list(tp.columns)}")
    else:
        print("WARNING: no --training-pairs given; leakage could not be checked.")

    # ---- similarities
    sims = {}
    fit_texts = list(nontest["ref"]) + list(nontest["stud"])
    sims["TF-IDF"] = tfidf_similarity(fit_texts, test["ref"], test["stud"])
    sims["General SBERT"] = sbert_similarity(args.base_model, list(test["ref"]),
                                             list(test["stud"]), cache_dir, args.batch_size)
    sims["SBERT-EduAdapt"] = sbert_similarity(args.eduadapt, list(test["ref"]),
                                              list(test["stud"]), cache_dir, args.batch_size)

    tune = None
    if args.threshold_source == "train":
        tune = nontest[~nontest["split_n"].isin(val_vals)] if val_vals else nontest
        print("NOTE: tuning on the training split favours the fine-tuned model "
              "(it has seen these pairs). Prefer cv-test or val.")
    elif args.threshold_source == "val":
        if not val_vals:
            sys.exit("--threshold-source val needs --val-splits")
        tune = df[df["split_n"].isin(val_vals)]
    tune_sims = {}
    if tune is not None:
        tune_sims["TF-IDF"] = tfidf_similarity(fit_texts, tune["ref"], tune["stud"])
        tune_sims["General SBERT"] = sbert_similarity(args.base_model, list(tune["ref"]),
                                                      list(tune["stud"]), cache_dir,
                                                      args.batch_size)
        tune_sims["SBERT-EduAdapt"] = sbert_similarity(args.eduadapt, list(tune["ref"]),
                                                       list(tune["stud"]), cache_dir,
                                                       args.batch_size)

    yb, yo = test["y_bin"].values, test["y_ord"].values
    rows, preds, thresholds = [], {}, {}
    for name, s in sims.items():
        if tune is not None:
            pb, po, used = predict_with_thresholds(
                s, yb, yo, args.threshold_source, tune_sims[name],
                tune["y_bin"].values, tune["y_ord"].values)
        else:
            pb, po, used = predict_with_thresholds(s, yb, yo, "cv-test", folds=args.folds)
        preds[name] = {"bin": pb, "ord": po}
        thresholds[name] = used
        point = metric_set(s, yb, yo, pb, po)
        lo, hi = bootstrap_ci(s, yb, yo, pb, po, args.n_boot, rng)
        for k, v in point.items():
            rows.append({"model": name, "metric": k, "value": v,
                         "ci_low": lo[k], "ci_high": hi[k]})
        test[f"sim_{name}"] = s
        test[f"pred_bin_{name}"] = pb
        test[f"pred_ord_{name}"] = po
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "metrics_with_ci.csv"), index=False)

    # ---- paired comparisons
    pairs = [("TF-IDF", "General SBERT"), ("General SBERT", "SBERT-EduAdapt"),
             ("TF-IDF", "SBERT-EduAdapt")]
    mf1 = lambda y, p: f1_score(y, p, average="macro", zero_division=0)
    qwk = lambda y, p: cohen_kappa_score(y, p, weights="quadratic")
    comp = []
    for a, b in pairs:
        n01, n10, p = mcnemar(yb, preds[a]["bin"], preds[b]["bin"])
        d, dl, dh = paired_boot_diff(yb, preds[a]["bin"], preds[b]["bin"], args.n_boot, rng, mf1)
        q, ql, qh = paired_boot_diff(yo, preds[a]["ord"], preds[b]["ord"], args.n_boot, rng, qwk)
        comp.append({"model_A": a, "model_B": b,
                     "mcnemar_A_right_B_wrong": n01, "mcnemar_A_wrong_B_right": n10,
                     "mcnemar_p_exact": p,
                     "delta_bin_macroF1_B_minus_A": d, "delta_ci_low": dl, "delta_ci_high": dh,
                     "delta_ord_QWK_B_minus_A": q, "qwk_ci_low": ql, "qwk_ci_high": qh})
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(args.out, "paired_comparisons.csv"), index=False)

    # ---- confusion matrices + per-class
    with open(os.path.join(args.out, "confusion_and_per_class.txt"), "w", encoding="utf-8") as f:
        for name in sims:
            f.write(f"=== {name} ===\nBinary (rows gold [1,0], cols pred [1,0]):\n")
            f.write(str(confusion_matrix(yb, preds[name]["bin"], labels=[1, 0])) + "\n")
            f.write("Ordinal (rows gold [C1,C2,C3/C4], cols pred):\n")
            f.write(str(confusion_matrix(yo, preds[name]["ord"], labels=[2, 1, 0])) + "\n")
            p, r, f1, sup = precision_recall_fscore_support(yo, preds[name]["ord"],
                                                            labels=[2, 1, 0], zero_division=0)
            for lab, pp, rr, ff, ss in zip(["C1", "C2", "C3/C4"], p, r, f1, sup):
                f.write(f"  {lab}: P={pp:.3f} R={rr:.3f} F1={ff:.3f} n={ss}\n")
            f.write("\n")

    test.drop(columns=["split_n"]).to_csv(os.path.join(args.out, "test_predictions.csv"),
                                          index=False, encoding="utf-8")

    # ---- Table 3 (paper-ready)
    def cell(model, metric):
        r = res[(res.model == model) & (res.metric == metric)].iloc[0]
        return f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"

    table_cols = [("bin_accuracy", "Binary Acc."), ("bin_macro_f1", "Binary macro-F1"),
                  ("bin_kappa", "Binary κ"), ("ord_qwk", "Ordinal QWK"),
                  ("roc_auc", "ROC-AUC"), ("spearman_rho", "Spearman ρ")]
    t3 = pd.DataFrame([{"Model": m, **{lab: cell(m, k) for k, lab in table_cols}}
                       for m in sims])
    t3.to_csv(os.path.join(args.out, "table3_for_manuscript.csv"), index=False)
    with open(os.path.join(args.out, "table3_for_manuscript.md"), "w", encoding="utf-8") as f:
        f.write(f"Held-out test split (n = {len(test)}); values = estimate [95% bootstrap CI]; "
                f"thresholds via {args.threshold_source}.\n\n")
        f.write("| " + " | ".join(t3.columns) + " |\n|" + "---|" * len(t3.columns) + "\n")
        for _, r in t3.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")

    make_figures(test, sims, preds, args.out)

    # ---- metadata
    meta = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "python": platform.python_version(),
        "data_file": os.path.abspath(args.data), "data_sha256": sha256(args.data),
        "n_total": int(len(df)), "n_test": int(len(test)),
        "test_splits": test_vals, "test_label_counts": test["label"].value_counts().to_dict(),
        "threshold_source": args.threshold_source, "folds": args.folds, "n_boot": args.n_boot,
        "base_model": args.base_model, "eduadapt": os.path.abspath(args.eduadapt)
        if os.path.exists(args.eduadapt) else args.eduadapt,
        "leakage_check": leak, "thresholds": thresholds,
        "runtime_sec": round(time.time() - t0, 1),
    }
    st = os.path.join(args.eduadapt, "model.safetensors")
    if os.path.exists(st):
        meta["eduadapt_model_sha256"] = sha256(st)
    try:
        import sklearn, scipy
        meta["versions"] = {"numpy": np.__version__, "pandas": pd.__version__,
                            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}
        import sentence_transformers, torch
        meta["versions"].update({"sentence-transformers": sentence_transformers.__version__,
                                 "torch": torch.__version__})
    except ImportError:
        pass
    with open(os.path.join(args.out, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)

    print("\n" + open(os.path.join(args.out, "table3_for_manuscript.md"), encoding="utf-8").read())
    print(comp.round(4).to_string(index=False))
    print(f"\nDone in {meta['runtime_sec']} s. Outputs in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

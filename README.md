# SEAR EduAdapt ablation

Code and held-out results for the manuscript **“Educationally Adapted Sentence Embeddings for Proposition-Level Concept Map Assessment: Evidence from Unseen Answers, Questions, and Domains.”**

The study evaluates the semantic-retrieval stage of a two-stage concept-map assessment design. It compares three ways of scoring a student answer by its cosine similarity to the reference answer:

| Condition | Description |
|---|---|
| TF-IDF | Lexical baseline; vectoriser fitted without labels on non-test text |
| General SBERT | `sentence-transformers/all-mpnet-base-v2`, unadapted |
| SBERT-EduAdapt | Same encoder fine-tuned on the SciEntsBank **training split only** (targets C1 = 1.0, C2 = 0.7, C3/C4 = 0.0; 5 epochs, batch 16, lr 2e-5, seed 42) |

All conditions are evaluated on the official SciEntsBank test splits: unseen answers (UA), unseen questions (UQ), and unseen domains (UD). Thresholds are chosen by five-fold cross-fitting inside each test set. Metrics come with 95% bootstrap CIs (1,000 resamples), and models are compared with exact McNemar tests.

## Main results (pooled test set, n = 5,835)

| Metric | TF-IDF | General SBERT | SBERT-EduAdapt |
|---|---|---|---|
| Binary macro-F1 | 0.656 [0.642, 0.668] | 0.658 [0.646, 0.671] | **0.728** [0.715, 0.741] |
| Binary Cohen’s κ | 0.312 | 0.317 | **0.456** |
| Ordinal QWK (C1 / C2 / C3–C4) | 0.404 | 0.378 | **0.487** |
| ROC-AUC | 0.720 | 0.718 | **0.806** |

- SBERT-EduAdapt vs General SBERT: Δ macro-F1 +0.070 [0.056, 0.084], McNemar p < 0.001.
- General SBERT vs TF-IDF: no difference (p = 0.48).
- Binary macro-F1 by split for SBERT-EduAdapt: UA 0.873, UQ 0.711, UD 0.714.

Full tables are in `results/<split>/`.

## Repository layout

```
pipeline_scientsbank.py      end-to-end pipeline (download → corpus → fine-tune → ablation → zip)
train_eduadapt_heldout.py    fine-tunes SBERT-EduAdapt on the non-test split only
run_ablation.py              ablation, thresholds, metrics, bootstrap CIs, McNemar, figures
make_paper_figures.py        regenerates manuscript Figures 2–3 from results/
run_windows.bat              double-click launcher for Windows
results/
  all/  test_ua/  test_uq/  test_ud/
    metrics_with_ci.csv            point estimates and 95% CIs
    paired_comparisons.csv         McNemar counts/p and paired-bootstrap deltas
    confusion_and_per_class.txt    confusion matrices, per-class P/R/F1
    table3_for_manuscript.{md,csv}
    test_predictions_notext.csv    per-instance similarities and predictions (no answer text)
    run_metadata.json              data/model SHA-256, versions, seed, fold thresholds
    fig_*.png
```

## Reproducing

```bash
pip install -r requirements.txt
python pipeline_scientsbank.py      # Windows: double-click run_windows.bat
```

The pipeline downloads SciEntsBank from the Hugging Face Hub (`nkazi/SciEntsBank`) and maps the labels to C1–C4 (correct → C1, partially_correct_incomplete → C2, contradictory → C3, irrelevant/non_domain → C4). It then fine-tunes SBERT-EduAdapt on the training split and writes everything to `sear_results/`. Fine-tuning took about 3 hours on a CPU; with a GPU it takes minutes. Steps that have already finished are skipped when the pipeline is rerun.

The reported run used the corpus with SHA-256 `9cf083b3…087e` and produced a fine-tuned model with SHA-256 `9c1b4458…93b3`. Both hashes are recorded in `run_metadata.json`.

## Data

SciEntsBank comes from the SemEval-2013 Task 7 shared task (Dzikovska et al., 2013). This repository does not redistribute the answer texts. `test_predictions_notext.csv` keeps the row order of the regenerated corpus, so predictions can be re-joined to the texts after running the pipeline.

## Citation

See `CITATION.cff`. Please also cite the dataset:

> M. O. Dzikovska et al., “SemEval-2013 Task 7: The joint student response analysis and 8th recognizing textual entailment challenge,” in *Proc. SemEval 2013*, pp. 263–274.

## License

Code: MIT (see `LICENSE`). The SciEntsBank data remain under the terms set by their original distributors.

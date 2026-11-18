#!/usr/bin/env python3
"""
Retrain SBERT-EduAdapt on the NON-TEST split only (use this if run_ablation.py
stops with a leakage warning). Same configuration as the manuscript:
all-mpnet-base-v2, 5 epochs, batch 16, lr 2e-5, seed 42, CosineSimilarityLoss,
targets C1=1.0, C2=0.7, C3/C4=0.0.

  python train_eduadapt_heldout.py --data semantic_corpus.csv --out ./SBERT-EduAdapt-v1-heldout
"""
import argparse
import random

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ref-col", default="reference_answer")
ap.add_argument("--stud-col", default="student_answer")
ap.add_argument("--label-col", default="normalized_label")
ap.add_argument("--split-col", default="split")
ap.add_argument("--test-splits", default="test")
ap.add_argument("--base-model", default="sentence-transformers/all-mpnet-base-v2")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--out", required=True)
args = ap.parse_args()

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

random.seed(42); np.random.seed(42); torch.manual_seed(42)
TARGET = {"C1": 1.0, "C2": 0.7, "C3": 0.0, "C4": 0.0}

df = pd.read_csv(args.data)
test_vals = {s.strip().lower() for s in args.test_splits.split(",")}
df = df[~df[args.split_col].astype(str).str.strip().str.lower().isin(test_vals)]
df = df.dropna(subset=[args.ref_col, args.stud_col, args.label_col])
clean = lambda t: " ".join(str(t).lower().split())
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ablation import normalize_label
lab = df[args.label_col].map(normalize_label)
examples = [InputExample(texts=[clean(r), clean(s)], label=TARGET[l])
            for r, s, l in zip(df[args.ref_col], df[args.stud_col], lab)]
if not examples:
    sys.exit("No training examples found - check label/split columns.")
print(f"Training pairs (non-test only): {len(examples)}")

model = SentenceTransformer(args.base_model)
loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
          optimizer_params={"lr": args.lr}, warmup_steps=int(0.1 * len(loader)),
          show_progress_bar=True)
model.save(args.out)
print("Saved to", args.out)

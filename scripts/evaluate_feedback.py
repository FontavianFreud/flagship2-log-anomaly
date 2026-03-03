from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    # simple KS distance without scipy
    a = np.sort(a)
    b = np.sort(b)
    grid = np.sort(np.unique(np.concatenate([a, b])))
    if len(grid) == 0:
        return 0.0
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate feedback loop: precision@k, stability, score drift")
    p.add_argument("--scored", type=str, default="data/processed/scored_windows.parquet")
    p.add_argument("--db", type=str, default="data/feedback/feedback.sqlite")
    p.add_argument("--model", type=str, default="iforest", choices=["iforest","ocsvm"])
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    scored = pd.read_parquet(args.scored)
    scored = scored[scored["model"] == args.model].copy()
    scored["day"] = scored["window_start"].dt.date

    # 1) anomaly-rate stability on val split
    val = scored[scored["split"] == "val"].copy()
    daily_rate = val.groupby("day")["is_anomaly"].mean()
    mean_rate = float(daily_rate.mean()) if len(daily_rate) else 0.0
    std_rate = float(daily_rate.std(ddof=1)) if len(daily_rate) > 1 else 0.0
    cv_rate = std_rate / (mean_rate + 1e-9)

    # 2) drift in score distribution (train vs val) using KS distance
    train_scores = scored[scored["split"] == "train"]["anomaly_score"].to_numpy()
    val_scores = scored[scored["split"] == "val"]["anomaly_score"].to_numpy()
    drift_ks = ks_distance(train_scores, val_scores) if len(train_scores) and len(val_scores) else 0.0

    # 3) precision@k on reviewed items (top-k per day, among labeled rows)
    conn = sqlite3.connect(args.db)
    labels = pd.read_sql_query(
        "SELECT model, grp as `group`, window_start, label FROM anomaly_labels WHERE model = ?",
        conn,
        params=(args.model,),
    )
    if len(labels):
        labels["window_start"] = pd.to_datetime(labels["window_start"])
        labels["day"] = labels["window_start"].dt.date

    # build top-k per day from validation scored windows
    topk = (val.sort_values(["day","anomaly_score"], ascending=[True, False])
              .groupby("day")
              .head(args.k)
              .copy())

    merged = topk.merge(labels, on=["group","window_start"], how="left")
    reviewed = merged[merged["label"].notna()].copy()

    precision = float(reviewed["label"].mean()) if len(reviewed) else float("nan")

    print(f"MODEL={args.model}")
    print(f"val daily anomaly-rate mean={mean_rate:.6f} std={std_rate:.6f} CV={cv_rate:.3f}")
    print(f"score drift KS(train vs val)={drift_ks:.4f}")
    print(f"precision@{args.k} on reviewed top-k/day (val)={precision}  (n_reviewed={len(reviewed)})")

if __name__ == "__main__":
    main()

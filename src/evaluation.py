"""Metrics shared by model and HMM experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def route_continuity(predicted: np.ndarray, arcs: pd.DataFrame) -> float:
    if len(predicted) < 2:
        return 1.0
    endpoints = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    valid = 0
    for previous, current in zip(predicted[:-1], predicted[1:]):
        valid += int(previous == current or endpoints[previous, 1] == endpoints[current, 0])
    return valid / (len(predicted) - 1)


def evaluate_candidates(
    table: pd.DataFrame, predicted: np.ndarray, arcs: pd.DataFrame
) -> dict[str, float]:
    groups = list(table.groupby("point_idx", sort=True))
    truth = np.array(
        [
            int(group.loc[group["label"].eq(1), "arc_id"].iloc[0])
            if group["label"].any()
            else -1
            for _, group in groups
        ]
    )
    top1 = float(np.mean(predicted == truth)) if len(truth) else 0.0
    return {
        "top1_accuracy": top1,
        "candidate_miss_rate": float(1.0 - table.groupby("point_idx")["label"].max().mean()),
        "route_continuity": route_continuity(predicted, arcs),
        "macro_f1_candidate": float(f1_score(table["label"], table["predicted_label"], average="macro", zero_division=0)),
        "log_loss_candidate": float(log_loss(table["label"], table["prob"], labels=[0, 1])),
    }

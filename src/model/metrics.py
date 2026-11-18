"""Candidate-grouped evaluation metrics for PathSnap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


GROUP_COLUMNS = ("trip_id", "point_idx")


def _group_positions(table: pd.DataFrame) -> dict[Any, np.ndarray]:
    """Return positional row indices for each candidate group."""
    return {
        key: np.asarray(positions, dtype=np.int64)
        for key, positions in table.groupby(list(GROUP_COLUMNS), sort=False).indices.items()
    }


def group_softmax(table: pd.DataFrame, score_col: str = "score") -> np.ndarray:
    """Convert arbitrary scores into probabilities that sum to one per GPS point."""
    scores = table[score_col].to_numpy(dtype=float)
    probabilities = np.zeros(len(table), dtype=float)
    for positions in _group_positions(table).values():
        values = scores[positions]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite scores in candidate group: {positions.tolist()}")
        shifted = values - np.max(values)
        weights = np.exp(shifted)
        probabilities[positions] = weights / weights.sum()
    return probabilities


def _best_positions(table: pd.DataFrame, score_col: str = "score") -> np.ndarray:
    """Select one highest-scoring candidate per GPS point with deterministic ties."""
    scores = table[score_col].to_numpy(dtype=float)
    ranks = table["candidate_rank"].to_numpy(dtype=int) if "candidate_rank" in table else np.zeros(len(table), dtype=int)
    arc_ids = table["arc_id"].to_numpy(dtype=int)
    best = []
    for positions in _group_positions(table).values():
        order = np.lexsort((arc_ids[positions], ranks[positions], -scores[positions]))
        best.append(int(positions[order[0]]))
    return np.asarray(best, dtype=np.int64)


def _route_continuity(top: pd.DataFrame, arcs: np.ndarray | None) -> tuple[int, int]:
    """Return connected and evaluated adjacent predicted-arc transition counts."""
    if arcs is None or len(top) < 2:
        return 0, 0

    connected = 0
    evaluated = 0
    for _, trip in top.sort_values(["trip_id", "point_idx"]).groupby("trip_id", sort=False):
        predicted = trip["arc_id"].to_numpy(dtype=int)
        gaps = trip["large_gap"].to_numpy(dtype=bool) if "large_gap" in trip else np.zeros(len(trip), dtype=bool)
        for index in range(1, len(predicted)):
            if gaps[index]:
                continue
            previous_id, current_id = predicted[index - 1], predicted[index]
            if not (0 <= previous_id < len(arcs) and 0 <= current_id < len(arcs)):
                continue
            evaluated += 1
            previous_arc, current_arc = arcs[previous_id], arcs[current_id]
            if previous_id == current_id or previous_arc[1] == current_arc[0]:
                connected += 1
    return connected, evaluated


def evaluate_candidate_table(
    table: pd.DataFrame,
    *,
    arcs: np.ndarray | None = None,
    score_col: str = "score",
) -> dict[str, float | int]:
    """Evaluate candidate scores at point and route level.

    ``candidate_macro_f1`` is binary candidate-row Macro F1: the selected
    candidate in each GPS group is treated as predicted positive.
    ``point_log_loss`` is calculated only for groups where a positive
    candidate exists.
    """
    required = set(GROUP_COLUMNS) | {"arc_id", "label", score_col}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("Cannot evaluate an empty candidate table")

    labels = table["label"].to_numpy(dtype=int)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("label must contain only 0 and 1")

    probabilities = group_softmax(table, score_col=score_col)
    best_positions = _best_positions(table, score_col=score_col)
    best_labels = labels[best_positions]
    predicted_positive = np.zeros(len(table), dtype=int)
    predicted_positive[best_positions] = 1

    groups = _group_positions(table)
    point_count = len(groups)
    covered_points = 0
    nll_values = []
    for positions in groups.values():
        positive = labels[positions] == 1
        if positive.any():
            covered_points += 1
            true_probability = float(probabilities[positions][positive].sum())
            nll_values.append(-np.log(max(true_probability, 1e-15)))

    connected, evaluated = _route_continuity(table.iloc[best_positions].copy(), arcs)
    result: dict[str, float | int] = {
        "point_count": point_count,
        "covered_points": covered_points,
        "missed_points": point_count - covered_points,
        "candidate_miss_rate": (point_count - covered_points) / point_count,
        "top1_correct": int(best_labels.sum()),
        "top1_accuracy": float(best_labels.mean()),
        "candidate_macro_f1": float(f1_score(labels, predicted_positive, average="macro", zero_division=0)),
        "point_log_loss": float(np.mean(nll_values)) if nll_values else np.nan,
        "route_connected_transitions": connected,
        "route_evaluated_transitions": evaluated,
        "route_continuity": connected / evaluated if evaluated else np.nan,
        "mean_selected_rank": float(table.iloc[best_positions]["candidate_rank"].mean())
        if "candidate_rank" in table
        else np.nan,
    }
    return result


def aggregate_trip_metrics(trip_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-trip metrics with point-weighted metrics where appropriate."""
    rows = []
    for partition, frame in trip_metrics.groupby("partition", sort=True):
        point_total = int(frame["point_count"].sum())
        covered_total = int(frame["covered_points"].sum())
        top1_total = int(frame["top1_correct"].sum())
        transition_total = int(frame["route_evaluated_transitions"].sum())
        connected_total = int(frame["route_connected_transitions"].sum())
        rows.append({
            "partition": partition,
            "trips": len(frame),
            "points": point_total,
            "covered_points": covered_total,
            "candidate_miss_rate": (point_total - covered_total) / point_total,
            "top1_accuracy": top1_total / point_total,
            "trip_mean_top1_accuracy": float(frame["top1_accuracy"].mean()),
            "trip_mean_candidate_macro_f1": float(frame["candidate_macro_f1"].mean()),
            "point_mean_log_loss": float(np.average(
                frame["point_log_loss"].fillna(0),
                weights=frame["covered_points"].clip(lower=0),
            )) if covered_total else np.nan,
            "route_continuity": connected_total / transition_total if transition_total else np.nan,
            "trip_mean_route_continuity": float(frame["route_continuity"].mean()),
            "mean_selected_rank": float(frame["mean_selected_rank"].mean()),
        })
    return pd.DataFrame(rows)

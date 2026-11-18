"""Connectivity-constrained Viterbi decoding."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def build_transition_graph(arcs: pd.DataFrame) -> dict[int, set[int]]:
    """Map each arc to arcs whose start node is its end node."""
    starts: dict[int, set[int]] = defaultdict(set)
    for arc_id, node in enumerate(arcs["from_node"].to_numpy(dtype=int)):
        starts[int(node)].add(arc_id)
    graph: dict[int, set[int]] = {}
    for arc_id, node in enumerate(arcs["to_node"].to_numpy(dtype=int)):
        graph[arc_id] = set(starts.get(int(node), set()))
        graph[arc_id].add(arc_id)
    return graph


def decode_viterbi(
    candidates: pd.DataFrame,
    transition_graph: dict[int, set[int]],
    transition_penalty: float = -4.0,
) -> np.ndarray:
    """Decode one arc per GPS point using candidate probabilities."""
    if candidates.empty:
        return np.array([], dtype=int)
    grouped = [group for _, group in candidates.groupby("point_idx", sort=True)]
    states = [group["arc_id"].to_numpy(dtype=int) for group in grouped]
    emissions = [np.log(np.clip(group["prob"].to_numpy(dtype=float), 1e-12, 1.0)) for group in grouped]
    scores = emissions[0].copy()
    backpointers: list[np.ndarray] = []
    for time in range(1, len(states)):
        previous = states[time - 1]
        current = states[time]
        next_scores = np.empty(len(current), dtype=float)
        pointer = np.empty(len(current), dtype=int)
        for j, arc_id in enumerate(current):
            transition = np.array(
                [0.0 if arc_id in transition_graph.get(int(prev), set()) else transition_penalty for prev in previous]
            )
            values = scores + transition
            pointer[j] = int(np.argmax(values))
            next_scores[j] = values[pointer[j]] + emissions[time][j]
        backpointers.append(pointer)
        scores = next_scores
    selected = np.empty(len(states), dtype=int)
    selected[-1] = int(np.argmax(scores))
    for time in range(len(states) - 2, -1, -1):
        selected[time] = backpointers[time][selected[time + 1]]
    return np.array([states[t][selected[t]] for t in range(len(states))], dtype=int)

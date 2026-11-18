"""Connectivity-constrained HMM and Viterbi decoding for PathSnap."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def build_transition_graph(arcs: pd.DataFrame) -> dict[int, set[int]]:
    """Map each arc to arcs whose start node is its end node (plus self-transition)."""
    starts: dict[int, set[int]] = defaultdict(set)
    for arc_id, node in enumerate(arcs["from_node"].to_numpy(dtype=int)):
        starts[int(node)].add(arc_id)
    graph: dict[int, set[int]] = {}
    for arc_id, node in enumerate(arcs["to_node"].to_numpy(dtype=int)):
        graph[arc_id] = set(starts.get(int(node), set()))
        graph[arc_id].add(arc_id)
    return graph


def compute_emission_log_probs(
    candidates_group: pd.DataFrame,
    prob_col: str = "prob",
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute group-normalized observation emission log probabilities.

    Returns:
        arcs: 1D array of arc IDs in the candidate group.
        log_emissions: 1D array of log-normalized emission probabilities.
    """
    arcs = candidates_group["arc_id"].to_numpy(dtype=int)
    raw_probs = candidates_group[prob_col].to_numpy(dtype=float) if prob_col in candidates_group else np.ones(len(arcs))

    # Numerical stability: clip positive probabilities
    clipped = np.clip(raw_probs, epsilon, None)
    prob_sum = float(np.sum(clipped))
    if prob_sum <= 0.0 or not np.isfinite(prob_sum):
        norm_probs = np.ones(len(arcs)) / max(len(arcs), 1)
    else:
        norm_probs = clipped / prob_sum

    log_emissions = np.log(np.clip(norm_probs, epsilon, 1.0))
    return arcs, log_emissions


def decode_viterbi(
    candidates: pd.DataFrame,
    transition_graph: dict[int, set[int]],
    transition_penalty: float = -4.0,
    emission_weight: float = 1.0,
    large_gap_penalty: float = 0.0,
    prob_col: str = "prob",
) -> np.ndarray:
    """Decode the optimal candidate arc sequence via the Viterbi dynamic programming algorithm.

    Args:
        candidates: DataFrame containing candidate rows for a trip.
        transition_graph: Adjacency map from each arc to its valid successors.
        transition_penalty: Log probability penalty for disconnected transitions.
        emission_weight: Multiplier on observation log emission probabilities.
        large_gap_penalty: Transition penalty applied when a large GPS time gap occurs.
        prob_col: Column name containing the candidate probability scores.

    Returns:
        1D numpy array of chosen arc IDs, one per GPS point.
    """
    if candidates.empty:
        return np.array([], dtype=int)

    # Group candidate rows by observation point
    point_col = "point_idx" if "point_idx" in candidates.columns else candidates.columns[1]
    grouped = [group for _, group in candidates.groupby(point_col, sort=True)]
    if not grouped:
        return np.array([], dtype=int)

    # Check for large_gap indicators per observation
    has_large_gap = "large_gap" in candidates.columns
    large_gaps = [
        bool(group["large_gap"].iloc[0]) if has_large_gap else False
        for group in grouped
    ]

    states: list[np.ndarray] = []
    emissions: list[np.ndarray] = []

    for group in grouped:
        arcs, log_emiss = compute_emission_log_probs(group, prob_col=prob_col)
        states.append(arcs)
        emissions.append(log_emiss * emission_weight)

    # Initialize Viterbi trellis at t = 0
    scores = emissions[0].copy()
    backpointers: list[np.ndarray] = []

    # Forward dynamic programming pass (t = 1 to T - 1)
    for time_step in range(1, len(states)):
        previous_states = states[time_step - 1]
        current_states = states[time_step]
        is_gap = large_gaps[time_step]

        n_prev = len(previous_states)
        n_curr = len(current_states)
        current_penalty = large_gap_penalty if is_gap else transition_penalty

        # Vectorized transition matrix evaluation
        trans_matrix = np.full((n_prev, n_curr), current_penalty, dtype=float)
        for i, prev in enumerate(previous_states):
            succs = transition_graph.get(int(prev), set())
            for j, curr in enumerate(current_states):
                if curr in succs:
                    trans_matrix[i, j] = 0.0

        total_matrix = scores[:, None] + trans_matrix
        pointer = np.argmax(total_matrix, axis=0)
        next_scores = total_matrix[pointer, np.arange(n_curr)] + emissions[time_step]

        backpointers.append(pointer)
        scores = next_scores

    # Termination: find highest score at final step
    selected_indices = np.empty(len(states), dtype=int)
    selected_indices[-1] = int(np.argmax(scores))

    # Backward traceback pass
    for time_step in range(len(states) - 2, -1, -1):
        selected_indices[time_step] = backpointers[time_step][selected_indices[time_step + 1]]

    # Extract chosen arc IDs
    decoded_arcs = np.array([states[t][selected_indices[t]] for t in range(len(states))], dtype=int)
    return decoded_arcs


class HMMRouteDecoder:
    """Configurable HMM Viterbi Decoder with transition graph and diagnostics."""

    def __init__(
        self,
        transition_penalty: float = -4.0,
        emission_weight: float = 1.0,
        large_gap_penalty: float = 0.0,
        prob_col: str = "prob",
    ) -> None:
        self.transition_penalty = float(transition_penalty)
        self.emission_weight = float(emission_weight)
        self.large_gap_penalty = float(large_gap_penalty)
        self.prob_col = str(prob_col)

    def decode_trip(
        self,
        candidates: pd.DataFrame,
        arcs: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Decode trip observations into both classifier-only and HMM-decoded routes.

        Returns:
            classifier_route: 1D array of argmax classifier arc IDs.
            hmm_route: 1D array of Viterbi decoded arc IDs.
            diagnostics: Dictionary containing transition and agreement statistics.
        """
        point_col = "point_idx" if "point_idx" in candidates.columns else candidates.columns[1]

        # 1. Classifier-only: select candidate with highest probability per observation
        classifier_route = (
            candidates.sort_values(self.prob_col)
            .groupby(point_col, sort=True)
            .tail(1)
            .sort_values(point_col)["arc_id"]
            .to_numpy(dtype=int)
        )

        # 2. Build graph and decode via Viterbi
        graph = build_transition_graph(arcs)
        hmm_route = decode_viterbi(
            candidates,
            transition_graph=graph,
            transition_penalty=self.transition_penalty,
            emission_weight=self.emission_weight,
            large_gap_penalty=self.large_gap_penalty,
            prob_col=self.prob_col,
        )

        # 3. Compute diagnostics
        endpoints = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
        n_points = len(hmm_route)
        disagreements = int(np.sum(classifier_route != hmm_route))
        disagreement_rate = (disagreements / n_points) * 100.0 if n_points > 0 else 0.0

        disconnected_clf = 0
        disconnected_hmm = 0
        evaluated_transitions = max(n_points - 1, 0)

        for i in range(evaluated_transitions):
            # Classifier transitions
            prev_c, curr_c = classifier_route[i], classifier_route[i + 1]
            if not (prev_c == curr_c or endpoints[prev_c, 1] == endpoints[curr_c, 0]):
                disconnected_clf += 1

            # HMM transitions
            prev_h, curr_h = hmm_route[i], hmm_route[i + 1]
            if not (prev_h == curr_h or endpoints[prev_h, 1] == endpoints[curr_h, 0]):
                disconnected_hmm += 1

        diagnostics = {
            "n_points": n_points,
            "evaluated_transitions": evaluated_transitions,
            "classifier_disconnected": disconnected_clf,
            "hmm_disconnected": disconnected_hmm,
            "disagreements": disagreements,
            "disagreement_rate_percent": disagreement_rate,
        }

        return classifier_route, hmm_route, diagnostics

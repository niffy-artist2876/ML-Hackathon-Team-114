"""Label-free directed-road transitions and Viterbi map matching."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush

import numpy as np
import pandas as pd


class DirectedRoadNetwork:
    """Find shortest directed distances inside a fixed routing radius."""

    def __init__(self, arcs: np.ndarray, node_xy: np.ndarray, max_route_m: float):
        if not np.isfinite(max_route_m) or max_route_m <= 0:
            raise ValueError("max_route_m must be finite and positive")
        if arcs.ndim != 2 or arcs.shape[1] != 2 or (arcs < 0).any() or (arcs >= len(node_xy)).any():
            raise ValueError("Expected directed arcs with valid node IDs")
        self.arcs = arcs
        self.max_route_m = max_route_m
        self.lengths = np.linalg.norm(node_xy[arcs[:, 1]] - node_xy[arcs[:, 0]], axis=1)
        if not np.isfinite(self.lengths).all():
            raise ValueError("Non-finite arc length")
        self.adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(node_xy))]
        for (source, target), length in zip(arcs, self.lengths, strict=True):
            self.adjacency[source].append((int(target), float(length)))
        self.reachable = lru_cache(maxsize=4096)(self._bounded_distances)

    def _bounded_distances(self, source: int) -> dict[int, float]:
        distances = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance > distances[node]:
                continue
            for target, length in self.adjacency[node]:
                new_distance = distance + length
                if new_distance <= self.max_route_m and new_distance < distances.get(target, np.inf):
                    distances[target] = new_distance
                    heappush(queue, (new_distance, target))
        return distances

    def distances(
        self,
        prev_ids: np.ndarray,
        prev_fraction: np.ndarray,
        next_ids: np.ndarray,
        next_fraction: np.ndarray,
        backward_jitter_m: float,
    ) -> np.ndarray:
        """Measure from the previous projection to the next projection."""
        prev_length, next_length = self.lengths[prev_ids], self.lengths[next_ids]
        prev_end = self.arcs[prev_ids, 1]
        next_start = self.arcs[next_ids, 0].tolist()
        between = np.array([
            [nearby.get(target, np.inf) for target in next_start]
            for nearby in (self.reachable(int(source)) for source in prev_end)
        ])
        via_nodes = (prev_length * (1.0 - prev_fraction))[:, None] + between + next_length * next_fraction
        along_arc = (next_fraction[None, :] - prev_fraction[:, None]) * prev_length[:, None]
        same_arc = prev_ids[:, None] == next_ids[None, :]
        forward = same_arc & (along_arc >= -backward_jitter_m)
        distance = np.minimum(via_nodes, np.where(forward, np.maximum(0.0, along_arc), np.inf))
        distance[distance > self.max_route_m] = np.inf
        return distance


@dataclass(frozen=True)
class CandidateLayer:
    arc_ids: np.ndarray
    fractions: np.ndarray
    emissions: np.ndarray


@dataclass(frozen=True)
class TripLattice:
    layers: list[CandidateLayer]
    distances: list[np.ndarray | None]
    gps_step_m: np.ndarray
    break_before: tuple[str, ...]


@dataclass(frozen=True)
class DecodedTrip:
    arc_ids: np.ndarray
    choices: np.ndarray
    break_before: tuple[str, ...]


def build_lattice(
    candidates: pd.DataFrame,
    road: DirectedRoadNetwork,
    xy: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_gap_s: float,
    backward_jitter_m: float,
) -> TripLattice:
    """Construct only transitions between candidates of consecutive GPS points."""
    if len(xy) != len(timestamps) or len(xy) == 0:
        raise ValueError("Expected nonempty, equally sized GPS coordinate and time arrays")
    if not (np.isfinite(xy).all() and np.isfinite(timestamps).all()):
        raise ValueError("Non-finite GPS coordinates or timestamps")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    if not np.isfinite(backward_jitter_m) or backward_jitter_m < 0:
        raise ValueError("backward_jitter_m must be finite and nonnegative")
    ordered = candidates.sort_values(["point_idx", "candidate_rank", "arc_id"])
    if ordered.duplicated(["point_idx", "arc_id"]).any():
        raise ValueError("Duplicate candidate arc at one point")
    point_ids = ordered["point_idx"].to_numpy(dtype=int)
    arc_ids = ordered["arc_id"].to_numpy(dtype=int)
    fractions = ordered["projection_fraction"].to_numpy(dtype=float)
    emissions = ordered["score"].to_numpy(dtype=float)
    if (point_ids < 0).any() or (point_ids >= len(xy)).any():
        raise ValueError("Candidate point index outside GPS track")
    if (arc_ids < 0).any() or (arc_ids >= len(road.arcs)).any():
        raise ValueError("Candidate arc index outside road graph")
    if not np.isfinite(emissions).all() or not ((0 <= fractions) & (fractions <= 1)).all():
        raise ValueError("Invalid emission score or projection fraction")

    offsets = np.r_[0, np.cumsum(np.bincount(point_ids, minlength=len(xy)))]
    layers = [
        CandidateLayer(arc_ids[a:b], fractions[a:b], emissions[a:b])
        for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    gps_step_m = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    elapsed = np.diff(timestamps, prepend=timestamps[0])
    breaks = ["start"]
    distances = [None]
    for index in range(1, len(layers)):
        prev, current = layers[index - 1], layers[index]
        reason, matrix = "", None
        if elapsed[index] > max_gap_s:
            reason = "gps_gap"
        elif elapsed[index] < 0:
            reason = "invalid_time"
        elif not len(prev.arc_ids):
            reason = "after_unmatched"
        elif len(current.arc_ids):
            matrix = road.distances(
                prev.arc_ids, prev.fractions, current.arc_ids, current.fractions, backward_jitter_m,
            )
        breaks.append(reason)
        distances.append(matrix)
    return TripLattice(layers, distances, gps_step_m, tuple(breaks))


def _decoded_trip(lattice: TripLattice, choices: np.ndarray, breaks: list[str]) -> DecodedTrip:
    arc_ids = np.full(len(choices), -1, dtype=np.int64)
    for index, (layer, choice) in enumerate(zip(lattice.layers, choices, strict=True)):
        if choice >= 0:
            arc_ids[index] = layer.arc_ids[choice]
    return DecodedTrip(arc_ids, choices, tuple(breaks))


def decode_classifier(lattice: TripLattice) -> DecodedTrip:
    """Choose the best emission at each point with the shared rank tie-break."""
    choices = np.array([np.argmax(layer.emissions) if len(layer.emissions) else -1 for layer in lattice.layers])
    breaks = [
        reason if choice >= 0 else "unmatched"
        for reason, choice in zip(lattice.break_before, choices, strict=True)
    ]
    return _decoded_trip(lattice, choices, breaks)


def decode_viterbi(lattice: TripLattice, beta_m: float) -> DecodedTrip:
    """Maximize emission score plus negative road/GPS distance mismatch."""
    if not np.isfinite(beta_m) or beta_m <= 0:
        raise ValueError("beta_m must be finite and positive")
    choices = np.full(len(lattice.layers), -1, dtype=np.int64)
    breaks = list(lattice.break_before)
    parents: list[np.ndarray | None] = [None] * len(choices)
    totals = None

    def finish_segment(last_point: int, scores: np.ndarray) -> None:
        choice = int(np.argmax(scores))
        for point in range(last_point, -1, -1):
            choices[point] = choice
            if parents[point] is None:
                break
            choice = int(parents[point][choice])

    for point, layer in enumerate(lattice.layers):
        if not len(layer.arc_ids):
            if totals is not None:
                finish_segment(point - 1, totals)
            breaks[point] = "unmatched"
            totals = None
            continue
        matrix = lattice.distances[point]
        if totals is not None and matrix is not None:
            transitions = totals[:, None] - np.abs(matrix - lattice.gps_step_m[point]) / beta_m
            best_parents = np.argmax(transitions, axis=0)
            best_scores = transitions[best_parents, np.arange(len(best_parents))]
            if np.isfinite(best_scores).any():
                totals = layer.emissions + best_scores
                parents[point] = best_parents
                continue
            breaks[point] = "no_transition"
        if totals is not None:
            finish_segment(point - 1, totals)
        totals = layer.emissions
    if totals is not None:
        finish_segment(len(choices) - 1, totals)
    return _decoded_trip(lattice, choices, breaks)

import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from scripts.verify_data import extract_allowed
from src.data.geometry import SegmentIndex, check_trip, segment_distances, load_geometry
from src.data.reference import monotone_alignment
from src.data.features import build_inference_table
from src.data.visualize import review_map
import pandas as pd


class DataStageTests(unittest.TestCase):
    def test_review_map_has_interactive_basemap_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "train" / "00000001"
            folder.mkdir(parents=True)
            for ext, content in {"track": "8.58 49.85 0\n", "nodes": "8.58 49.85\n8.581 49.85\n", "arcs": "0 1\n"}.items():
                (folder / f"00000001.{ext}").write_text(content)
            g = load_geometry("train", "00000001", root=Path(tmp))
            a = pd.DataFrame({"point_idx": [0], "reference_arc_id": [0], "route_occurrence": [0],
                              "reference_fraction": [0.], "reference_distance_m": [0.], "reference_far": [False]})
            output = Path(tmp) / "map.html"
            review_map(g, np.array([0]), a, np.array([0]), output)
            html = output.read_text(encoding="utf-8")
            self.assertIn("OpenStreetMap", html)
            self.assertIn("Esri Satellite", html)
            self.assertIn("Far Reference Matches", html)

    def test_exact_nearest_matches_brute_force(self):
        rng = np.random.default_rng(7)
        starts = rng.uniform(-100, 100, (80, 2))
        ends = starts + rng.uniform(-200, 200, (80, 2))
        ends[0] = starts[0]  # zero length
        starts[1], ends[1] = ends[2].copy(), starts[2].copy()  # reverse direction
        index = SegmentIndex(starts, ends)
        for point in rng.uniform(-300, 300, (20, 2)):
            expected, _ = segment_distances(point, starts, ends)
            order = np.lexsort((np.arange(len(starts)), expected))[:16]
            actual, distances, _ = index.nearest(point, 16)
            np.testing.assert_array_equal(actual, order)
            np.testing.assert_allclose(distances, expected[order])

    def test_alignment_matches_exhaustive_monotone_search(self):
        starts = np.array([[0., 0.], [1., 0.], [0., 0.]])
        ends = starts + [1., 0.]
        points = np.array([[.2, .1], [1.8, .1], [.1, .1]])
        costs = np.array([.5 * segment_distances(p, starts, ends)[0] ** 2 for p in points])
        legal = list(itertools.combinations_with_replacement(range(3), 3))
        minimum = min(sum(costs[i, j] for i, j in enumerate(path)) for path in legal)
        states = monotone_alignment(points, starts, ends)
        self.assertAlmostEqual(sum(costs[i, j] for i, j in enumerate(states)), minimum)
        self.assertTrue((np.diff(states) >= 0).all())
        self.assertEqual(states.tolist(), [0, 1, 2])  # repeated physical segment

    def test_inference_without_route_and_partition_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "train" / "00000001"
            folder.mkdir(parents=True)
            for ext, content in {
                "track": "8.58 49.85 0\n8.5801 49.85 1\n8.5802 49.85 100\n",
                "nodes": "8.58 49.85\n8.581 49.85\n",
                "arcs": "0 1\n1 0\n",
            }.items():
                (folder / f"00000001.{ext}").write_text(content)
            with patch("src.data.loader.read_route", side_effect=AssertionError("Route access")):
                table = build_inference_table("train", "00000001", k=4, root=Path(tmp))
            self.assertEqual(len(table), 6)
            self.assertNotIn("label", table)
            self.assertTrue(np.isfinite(table.select_dtypes("number")).all().all())
            self.assertTrue(table.loc[table.point_idx == 2, "large_gap"].all())
            self.assertFalse(table.loc[table.point_idx == 2, "heading_valid"].any())
        for partition, trip in [("evaluator_test", "00000001"), ("train", "../other")]:
            with self.assertRaises(ValueError):
                check_trip(partition, trip)

    def test_archive_allowlist_and_preserve_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "synthetic.zip"
            root = Path(tmp) / "raw"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("Student folder/train/00000001/00000001.track", "8 49 0\n")
                z.writestr("Student folder/evaluator_test/00000002/00000002.track", "DO NOT READ")
                z.writestr("Student folder/train/../../escape.track", "NO")
            original = zipfile.ZipFile.open
            def guarded_open(instance, member, *args, **kwargs):
                name = member.filename if isinstance(member, zipfile.ZipInfo) else member
                if "evaluator" in name or ".." in name:
                    raise AssertionError("Forbidden archive member opened")
                return original(instance, member, *args, **kwargs)
            with patch.object(zipfile.ZipFile, "open", guarded_open):
                self.assertEqual(extract_allowed(archive, root), 1)
                self.assertEqual(extract_allowed(archive, root), 1)
            target = root / "train/00000001/00000001.track"
            target.write_text("changed")
            with self.assertRaises(ValueError):
                extract_allowed(archive, root)
            self.assertEqual(target.read_text(), "changed")


if __name__ == "__main__":
    unittest.main()

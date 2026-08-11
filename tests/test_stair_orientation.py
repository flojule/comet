#!/usr/bin/env python3
# tests/test_stair_orientation.py
"""Tests for stair orientation estimation.

Synthetic staircases are drawn at known angles, so the estimator's output can
be checked against ground truth rather than eyeballed.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_bag import stair_frame                            # noqa: E402
from stair_orientation import (                             # noqa: E402
    Orientation,
    OrientationConfig,
    annotate,
    circular_mean_180,
    estimate_frame,
    smooth_series,
    summarise,
)

W, H = 320, 240


def full_mask(w: int = W, h: int = H) -> np.ndarray:
    """Whole-frame mask, as if SAM 3 segmented the entire staircase."""
    return np.ones((h, w), dtype=bool)


def angle_error(got: float, want: float) -> float:
    """Smallest difference between two lines' angles, in [0, 90]."""
    return abs(((got - want + 90.0) % 180.0) - 90.0)


class TestCircularMean(unittest.TestCase):
    def test_wraps_across_zero(self):
        # The reason this helper exists: a plain mean of 179 and 1 gives 90,
        # which is perpendicular to both inputs.
        self.assertLess(angle_error(circular_mean_180([179.0, 1.0]), 0.0), 1.0)

    def test_plain_case(self):
        self.assertAlmostEqual(circular_mean_180([20.0, 22.0, 24.0]), 22.0, delta=0.5)

    def test_weights_pull_the_mean(self):
        m = circular_mean_180([10.0, 50.0], [10.0, 1.0])
        self.assertLess(m, 20.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            circular_mean_180([])


class TestEstimateFrame(unittest.TestCase):
    def test_recovers_known_angles(self):
        for want in (0.0, 20.0, 45.0, 70.0, 90.0, 120.0, 160.0):
            img = stair_frame(W, H, 0, angle_deg=want)
            o = estimate_frame(img, full_mask(), 0)
            self.assertTrue(o.ok, f"no estimate at {want}deg")
            self.assertLess(angle_error(o.angle_deg, want), 6.0,
                            f"got {o.angle_deg:.1f} want {want}")

    def test_angle_is_wrapped_to_0_180(self):
        for want in (0.0, 45.0, 170.0):
            o = estimate_frame(stair_frame(W, H, 0, want), full_mask(), 0)
            self.assertGreaterEqual(o.angle_deg, 0.0)
            self.assertLess(o.angle_deg, 180.0)

    def test_confidence_high_on_clean_stairs(self):
        o = estimate_frame(stair_frame(W, H, 0, 30.0), full_mask(), 0)
        self.assertGreater(o.confidence, 0.5)
        self.assertGreaterEqual(o.n_segments, 3)

    def test_empty_mask_gives_no_estimate(self):
        o = estimate_frame(stair_frame(W, H, 0, 20.0),
                           np.zeros((H, W), dtype=bool), 7)
        self.assertFalse(o.ok)
        self.assertEqual(o.frame, 7)
        self.assertEqual(o.confidence, 0.0)

    def test_none_mask_gives_no_estimate(self):
        self.assertFalse(estimate_frame(stair_frame(W, H, 0), None, 0).ok)

    def test_featureless_frame_gives_no_estimate(self):
        # Flat grey: no edges at all, so there is nothing to be confident about.
        flat = np.full((H, W, 3), 128, dtype=np.uint8)
        self.assertFalse(estimate_frame(flat, full_mask(), 0).ok)

    def test_random_texture_is_rejected_not_guessed(self):
        # Isotropic noise has no dominant direction; returning an angle anyway
        # would be a confident lie.
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        o = estimate_frame(noise, full_mask(), 0)
        if o.ok:
            self.assertLess(o.confidence, 0.6)

    def test_mask_restricts_the_search(self):
        # Stairs at 20 deg on the left, a strong vertical grating on the right.
        img = stair_frame(W, H, 0, angle_deg=20.0)
        img[:, W // 2:] = 60
        for x in range(W // 2, W, 8):
            img[:, x:x + 3] = 230

        left = np.zeros((H, W), dtype=bool)
        left[:, :W // 2] = True
        o = estimate_frame(img, left, 0)
        self.assertTrue(o.ok)
        self.assertLess(angle_error(o.angle_deg, 20.0), 8.0)

    def test_segments_are_returned_for_drawing(self):
        o = estimate_frame(stair_frame(W, H, 0, 20.0), full_mask(), 0)
        self.assertTrue(o.segments)
        for seg in o.segments:
            self.assertEqual(len(seg), 4)

    def test_competing_directions_lower_the_confidence(self):
        # Two perpendicular line families in equal measure: an answer is still
        # returned, but the confidence must show that half the evidence
        # disagrees, so a caller can tell this from a clean read.
        import cv2
        cross = cv2.addWeighted(stair_frame(W, H, 0, 20.0), 0.5,
                                stair_frame(W, H, 0, 110.0), 0.5, 0)
        ambiguous = estimate_frame(cross, full_mask(), 0)
        clean = estimate_frame(stair_frame(W, H, 0, 20.0), full_mask(), 0)
        self.assertGreater(clean.confidence, 0.9)
        self.assertLess(ambiguous.confidence, 0.7)

    def test_min_dominant_share_rejects_ambiguous_frames(self):
        import cv2
        cross = cv2.addWeighted(stair_frame(W, H, 0, 20.0), 0.5,
                                stair_frame(W, H, 0, 110.0), 0.5, 0)
        strict = OrientationConfig(min_dominant_share=0.8)
        self.assertFalse(estimate_frame(cross, full_mask(), 0, strict).ok)
        # The same frame passes at the default threshold.
        self.assertTrue(estimate_frame(cross, full_mask(), 0).ok)


class TestSmoothing(unittest.TestCase):
    def _series(self, angles):
        return [Orientation(i, a, 0.8, 5, 100.0) if a is not None
                else Orientation(i, None, 0.0, 0, 0.0)
                for i, a in enumerate(angles)]

    def test_smoothing_reduces_jitter(self):
        noisy = [20.0, 26.0, 14.0, 22.0, 18.0, 24.0, 16.0, 21.0, 19.0]
        out = smooth_series(self._series(noisy), window=5)
        spread = max(o.angle_deg for o in out) - min(o.angle_deg for o in out)
        self.assertLess(spread, max(noisy) - min(noisy))

    def test_smoothing_handles_the_wrap(self):
        out = smooth_series(self._series([179.0, 0.0, 1.0, 179.5, 0.5]), window=5)
        for o in out:
            self.assertLess(angle_error(o.angle_deg, 0.0), 5.0)

    def test_bad_frames_stay_bad(self):
        # A frame with no estimate must not acquire one from its neighbours.
        out = smooth_series(self._series([20.0, None, 22.0, 21.0, None]), window=5)
        self.assertIsNone(out[1].angle_deg)
        self.assertIsNone(out[4].angle_deg)

    def test_window_below_three_is_a_noop(self):
        s = self._series([20.0, 40.0, 60.0])
        self.assertEqual([o.angle_deg for o in smooth_series(s, 0)],
                         [20.0, 40.0, 60.0])


class TestSummarise(unittest.TestCase):
    def test_reports_mean_spread_and_coverage(self):
        s = summarise([Orientation(i, 20.0 + (i % 3), 0.8, 5, 99.0)
                       for i in range(10)])
        self.assertEqual(s["frames"], 10)
        self.assertEqual(s["frames_with_estimate"], 10)
        self.assertEqual(s["coverage"], 1.0)
        self.assertLess(angle_error(s["angle_deg_mean"], 21.0), 1.5)
        self.assertLess(s["angle_deg_std"], 2.0)

    def test_partial_coverage(self):
        rs = [Orientation(0, 20.0, 0.8, 5, 9.0),
              Orientation(1, None, 0.0, 0, 0.0),
              Orientation(2, 22.0, 0.8, 5, 9.0),
              Orientation(3, None, 0.0, 0, 0.0)]
        s = summarise(rs)
        self.assertEqual(s["coverage"], 0.5)
        self.assertIsNone(s["per_frame"]["1"])

    def test_no_estimates_at_all(self):
        s = summarise([Orientation(i, None, 0.0, 0, 0.0) for i in range(3)])
        self.assertEqual(s["coverage"], 0.0)
        self.assertIsNone(s["angle_deg_mean"])

    def test_std_is_small_across_the_wrap(self):
        # 179 and 1 are 2 degrees apart, not 178.
        s = summarise([Orientation(0, 179.0, 0.9, 5, 9.0),
                       Orientation(1, 1.0, 0.9, 5, 9.0)])
        self.assertLess(s["angle_deg_std"], 3.0)


class TestAnnotate(unittest.TestCase):
    def test_draws_without_mutating_input(self):
        img = stair_frame(W, H, 0, 20.0)
        before = img.copy()
        o = estimate_frame(img, full_mask(), 0)
        out = annotate(img, o)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.array_equal(img, before))
        self.assertFalse(np.array_equal(out, img))

    def test_handles_a_frame_with_no_estimate(self):
        img = np.full((H, W, 3), 128, dtype=np.uint8)
        out = annotate(img, Orientation(0, None, 0.0, 0, 0.0))
        self.assertEqual(out.shape, img.shape)


if __name__ == "__main__":
    unittest.main()

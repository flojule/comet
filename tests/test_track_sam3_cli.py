#!/usr/bin/env python3
# tests/test_track_sam3_cli.py
"""End-to-end tests of track_sam3.main() with the model swapped for a fake.

Covers the whole CLI path — ROI loading, prompting, propagation, quality
report, gap fill, debug video, tracking JSON and mask sidecar — and then feeds
the result to render.py to prove the two really do interoperate.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2                                                  # noqa: E402
from fake_sam3 import FakePredictor                         # noqa: E402

import render                                               # noqa: E402
import track_sam3                                           # noqa: E402
from maskstore import MaskStore, sidecar_path               # noqa: E402
from test_sam3_backend import make_video                    # noqa: E402


class CliHarness(unittest.TestCase):
    """Temp project with a video and rois.json, and a patched predictor."""

    N_FRAMES = 40
    W, H = 160, 120

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "clip.mp4"
        make_video(self.video, self.N_FRAMES, self.W, self.H)

        self.rois = self.dir / "rois.json"
        self.rois.write_text(json.dumps({
            "start_frame": 10,
            "end_frame": 29,
            "rois": {"cf1": [35, 25, 10, 10], "cf2": [75, 55, 10, 10]},
        }))
        self.out = self.dir / "out" / "clip_sam3.mp4"
        self.fake = FakePredictor(height=self.H, width=self.W)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, extra: list[str] | None = None, fake=None):
        argv = ["--video", str(self.video), "--rois", str(self.rois),
                "--out", str(self.out)] + (extra or [])
        args = track_sam3.parse_args(argv)
        with mock.patch.object(track_sam3, "build_predictor",
                               return_value=fake or self.fake):
            track_sam3.main(args)
        return json.loads(self.tracking_json.read_text())

    @property
    def tracking_json(self) -> Path:
        return self.out.with_name(self.out.stem + "_tracking.json")


class TestFromRois(CliHarness):
    def test_writes_tracking_json_with_named_agents(self):
        d = self.run_cli(["--no-debug-video"])
        self.assertEqual(sorted(d["trails"]), ["cf1", "cf2"])
        self.assertEqual(d["tracker"], "sam3")
        self.assertEqual(d["sam3_prompt"], {"rois": ["cf1", "cf2"]})

    def test_trail_frames_are_absolute_and_inside_the_window(self):
        d = self.run_cli(["--no-debug-video"])
        frames = sorted(int(f) for f in d["trails"]["cf1"])
        self.assertEqual(frames[0], 10)
        self.assertEqual(frames[-1], 29)
        self.assertEqual(len(frames), 20)

    def test_video_metadata_is_the_full_source_not_the_window(self):
        # render.py iterates range(total_frames) over the ORIGINAL video, so
        # these must describe the source, not the extracted sub-range.
        d = self.run_cli(["--no-debug-video"])
        self.assertEqual(d["total_frames"], self.N_FRAMES)
        self.assertEqual((d["width"], d["height"]), (self.W, self.H))
        self.assertEqual((d["start_frame"], d["end_frame"]), (10, 29))

    def test_every_agent_has_a_colour(self):
        d = self.run_cli(["--no-debug-video"])
        for name in d["trails"]:
            self.assertIn(name, d["trail_color"])

    def test_quality_stats_recorded(self):
        d = self.run_cli(["--no-debug-video"])
        self.assertEqual(d["sam3_stats"]["cf1"]["coverage"], 1.0)
        self.assertEqual(d["sam3_stats"]["cf1"]["longest_gap"], 0)
        self.assertGreater(d["sam3_stats"]["cf1"]["area_median"], 0)

    def test_agents_subset(self):
        d = self.run_cli(["--no-debug-video", "--agents", "cf1"])
        self.assertEqual(sorted(d["trails"]), ["cf1"])

    def test_frame_range_override(self):
        d = self.run_cli(["--no-debug-video", "--start-frame", "12",
                          "--end-frame", "17"])
        frames = sorted(int(f) for f in d["trails"]["cf1"])
        self.assertEqual((frames[0], frames[-1]), (12, 17))

    def test_debug_video_is_written(self):
        self.run_cli()
        dbg = self.out.with_name(self.out.stem + "_debug.mp4")
        self.assertTrue(dbg.exists())
        cap = cv2.VideoCapture(str(dbg))
        self.assertGreater(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        cap.release()

    def test_point_prompt_shape(self):
        d = self.run_cli(["--no-debug-video", "--prompt-shape", "point"])
        self.assertTrue(any("points" in r for r in self.fake.requests))
        self.assertEqual(sorted(d["trails"]), ["cf1", "cf2"])

    def test_reprompt_every_chunks_the_run(self):
        d = self.run_cli(["--no-debug-video", "--reprompt-every", "5"])
        starts = [r for r in self.fake.requests if r["type"] == "start_session"]
        self.assertGreater(len(starts), 1)
        self.assertEqual(len(d["trails"]["cf1"]), 20)


class TestGapFill(CliHarness):
    def test_dropouts_are_interpolated_by_default(self):
        fake = FakePredictor(height=self.H, width=self.W,
                             drop_frames={i: {0} for i in range(4, 12)})
        d = self.run_cli(["--no-debug-video"], fake=fake)
        frames = sorted(int(f) for f in d["trails"]["cf1"])
        self.assertEqual(frames, list(range(10, 30)))
        self.assertLess(d["sam3_stats"]["cf1"]["coverage"], 1.0)
        self.assertEqual(d["sam3_stats"]["cf1"]["longest_gap"], 8)

    def test_no_gap_fill_leaves_the_hole(self):
        fake = FakePredictor(height=self.H, width=self.W,
                             drop_frames={i: {0} for i in range(4, 12)})
        d = self.run_cli(["--no-debug-video", "--no-gap-fill"], fake=fake)
        frames = sorted(int(f) for f in d["trails"]["cf1"])
        self.assertNotIn(15, frames)
        self.assertEqual(len(frames), 12)


class TestTextMode(CliHarness):
    def test_discovered_objects_get_default_names(self):
        d = self.run_cli(["--no-debug-video", "--text", "drone"])
        self.assertEqual(sorted(d["trails"]), ["obj0", "obj1"])
        self.assertEqual(d["sam3_prompt"], {"text": "drone"})

    def test_names_can_be_mapped(self):
        d = self.run_cli(["--no-debug-video", "--text", "drone",
                          "--name", "0=cf1", "--name", "1=cf2"])
        self.assertEqual(sorted(d["trails"]), ["cf1", "cf2"])
        # Mapped names must pick up the project palette, not a fallback colour.
        self.assertEqual(d["trail_color"]["cf1"], [0, 0, 255])

    def test_discovered_names_all_get_distinct_colours(self):
        d = self.run_cli(["--no-debug-video", "--text", "drone"])
        cols = [tuple(c) for c in d["trail_color"].values()]
        self.assertEqual(len(cols), len(set(cols)))


class TestMasksAndRender(CliHarness):
    def test_mask_sidecar_written_and_loadable(self):
        self.run_cli(["--no-debug-video", "--save-masks"])
        p = sidecar_path(self.tracking_json)
        self.assertTrue(p.exists())
        store = MaskStore.load(p)
        self.assertEqual(sorted(store.names()), ["cf1", "cf2"])
        m = store.get(10, "cf1")
        self.assertEqual(m.shape, (self.H, self.W))
        self.assertTrue(m.any())

    def test_render_consumes_the_output(self):
        self.run_cli(["--no-debug-video", "--save-masks"])
        render.render(str(self.tracking_json), mask_mode="off")
        for suffix in ("_persistent.mp4", "_transient.mp4"):
            p = self.out.parent / ("clip_sam3" + suffix)
            self.assertTrue(p.exists(), p)

    def test_render_mask_modes_run(self):
        self.run_cli(["--no-debug-video", "--save-masks"])
        for mode in ("occlude", "glow"):
            render.render(str(self.tracking_json), mask_mode=mode)
            p = self.out.parent / "clip_sam3_transient.mp4"
            self.assertTrue(p.exists())

    def test_missing_sidecar_falls_back_instead_of_crashing(self):
        self.run_cli(["--no-debug-video"])          # no --save-masks
        render.render(str(self.tracking_json), mask_mode="glow")
        self.assertTrue((self.out.parent / "clip_sam3_persistent.mp4").exists())


class TestCompositeMasks(unittest.TestCase):
    def test_object_pixels_are_restored_over_the_trail(self):
        base = np.full((20, 20, 3), 30, dtype=np.uint8)
        blended = np.full((20, 20, 3), 200, dtype=np.uint8)   # trail everywhere
        m = np.zeros((20, 20), dtype=bool)
        m[5:10, 5:10] = True
        out = render.composite_masks(base, blended, {"cf1": m},
                                     {"cf1": (0, 0, 255)}, "occlude")
        self.assertTrue((out[5:10, 5:10] == 30).all())    # object shows through
        self.assertTrue((out[0, 0] == 200).all())         # trail survives

    def test_off_mode_is_a_passthrough(self):
        base = np.zeros((8, 8, 3), np.uint8)
        blended = np.full((8, 8, 3), 99, np.uint8)
        m = np.ones((8, 8), dtype=bool)
        out = render.composite_masks(base, blended, {"a": m}, {}, "off")
        self.assertTrue((out == 99).all())


class TestHybridBorrow(CliHarness):
    def test_borrowed_agent_comes_from_the_other_tracker(self):
        donor = self.dir / "donor_tracking.json"
        donor.write_text(json.dumps({
            "trails": {"cf2": {str(f): [f, 99] for f in range(10, 30)}}}))
        d = self.run_cli(["--no-debug-video", "--borrow-agents", "cf2",
                          "--borrow-from", str(donor)])
        self.assertEqual(sorted(d["trails"]), ["cf1", "cf2"])
        self.assertEqual(d["trails"]["cf2"]["15"], [15, 99])
        # The borrowed agent must not have been prompted into SAM 3.
        self.assertEqual(d["sam3_prompt"], {"rois": ["cf1"]})

    def test_unknown_borrowed_agent_is_an_error(self):
        donor = self.dir / "donor_tracking.json"
        donor.write_text(json.dumps({"trails": {"other": {}}}))
        with self.assertRaises(KeyError):
            self.run_cli(["--no-debug-video", "--borrow-agents", "cf2",
                          "--borrow-from", str(donor)])

    def test_borrow_without_source_exits(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["--no-debug-video", "--borrow-agents", "cf2"])


class TestArgValidation(CliHarness):
    def test_unknown_agent_exits(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["--agents", "nope"])

    def test_bad_name_mapping_exits(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["--text", "drone", "--name", "cf1"])

    def test_zoom_on_unknown_agent_exits(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["--no-debug-video", "--zoom-agents", "nope"])


class TestZoomPass(CliHarness):
    def test_zoom_results_land_in_full_frame_coordinates(self):
        d = self.run_cli(["--no-debug-video", "--zoom-agents", "cf1",
                          "--zoom-crop", "80", "--zoom-upscale", "2",
                          "--zoom-segment", "10"])
        pts = [tuple(p) for p in d["trails"]["cf1"].values()]
        self.assertTrue(pts)
        for x, y in pts:
            self.assertTrue(0 <= x < self.W, (x, y))
            self.assertTrue(0 <= y < self.H, (x, y))


if __name__ == "__main__":
    unittest.main()

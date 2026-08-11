#!/usr/bin/env python3
# tests/test_stairs_pipeline.py
"""End-to-end tests of the mcap → SAM 3 → overlay/orientation pipeline.

SAM 3 is swapped for the fake predictor, so this exercises every stage except
the model weights themselves: bag reading, frame extraction, mask storage,
overlay rendering, orientation measurement and the output JSON.
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
from make_bag import make_split_recording, stair_frame      # noqa: E402

import stairs_pipeline                                      # noqa: E402
from media import classify, resolve                         # noqa: E402

COLOR_TOPIC = "/camera/color/image_raw"


class TestMediaResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_classify(self):
        (self.dir / "a.mp4").write_bytes(b"x")
        self.assertEqual(classify(self.dir / "a.mp4"), "video")
        cv2.imwrite(str(self.dir / "b.png"), np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(classify(self.dir / "b.png"), "image")
        bags = self.dir / "bags"
        make_split_recording(bags, splits=1, frames_per_split=2,
                             extra_topics=False)
        self.assertEqual(classify(bags), "mcap")

    def test_classify_rejects_junk(self):
        (self.dir / "notes.txt").write_text("hello")
        with self.assertRaises(ValueError):
            classify(self.dir / "notes.txt")
        with self.assertRaises(ValueError):
            classify(self.dir)          # folder with no images or bags
        with self.assertRaises(FileNotFoundError):
            classify(self.dir / "missing")

    def test_single_photo_becomes_a_short_clip(self):
        p = self.dir / "photo.jpg"
        cv2.imwrite(str(p), stair_frame(64, 48, 0, 20.0))
        with resolve(p, still_frames=5) as m:
            self.assertEqual(m.kind, "image")
            self.assertEqual(m.frames, 5)
            self.assertEqual((m.width, m.height), (64, 48))
            self.assertTrue(Path(m.video).exists())

    def test_folder_of_photos_becomes_a_clip_in_name_order(self):
        for i in range(4):
            cv2.imwrite(str(self.dir / f"img_{i:03d}.png"),
                        stair_frame(64, 48, i, 20.0))
        with resolve(self.dir, fps=10.0) as m:
            self.assertEqual(m.kind, "images")
            self.assertEqual(m.frames, 4)
            self.assertEqual(m.fps, 10.0)
            self.assertEqual(m.detail["files"][0], "img_000.png")

    def test_mixed_size_photos_are_resized_not_dropped(self):
        cv2.imwrite(str(self.dir / "a.png"), np.zeros((48, 64, 3), np.uint8))
        cv2.imwrite(str(self.dir / "b.png"), np.zeros((96, 128, 3), np.uint8))
        with resolve(self.dir) as m:
            self.assertEqual(m.frames, 2)

    def test_mcap_needs_a_topic(self):
        make_split_recording(self.dir, splits=1, frames_per_split=2,
                             extra_topics=False)
        with self.assertRaises(ValueError) as ctx:
            resolve(self.dir)
        self.assertIn("topic", str(ctx.exception))

    def test_mcap_resolves_with_a_topic(self):
        make_split_recording(self.dir, splits=2, frames_per_split=3,
                             w=64, h=48, extra_topics=False)
        with resolve(self.dir, topic=COLOR_TOPIC) as m:
            self.assertEqual(m.kind, "mcap")
            self.assertEqual(m.frames, 6)

    def test_temp_video_is_cleaned_up(self):
        p = self.dir / "photo.jpg"
        cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))
        m = resolve(p, still_frames=2)
        v = Path(m.video)
        self.assertTrue(v.exists())
        m.cleanup()
        self.assertFalse(v.exists())


class TestPipeline(unittest.TestCase):
    W, H = 160, 120

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.bags = self.dir / "bags"
        make_split_recording(self.bags, splits=2, frames_per_split=6,
                             w=self.W, h=self.H, angle_deg=20.0,
                             extra_topics=True)
        self.out = self.dir / "out" / "stairs"
        self.fake = FakePredictor(height=self.H, width=self.W, radius=30,
                                  velocity=(0.0, 0.0))

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **kw):
        logs: list[str] = []
        with mock.patch.object(stairs_pipeline, "build_predictor",
                               return_value=self.fake):
            res = stairs_pipeline.run(
                self.bags, topic=COLOR_TOPIC, prompt="stairs",
                out_stem=self.out, log=logs.append, **kw)
        return res, logs

    def test_runs_end_to_end_and_writes_outputs(self):
        res, _ = self._run()
        self.assertEqual(res.frames, 12)
        self.assertTrue(Path(res.data).exists())
        self.assertTrue(Path(res.overlay).exists())
        self.assertTrue((self.out.parent / "stairs_masks.npz").exists())

    def test_objects_named_after_the_prompt(self):
        res, _ = self._run()
        self.assertTrue(res.objects)
        for name in res.objects.values():
            self.assertTrue(name.startswith("stairs"), name)

    def test_json_records_the_provenance(self):
        res, _ = self._run()
        d = json.loads(Path(res.data).read_text())
        self.assertEqual(d["topic"], COLOR_TOPIC)
        self.assertEqual(d["prompt"], "stairs")
        self.assertEqual(d["frames"], 12)
        self.assertEqual((d["width"], d["height"]), (self.W, self.H))
        self.assertIn("orientation", d)
        self.assertIn("coverage", d)

    def test_overlay_video_has_every_frame(self):
        res, _ = self._run()
        cap = cv2.VideoCapture(res.overlay)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertEqual(n, 12)

    def test_orientation_recovers_the_synthetic_angle(self):
        # The bag was drawn with 20-degree nosings; the pipeline should say so.
        res, _ = self._run()
        mean = res.orientation.get("angle_deg_mean")
        self.assertIsNotNone(mean)
        err = abs(((mean - 20.0 + 90.0) % 180.0) - 90.0)
        self.assertLess(err, 8.0, f"got {mean}")

    def test_orientation_can_be_disabled(self):
        res, _ = self._run(orientation=False)
        self.assertEqual(res.orientation, {})

    def test_overlay_can_be_disabled(self):
        res, _ = self._run(overlay=False)
        self.assertIsNone(res.overlay)

    def test_max_frames_limits_the_run(self):
        res, _ = self._run(max_frames=5)
        self.assertEqual(res.frames, 5)

    def test_coverage_is_reported_per_object(self):
        res, _ = self._run()
        self.assertTrue(res.coverage)
        for cov in res.coverage.values():
            self.assertGreaterEqual(cov, 0.0)
            self.assertLessEqual(cov, 1.0)

    def test_logs_mention_the_key_stages(self):
        _, logs = self._run()
        blob = "\n".join(logs)
        for expected in ("Resolving input", "SAM 3", "orientation", "Data"):
            self.assertIn(expected, blob)

    def test_nothing_found_is_reported_not_crashed(self):
        empty = FakePredictor(height=self.H, width=self.W)
        empty.handle_request = _no_detections(empty)
        logs: list[str] = []
        with mock.patch.object(stairs_pipeline, "build_predictor",
                               return_value=empty):
            res = stairs_pipeline.run(
                self.bags, topic=COLOR_TOPIC, prompt="unicorn",
                out_stem=self.out, log=logs.append)
        self.assertEqual(res.objects, {})
        self.assertIn("found nothing", "\n".join(logs))

    def test_works_on_a_plain_video_with_no_topic(self):
        vid = self.dir / "clip.mp4"
        wr = cv2.VideoWriter(str(vid), cv2.VideoWriter.fourcc(*"mp4v"), 30.0,
                             (self.W, self.H))
        for i in range(8):
            wr.write(stair_frame(self.W, self.H, i, 45.0))
        wr.release()
        logs: list[str] = []
        with mock.patch.object(stairs_pipeline, "build_predictor",
                               return_value=self.fake):
            res = stairs_pipeline.run(vid, prompt="stairs", out_stem=self.out,
                                      log=logs.append)
        self.assertEqual(res.frames, 8)
        mean = res.orientation.get("angle_deg_mean")
        self.assertIsNotNone(mean)
        self.assertLess(abs(((mean - 45.0 + 90.0) % 180.0) - 90.0), 8.0)

    def test_works_on_a_single_photo(self):
        p = self.dir / "one.png"
        cv2.imwrite(str(p), stair_frame(self.W, self.H, 0, 70.0))
        with mock.patch.object(stairs_pipeline, "build_predictor",
                               return_value=self.fake):
            res = stairs_pipeline.run(p, prompt="stairs", out_stem=self.out,
                                      log=lambda *_: None)
        self.assertGreater(res.frames, 0)


def _no_detections(fp):
    """Wrap a FakePredictor so its text prompt seeds no objects."""
    original = fp.handle_request

    def handler(request):
        if request.get("type") == "add_prompt" and "text" in request:
            return {"outputs": None}            # detector finds nothing
        return original(request)
    return handler


class TestStageSplit(unittest.TestCase):
    """The GPU boundary: segment needs CUDA, analyse does not."""

    W, H = 160, 120

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.bags = self.dir / "bags"
        make_split_recording(self.bags, splits=2, frames_per_split=6,
                             w=self.W, h=self.H, angle_deg=35.0,
                             extra_topics=False)
        self.out = self.dir / "out" / "stairs"
        self.fake = FakePredictor(height=self.H, width=self.W, radius=30,
                                  velocity=(0.0, 0.0))

    def tearDown(self):
        self.tmp.cleanup()

    def _segment(self, **kw):
        with mock.patch.object(stairs_pipeline, "build_predictor",
                               return_value=self.fake):
            return stairs_pipeline.segment(
                self.bags, topic=COLOR_TOPIC, prompt="stairs",
                out_stem=self.out, log=lambda *_: None, **kw)

    def test_segment_writes_a_complete_handoff(self):
        self._segment()
        for suffix in ("_stairs.json", "_masks.npz", "_frames.mp4"):
            p = self.out.with_name(self.out.name + suffix)
            self.assertTrue(p.exists(), suffix)

    def test_segment_does_no_orientation_or_overlay(self):
        data = self._segment()
        self.assertEqual(data["orientation"], {})
        self.assertFalse((self.out.with_name(self.out.name + "_overlay.mp4")).exists())

    def test_analyse_runs_from_artifacts_alone(self):
        self._segment()
        # Deleting the bags proves analyse needs nothing but stage 1's output —
        # this is what lets it run on a different machine.
        import shutil
        shutil.rmtree(self.bags)

        res = stairs_pipeline.analyse(self.out, log=lambda *_: None)
        self.assertTrue(Path(res.overlay).exists())
        self.assertIsNotNone(res.orientation.get("angle_deg_mean"))

    def test_analyse_needs_no_predictor_at_all(self):
        self._segment()
        # build_predictor raising proves stage 2 never touches the GPU path.
        def explode(*a, **k):
            raise AssertionError("analyse must not build a predictor")
        with mock.patch.object(stairs_pipeline, "build_predictor", explode):
            stairs_pipeline.analyse(self.out, log=lambda *_: None)

    def test_split_matches_a_single_machine_run(self):
        self._segment()
        split = stairs_pipeline.analyse(self.out, log=lambda *_: None)

        whole_stem = self.dir / "out2" / "stairs"
        with mock.patch.object(stairs_pipeline, "build_predictor",
                               return_value=self.fake):
            whole = stairs_pipeline.run(
                self.bags, topic=COLOR_TOPIC, prompt="stairs",
                out_stem=whole_stem, log=lambda *_: None)
        self.assertEqual(split.frames, whole.frames)
        self.assertAlmostEqual(split.orientation["angle_deg_mean"],
                               whole.orientation["angle_deg_mean"], delta=0.01)

    def test_analyse_updates_the_json_in_place(self):
        self._segment()
        stairs_pipeline.analyse(self.out, log=lambda *_: None)
        d = json.loads((self.out.with_name(self.out.name + "_stairs.json")).read_text())
        self.assertIsNotNone(d["orientation"].get("angle_deg_mean"))

    def test_analyse_is_rerunnable_with_new_settings(self):
        # The point of the split: retuning orientation must not need the model.
        self._segment()
        import stair_orientation as so
        a = stairs_pipeline.analyse(self.out, log=lambda *_: None)
        b = stairs_pipeline.analyse(
            self.out, log=lambda *_: None,
            orientation_cfg=so.OrientationConfig(min_dominant_share=0.99))
        self.assertIsNotNone(a.orientation.get("angle_deg_mean"))
        self.assertLessEqual(b.orientation.get("coverage", 1.0),
                             a.orientation.get("coverage", 1.0))

    def test_analyse_without_segment_says_so(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            stairs_pipeline.analyse(self.dir / "never" / "ran")
        self.assertIn("segment", str(ctx.exception))

    def test_analyse_without_frames_video_says_so(self):
        self._segment(keep_video=False)
        with self.assertRaises(FileNotFoundError) as ctx:
            stairs_pipeline.analyse(self.out, log=lambda *_: None)
        self.assertIn("--keep-video", str(ctx.exception))

    def test_missing_mask_sidecar_says_so(self):
        self._segment()
        (self.out.with_name(self.out.name + "_masks.npz")).unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            stairs_pipeline.analyse(self.out, log=lambda *_: None)
        self.assertIn("Mask sidecar", str(ctx.exception))

    def test_analyse_via_cli_needs_no_source_argument(self):
        # The analyse stage runs where the bags are not, so requiring a source
        # positional would defeat the whole point of the split.
        self._segment()
        code = stairs_pipeline.main(
            stairs_pipeline.parse_args(["--stage", "analyse",
                                        "--out", str(self.out)]))
        self.assertEqual(code, 0)
        self.assertTrue((self.out.with_name(self.out.name + "_overlay.mp4")).exists())

    def test_stage_flags_parse(self):
        a = stairs_pipeline.parse_args(["src", "--stage", "segment"])
        self.assertEqual(a.stage, "segment")
        b = stairs_pipeline.parse_args(["--stage", "analyse", "--out", "o"])
        self.assertEqual((b.stage, b.out), ("analyse", "o"))


class TestCli(unittest.TestCase):
    def test_list_topics_prints_them(self):
        with tempfile.TemporaryDirectory() as td:
            bags = Path(td) / "bags"
            make_split_recording(bags, splits=1, frames_per_split=2,
                                 w=32, h=24)
            args = stairs_pipeline.parse_args([str(bags), "--list"])
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = stairs_pipeline.main(args)
            self.assertEqual(code, 0)
            self.assertIn(COLOR_TOPIC, buf.getvalue())

    def test_defaults(self):
        a = stairs_pipeline.parse_args(["/tmp/x"])
        self.assertEqual(a.prompt, "stairs")
        self.assertEqual(a.out, "output/stairs")


if __name__ == "__main__":
    unittest.main()

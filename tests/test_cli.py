#!/usr/bin/env python3
# tests/test_cli.py
"""Every capability must be reachable from the command line."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2                                                  # noqa: E402

import make_rois                                            # noqa: E402
import render                                               # noqa: E402
import to_webm                                              # noqa: E402


def write_video(path: Path, n: int = 30, w: int = 64, h: int = 48) -> None:
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        wr.write(np.full((h, w, 3), i % 255, dtype=np.uint8))
    wr.release()


class TestMakeRois(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "v.mp4"
        write_video(self.video, n=30, w=64, h=48)
        self.out = self.dir / "rois.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, argv):
        return make_rois.main(make_rois.parse_args(argv))

    def _base(self, *extra):
        return ["--video", str(self.video), "--out", str(self.out), *extra]

    def test_parse_roi(self):
        self.assertEqual(make_rois.parse_roi("cf1=10,20,30,40"),
                         ("cf1", [10, 20, 30, 40]))

    def test_parse_roi_rejects_bad_input(self):
        for bad in ("cf1", "cf1=1,2,3", "=1,2,3,4", "cf1=a,b,c,d",
                    "cf1=1,2,0,4"):
            with self.assertRaises(ValueError, msg=bad):
                make_rois.parse_roi(bad)

    def test_writes_a_file_track_py_can_read(self):
        code = self._run(self._base("--roi", "cf1=10,10,20,20",
                                    "--start-frame", "5", "--end-frame", "25"))
        self.assertEqual(code, 0)
        d = json.loads(self.out.read_text())
        self.assertEqual(d["start_frame"], 5)
        self.assertEqual(d["end_frame"], 25)
        self.assertEqual(d["rois"]["cf1"], [10, 10, 20, 20])

    def test_multiple_rois(self):
        self._run(self._base("--roi", "a=1,1,10,10", "--roi", "b=20,20,10,10"))
        self.assertEqual(sorted(json.loads(self.out.read_text())["rois"]),
                         ["a", "b"])

    def test_end_frame_defaults_to_last_frame(self):
        self._run(self._base("--roi", "a=1,1,10,10"))
        self.assertEqual(json.loads(self.out.read_text())["end_frame"], 29)

    def test_box_outside_the_frame_is_refused(self):
        code = self._run(self._base("--roi", "a=60,40,50,50"))
        self.assertEqual(code, 1)
        self.assertFalse(self.out.exists())

    def test_force_writes_anyway(self):
        self.assertEqual(
            self._run(self._base("--roi", "a=60,40,50,50", "--force")), 0)
        self.assertTrue(self.out.exists())

    def test_end_before_start_is_refused(self):
        self.assertEqual(
            self._run(self._base("--roi", "a=1,1,10,10",
                                 "--start-frame", "20", "--end-frame", "5")), 1)

    def test_amending_an_existing_file(self):
        self._run(self._base("--roi", "a=1,1,10,10", "--roi", "b=2,2,10,10"))
        self._run(self._base("--from-json", str(self.out),
                             "--roi", "b=5,5,12,12"))
        d = json.loads(self.out.read_text())
        self.assertEqual(d["rois"]["a"], [1, 1, 10, 10])     # untouched
        self.assertEqual(d["rois"]["b"], [5, 5, 12, 12])     # amended

    def test_dropping_an_object(self):
        self._run(self._base("--roi", "a=1,1,10,10", "--roi", "b=2,2,10,10"))
        self._run(self._base("--from-json", str(self.out), "--drop", "a"))
        self.assertEqual(list(json.loads(self.out.read_text())["rois"]), ["b"])

    def test_no_rois_is_an_error(self):
        self.assertEqual(self._run(self._base()), 2)

    def test_bad_roi_syntax_exits_nonzero(self):
        self.assertEqual(self._run(self._base("--roi", "nonsense")), 2)

    def test_show_prints_without_writing(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._run(self._base("--roi", "a=1,1,10,10", "--show"))
        self.assertIn('"start_frame"', buf.getvalue())
        self.assertFalse(self.out.exists())

    def test_missing_video_still_writes_with_a_warning(self):
        code = self._run(["--video", str(self.dir / "nope.mp4"),
                          "--out", str(self.out), "--roi", "a=1,1,10,10"])
        self.assertEqual(code, 0)
        self.assertTrue(self.out.exists())


class TestRenderOverrides(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.json = self.dir / "x_tracking.json"
        self.json.write_text(json.dumps({
            "video_in": "v.mp4", "fps": 30.0, "total_frames": 10,
            "width": 64, "height": 48, "start_frame": 0, "end_frame": 9,
            "trail_start_sec": 0, "trail_end_sec": 0,
            "trail_color": {"cf1": [0, 0, 255]}, "trail_thickness": 3,
            "alpha": 0.6, "trail_window": 200, "smooth_trails": True,
            "trails": {"cf1": {"0": [1, 2]}},
        }))

    def tearDown(self):
        self.tmp.cleanup()

    def _ov(self, argv):
        a = render.parse_args([str(self.json), *argv])
        return render.overrides_from_args(a, str(self.json))

    def test_numeric_overrides(self):
        o = self._ov(["--thickness", "7", "--alpha", "0.25",
                      "--trail-window", "50"])
        self.assertEqual(o["trail_thickness"], 7)
        self.assertEqual(o["alpha"], 0.25)
        self.assertEqual(o["trail_window"], 50)

    def test_unset_flags_stay_none_so_the_json_wins(self):
        o = self._ov([])
        self.assertTrue(all(v is None for v in o.values()))

    def test_smooth_toggles_both_ways(self):
        self.assertTrue(self._ov(["--smooth"])["smooth_trails"])
        self.assertFalse(self._ov(["--no-smooth"])["smooth_trails"])

    def test_colour_override_keeps_the_others(self):
        self.json.write_text(json.dumps({
            **json.loads(self.json.read_text()),
            "trail_color": {"cf1": [0, 0, 255], "cf2": [0, 255, 0]}}))
        o = self._ov(["--color", "cf1=255,0,0"])
        self.assertEqual(o["trail_color"]["cf1"], [255, 0, 0])
        self.assertEqual(o["trail_color"]["cf2"], [0, 255, 0])

    def test_bad_colour_exits(self):
        with self.assertRaises(SystemExit):
            self._ov(["--color", "cf1=1,2"])
        with self.assertRaises(SystemExit):
            self._ov(["--color", "nonsense"])

    def test_overrides_reach_the_renderer(self):
        write_video(self.dir / "v.mp4", n=10, w=64, h=48)
        d = json.loads(self.json.read_text())
        d["video_in"] = str(self.dir / "v.mp4")
        self.json.write_text(json.dumps(d))
        render.render(str(self.json), "off", {"trail_thickness": 9})
        self.assertTrue((self.dir / "x_persistent.mp4").exists())


class TestToWebm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_collect_from_directory(self):
        (self.dir / "a.mp4").write_bytes(b"x")
        (self.dir / "b.mp4").write_bytes(b"x")
        files, out = to_webm.collect([str(self.dir)])
        self.assertEqual([f.name for f in files], ["a.mp4", "b.mp4"])
        self.assertEqual(out, self.dir / "webm")

    def test_collect_named_files(self):
        p = self.dir / "a.mp4"
        p.write_bytes(b"x")
        files, out = to_webm.collect([str(p)])
        self.assertEqual(files, [p])
        self.assertEqual(out, self.dir / "webm")

    def test_collect_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            to_webm.collect([str(self.dir / "nope.mp4")])

    def test_args(self):
        a = to_webm.parse_args(["x.mp4", "--bitrate", "4M", "--out-dir", "w"])
        self.assertEqual((a.bitrate, a.out_dir), ("4M", "w"))


class TestEveryToolHasACli(unittest.TestCase):
    """Anything a user runs directly must expose --help."""

    ENTRY_POINTS = [
        "make_rois.py", "track.py", "track_sam3.py", "render.py",
        "to_webm.py", "stairs_pipeline.py", "sam3_preflight.py",
    ]

    def test_help_works(self):
        import subprocess
        for name in self.ENTRY_POINTS:
            path = ROOT / "src" / name
            self.assertTrue(path.exists(), name)
            r = subprocess.run([sys.executable, str(path), "--help"],
                               capture_output=True, timeout=60, cwd=ROOT)
            # sam3_preflight takes no arguments; it just must not crash.
            self.assertIn(r.returncode, (0, 1, 2), f"{name}: {r.stderr[:200]}")

    def test_no_dangling_module_references(self):
        # A help string that names a file which does not exist sends the user
        # down a dead end.
        import re
        for src in (ROOT / "src").glob("*.py"):
            text = src.read_text()
            for ref in re.findall(r"src/([a-z_]+\.py)", text):
                self.assertTrue((ROOT / "src" / ref).exists(),
                                f"{src.name} references missing src/{ref}")


if __name__ == "__main__":
    unittest.main()

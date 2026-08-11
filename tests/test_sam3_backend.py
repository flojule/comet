#!/usr/bin/env python3
# tests/test_sam3_backend.py
"""Tests for the SAM 3 backend plumbing, using a fake predictor.

These cover everything around the model — index mapping, prompt dispatch,
output normalisation, chunking, trail assembly, mask storage.  They cannot
verify SAM 3's tracking quality; that needs a GPU and the real checkpoint.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
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

from maskstore import MaskStore, decode_rle, encode_rle, sidecar_path   # noqa: E402
from sam3_backend import (                                  # noqa: E402
    CHECKPOINT_ENV,
    FrameWindow,
    Observation,
    PromptSpec,
    Sam3Config,
    binarize_mask,
    discover_id_names,
    mask_to_observation,
    normalize_frame_outputs,
    observations_to_trails,
    resolve_checkpoint,
    track_window,
    track_window_chunked,
)
from trails import (                                        # noqa: E402
    build_palette,
    build_tracking_data,
    fill_gaps_bidirectional,
)


BAR_W, BAR_STEP = 6, 2


def make_video(path: Path, n_frames: int, w: int = 160, h: int = 120) -> None:
    """Synthetic clip whose frame index is encoded as a bar's x position.

    Position survives lossy inter-frame coding; a flat grey level does not —
    mp4v shifts a uniform fill by several counts, so encoding the index in
    pixel VALUE would test the codec rather than the frame indexing.
    """
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n_frames):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        x = (i * BAR_STEP) % (w - BAR_W)
        cv2.rectangle(f, (x, 40), (x + BAR_W, 80), (255, 255, 255), -1)
        wr.write(f)
    wr.release()


def frame_index_of(img: np.ndarray) -> int:
    """Recover the index `make_video` encoded into a frame."""
    xs = np.nonzero((img[:, :, 0] > 128).any(axis=0))[0]
    if xs.size == 0:
        return -1
    return round((float(xs.mean()) - BAR_W / 2) / BAR_STEP)


# ── Mask reduction ─────────────────────────────────────────────────────────────

class TestMaskReduction(unittest.TestCase):
    def test_binarize_bool_passthrough(self):
        m = np.zeros((4, 4), dtype=bool)
        m[1, 1] = True
        self.assertTrue(np.array_equal(binarize_mask(m), m))

    def test_binarize_logits_split_at_zero(self):
        raw = np.array([[-4.0, 4.0], [0.5, -0.5]], dtype=np.float32)
        self.assertEqual(binarize_mask(raw).tolist(), [[False, True], [True, False]])

    def test_binarize_probabilities_split_at_half(self):
        # Values inside [0, 1] are probabilities: a 0.4 must NOT be foreground,
        # which a naive >0 threshold would get wrong.
        raw = np.array([[0.1, 0.9], [0.4, 0.6]], dtype=np.float32)
        self.assertEqual(binarize_mask(raw).tolist(), [[False, True], [False, True]])

    def test_binarize_squeezes_leading_axes(self):
        m = np.zeros((1, 1, 3, 3), dtype=bool)
        m[0, 0, 2, 2] = True
        out = binarize_mask(m)
        self.assertEqual(out.shape, (3, 3))
        self.assertTrue(out[2, 2])

    def test_centroid_vs_bbox_point_mode(self):
        # An L-shape: its barycentre and its bbox centre are deliberately apart.
        m = np.zeros((10, 10), dtype=bool)
        m[0:8, 0:2] = True
        m[6:8, 0:8] = True
        c = mask_to_observation(m, 1, 0.9, "centroid")
        b = mask_to_observation(m, 1, 0.9, "bbox")
        self.assertEqual(b.bbox, (0, 0, 8, 8))
        self.assertEqual((b.cx, b.cy), (4, 4))
        self.assertNotEqual((c.cx, c.cy), (b.cx, b.cy))
        self.assertEqual(c.area, int(m.sum()))

    def test_empty_mask_is_no_observation(self):
        self.assertIsNone(
            mask_to_observation(np.zeros((5, 5), bool), 0, 1.0))


# ── Output normalisation ───────────────────────────────────────────────────────

class TestNormalizeOutputs(unittest.TestCase):
    def setUp(self):
        self.cfg = Sam3Config()
        self.shape = (120, 160)

    def _run(self, layout, dtype="bool"):
        fp = FakePredictor(output_layout=layout, mask_dtype=dtype)
        raw = fp._outputs({0: (40.0, 30.0), 1: (80.0, 60.0)}, 0)
        return normalize_frame_outputs(raw, self.shape, self.cfg)

    def test_all_layouts_agree(self):
        ref = self._run("arrays")
        self.assertEqual(sorted(ref), [0, 1])
        for layout in ("mapping", "mapping_dict", "attrs"):
            got = self._run(layout)
            self.assertEqual(sorted(got), [0, 1], layout)
            for oid in ref:
                self.assertEqual((got[oid].cx, got[oid].cy),
                                 (ref[oid].cx, ref[oid].cy), layout)

    def test_all_mask_dtypes_agree(self):
        ref = self._run("arrays", "bool")
        for dtype in ("logits", "probs"):
            got = self._run("arrays", dtype)
            for oid in ref:
                self.assertEqual((got[oid].cx, got[oid].cy),
                                 (ref[oid].cx, ref[oid].cy), dtype)

    def test_masks_at_other_resolution_are_resized(self):
        fp = FakePredictor(mask_scale=0.5)
        raw = fp._outputs({0: (40.0, 30.0)}, 0)
        got = normalize_frame_outputs(raw, self.shape, self.cfg)
        # Half-resolution mask must land back at full-frame coordinates.
        self.assertAlmostEqual(got[0].cx, 40, delta=2)
        self.assertAlmostEqual(got[0].cy, 30, delta=2)

    def test_none_and_empty(self):
        self.assertEqual(normalize_frame_outputs(None, self.shape, self.cfg), {})
        self.assertEqual(
            normalize_frame_outputs(
                {"pred_masks": np.zeros((0, 1, 1)), "obj_ids": [], "scores": []},
                self.shape, self.cfg), {})

    def test_min_area_and_min_score_filters(self):
        fp = FakePredictor(radius=2)
        raw = fp._outputs({0: (40.0, 30.0)}, 0)
        strict = Sam3Config(min_area=10_000)
        self.assertEqual(normalize_frame_outputs(raw, self.shape, strict), {})
        strict = Sam3Config(min_score=0.99)
        self.assertEqual(normalize_frame_outputs(raw, self.shape, strict), {})

    def test_mask_id_count_mismatch_is_loud(self):
        raw = {"pred_masks": np.ones((3, 1, 4, 4), dtype=bool), "obj_ids": [0, 1]}
        with self.assertRaises(ValueError):
            normalize_frame_outputs(raw, (4, 4), self.cfg)

    def test_unrecognised_output_raises(self):
        with self.assertRaises(TypeError):
            normalize_frame_outputs({"nothing": 1}, self.shape, self.cfg)


# ── Frame window ───────────────────────────────────────────────────────────────

class TestFrameWindow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name) / "clip.mp4"
        make_video(self.video, 40)

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_mapping_is_inverse(self):
        w = FrameWindow(str(self.video), 520, 1200)
        self.assertEqual(w.to_absolute(0), 520)
        self.assertEqual(w.to_local(520), 0)
        for i in (0, 1, 137, 680):
            self.assertEqual(w.to_local(w.to_absolute(i)), i)

    def test_extract_writes_only_the_window(self):
        with FrameWindow(str(self.video), 10, 19) as w:
            d = w.extract()
            self.assertEqual(w.n_frames, 10)
            self.assertEqual(len(list(d.glob("*.jpg"))), 10)
            self.assertEqual(w.to_absolute(w.n_frames - 1), 19)
            self.assertEqual((w.width, w.height), (160, 120))

    def test_extract_clamps_past_the_end(self):
        with FrameWindow(str(self.video), 35, 999) as w:
            w.extract()
            self.assertEqual(w.n_frames, 5)

    def test_extracted_frames_are_the_right_ones(self):
        # The load-bearing property of the whole backend: the frame stored under
        # absolute index N really is source frame N.  If extraction is off by
        # even one, every trail is silently offset against the footage.
        with FrameWindow(str(self.video), 12, 20) as w:
            w.extract()
            for abs_idx in (12, 15, 20):
                img = w.read_frame(abs_idx)
                self.assertIsNotNone(img, abs_idx)
                self.assertEqual(frame_index_of(img), abs_idx)

    def test_extraction_matches_a_sequential_read_of_the_source(self):
        # render.py consumes the source sequentially from frame 0, so that read
        # order is the reference the extractor has to agree with.
        cap = cv2.VideoCapture(str(self.video))
        expected = []
        for i in range(25):
            ok, f = cap.read()
            if not ok:
                break
            if i >= 12:
                expected.append(frame_index_of(f))
        cap.release()

        with FrameWindow(str(self.video), 12, 24) as w:
            w.extract()
            got = [frame_index_of(w.read_frame(w.to_absolute(i)))
                   for i in range(w.n_frames)]
        self.assertEqual(got, expected)

    def test_empty_window_rejected(self):
        with FrameWindow(str(self.video), 20, 10) as w:
            with self.assertRaises(ValueError):
                w.extract()

    def test_cleanup_removes_temp_dir(self):
        w = FrameWindow(str(self.video), 0, 4)
        d = w.extract()
        self.assertTrue(d.exists())
        w.cleanup()
        self.assertFalse(d.exists())


# ── Tracking with the fake predictor ───────────────────────────────────────────

class TestTrackWindow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name) / "clip.mp4"
        make_video(self.video, 60)
        self.cfg = Sam3Config()

    def tearDown(self):
        self.tmp.cleanup()

    def _window(self, start=20, end=39):
        w = FrameWindow(str(self.video), start, end)
        w.extract()
        return w

    def test_results_are_keyed_by_absolute_frame(self):
        fp = FakePredictor()
        with self._window(20, 39) as w:
            res = track_window(
                w, self.cfg, predictor=fp,
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                progress=False)
        # The regression this guards: keys must be 20..39, not 0..19.
        self.assertEqual(min(res), 20)
        self.assertEqual(max(res), 39)
        self.assertEqual(len(res), 20)

    def test_trail_follows_the_known_motion(self):
        fp = FakePredictor(velocity=(2.0, 1.0))
        with self._window(20, 39) as w:
            res = track_window(
                w, self.cfg, predictor=fp,
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                progress=False)
        trails = observations_to_trails(res, {0: "cf1"})
        x0, y0 = trails["cf1"][20]
        x5, y5 = trails["cf1"][25]
        self.assertAlmostEqual(x5 - x0, 10, delta=1)   # 2 px/frame × 5
        self.assertAlmostEqual(y5 - y0, 5,  delta=1)

    def test_box_prompt_keyword_is_probed(self):
        # This build only accepts 'bbox'; the backend must find it by probing.
        fp = FakePredictor(box_key="bbox")
        with self._window(20, 29) as w:
            res = track_window(
                w, self.cfg, predictor=fp,
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                progress=False)
        self.assertTrue(any(r["type"] == "add_prompt" and "bbox" in r
                            for r in fp.requests))
        self.assertEqual(len(res), 10)

    def test_no_box_keyword_accepted_raises_clearly(self):
        fp = FakePredictor(box_key="totally_unknown_key")
        with self._window(20, 29) as w:
            with self.assertRaises(RuntimeError) as ctx:
                track_window(w, self.cfg, predictor=fp,
                             prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                             progress=False)
        self.assertIn("--box-key", str(ctx.exception))

    def test_point_prompt_path(self):
        fp = FakePredictor()
        with self._window(20, 29) as w:
            res = track_window(
                w, self.cfg, predictor=fp,
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                prompt_shape="point", progress=False)
        self.assertTrue(any("points" in r for r in fp.requests))
        self.assertEqual(len(res), 10)

    def test_normalized_coords_are_sent_in_unit_range(self):
        fp = FakePredictor()
        cfg = Sam3Config(normalize_coords=True)
        with self._window(20, 29) as w:
            track_window(w, cfg, predictor=fp,
                         prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                         progress=False)
        box = next(np.asarray(r["box"]).reshape(-1)
                   for r in fp.requests if "box" in r)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in box.tolist()), box)

    def test_text_prompt_discovers_objects(self):
        fp = FakePredictor()
        with self._window(20, 39) as w:
            res = track_window(w, self.cfg, predictor=fp, text="drone",
                               progress=False)
        names = discover_id_names(res)
        self.assertEqual(names, {0: "obj0", 1: "obj1"})
        self.assertTrue(any(r.get("text") == "drone" for r in fp.requests))

    def test_discovered_names_can_be_overridden(self):
        fp = FakePredictor()
        with self._window(20, 39) as w:
            res = track_window(w, self.cfg, predictor=fp, text="drone",
                               progress=False)
        self.assertEqual(discover_id_names(res, name_map={1: "cf2"}),
                         {0: "obj0", 1: "cf2"})

    def test_dropouts_leave_holes_not_wrong_points(self):
        fp = FakePredictor(drop_frames={3: {0}, 4: {0}, 5: {0}})
        with self._window(20, 39) as w:
            res = track_window(
                w, self.cfg, predictor=fp,
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                progress=False)
        trails = observations_to_trails(res, {0: "cf1"})
        for missing in (23, 24, 25):
            self.assertNotIn(missing, trails["cf1"])
        self.assertIn(26, trails["cf1"])

    def test_mask_sink_receives_masks_and_frees_them(self):
        # Holding raw masks would be ~2 MB each at 1080p; the sink exists so
        # they are compressed on arrival and dropped from the Observation.
        seen: list[tuple[int, int, tuple]] = []
        cfg = Sam3Config(
            keep_masks=True,
            mask_sink=lambda f, oid, m: seen.append((f, oid, m.shape)),
        )
        with self._window(20, 29) as w:
            res = track_window(
                w, cfg, predictor=FakePredictor(),
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                progress=False)
        self.assertEqual(len(seen), 10)
        self.assertEqual(seen[0][0], 20)                  # absolute frame index
        self.assertEqual(seen[0][2], (120, 160))          # full-frame shape
        for per_obj in res.values():
            for obs in per_obj.values():
                self.assertIsNone(obs.mask)

    def test_masks_are_not_decoded_without_keep_masks(self):
        cfg = Sam3Config(keep_masks=False)
        with self._window(20, 29) as w:
            res = track_window(
                w, cfg, predictor=FakePredictor(),
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                progress=False)
        self.assertTrue(all(o.mask is None
                            for p in res.values() for o in p.values()))

    def test_prompts_and_text_are_mutually_exclusive(self):
        with self._window(20, 29) as w:
            with self.assertRaises(ValueError):
                track_window(w, self.cfg, predictor=FakePredictor(), progress=False)
            with self.assertRaises(ValueError):
                track_window(w, self.cfg, predictor=FakePredictor(), text="a",
                             prompts=[PromptSpec("x", 0, box=(1, 1, 2, 2))],
                             progress=False)

    def test_caller_owned_predictor_is_not_shut_down(self):
        fp = FakePredictor()
        with self._window(20, 29) as w:
            track_window(w, self.cfg, predictor=fp,
                         prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                         progress=False)
        self.assertEqual(fp.shutdown_calls, 0)

    def test_session_is_closed(self):
        fp = FakePredictor()
        with self._window(20, 29) as w:
            track_window(w, self.cfg, predictor=fp,
                         prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))],
                         progress=False)
        self.assertEqual(fp.sessions, {})


class TestChunkedTracking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name) / "clip.mp4"
        make_video(self.video, 80)

    def tearDown(self):
        self.tmp.cleanup()

    def test_chunking_covers_every_frame_and_keeps_ids(self):
        fp = FakePredictor()
        cfg = Sam3Config(chunk_size=10, chunk_overlap=2)
        w = FrameWindow(str(self.video), 20, 69)
        w.extract()
        try:
            res = track_window_chunked(
                w, cfg, predictor=fp,
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10)),
                         PromptSpec("cf2", 1, box=(80, 60, 10, 10))])
        finally:
            w.cleanup()
        self.assertEqual(sorted(res), list(range(20, 70)))
        for f in range(20, 70):
            self.assertEqual(sorted(res[f]), [0, 1], f)
        self.assertGreater(len(fp.sessions), -1)   # all sessions closed
        self.assertEqual(fp.sessions, {})

    def test_overlap_at_least_chunk_size_still_terminates(self):
        # --reprompt-every 5 with the default --chunk-overlap 8 used to make the
        # window start go backwards, looping forever.
        fp = FakePredictor()
        cfg = Sam3Config(chunk_size=5, chunk_overlap=8)
        w = FrameWindow(str(self.video), 20, 39)
        w.extract()
        try:
            res = track_window_chunked(
                w, cfg, predictor=fp,
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))])
        finally:
            w.cleanup()
        self.assertEqual(sorted(res), list(range(20, 40)))

    def test_chunk_size_zero_is_a_single_pass(self):
        fp = FakePredictor()
        cfg = Sam3Config(chunk_size=0)
        w = FrameWindow(str(self.video), 20, 39)
        w.extract()
        try:
            res = track_window_chunked(
                w, cfg, predictor=fp,
                prompts=[PromptSpec("cf1", 0, box=(40, 30, 10, 10))])
        finally:
            w.cleanup()
        starts = [r for r in fp.requests if r["type"] == "start_session"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(sorted(res), list(range(20, 40)))


# ── Trails, palette, gap fill ──────────────────────────────────────────────────

class TestTrailAssembly(unittest.TestCase):
    def test_unknown_ids_are_ignored(self):
        res = {5: {0: Observation(0, 1, 2, (0, 0, 1, 1), 1, 1.0),
                   9: Observation(9, 3, 4, (0, 0, 1, 1), 1, 1.0)}}
        self.assertEqual(observations_to_trails(res, {0: "cf1"}),
                         {"cf1": {5: (1, 2)}})

    def test_named_agent_with_no_observations_still_present(self):
        self.assertEqual(observations_to_trails({}, {0: "cf1"}), {"cf1": {}})

    def test_discover_names_ordered_by_first_appearance(self):
        o = lambda i: Observation(i, 0, 0, (0, 0, 1, 1), 1, 1.0)   # noqa: E731
        res = {10: {7: o(7)}, 11: {7: o(7), 3: o(3)}}
        self.assertEqual(discover_id_names(res), {7: "obj0", 3: "obj1"})

    def test_palette_covers_unknown_names(self):
        # render.py does trail_color[name] unguarded — a miss is a KeyError.
        pal = build_palette(["cf1", "obj0", "obj1"], {"cf1": (0, 0, 255)})
        self.assertEqual(pal["cf1"], (0, 0, 255))
        self.assertEqual(len(pal), 3)
        self.assertNotEqual(pal["obj0"], pal["obj1"])

    def test_gap_fill_interpolates_without_detections(self):
        trails = {"cf1": {0: (0, 0), 10: (100, 50)}}
        filled = fill_gaps_bidirectional(trails, {}, min_gap_frames=5)
        self.assertEqual(sorted(filled["cf1"]), list(range(11)))
        self.assertEqual(filled["cf1"][5], (50, 25))

    def test_gap_fill_leaves_short_gaps_alone(self):
        trails = {"cf1": {0: (0, 0), 3: (30, 0)}}
        filled = fill_gaps_bidirectional(trails, {}, min_gap_frames=5)
        self.assertEqual(sorted(filled["cf1"]), [0, 3])

    def test_gap_fill_handles_empty_trail(self):
        self.assertEqual(fill_gaps_bidirectional({"cf1": {}}, {}), {"cf1": {}})


# ── Tracking JSON contract ─────────────────────────────────────────────────────

class TestTrackingJsonContract(unittest.TestCase):
    REQUIRED = {
        "video_in", "fps", "total_frames", "width", "height", "start_frame",
        "end_frame", "trail_start_sec", "trail_end_sec", "trail_color",
        "trail_thickness", "alpha", "trail_window", "smooth_trails", "trails",
    }

    def _build(self, **kw):
        base = dict(
            video_in="input/x.mp4", fps=59.94, total_frames=1323, width=1920,
            height=1080, start_frame=520, end_frame=1200, trail_start_sec=2,
            trail_end_sec=2, trail_color={"cf1": (0, 0, 255)}, trail_thickness=3,
            alpha=0.6, trail_window=200, smooth_trails=True,
            trails={"cf1": {521: (10, 20), 520: (5, 6)}},
        )
        base.update(kw)
        return build_tracking_data(**base)

    def test_has_every_key_render_reads(self):
        self.assertTrue(self.REQUIRED.issubset(self._build()))

    def test_frame_keys_are_strings_and_sorted(self):
        d = self._build()
        self.assertEqual(list(d["trails"]["cf1"]), ["520", "521"])
        self.assertEqual(d["trails"]["cf1"]["520"], [5, 6])

    def test_matches_the_committed_reference_output(self):
        ref_path = ROOT / "output" / "crazyflo_path_tracking.json"
        if not ref_path.exists():
            self.skipTest("reference tracking JSON not in the checkout")
        ref = json.loads(ref_path.read_text())
        rebuilt = build_tracking_data(
            video_in=ref["video_in"], fps=ref["fps"],
            total_frames=ref["total_frames"], width=ref["width"],
            height=ref["height"], start_frame=ref["start_frame"],
            end_frame=ref["end_frame"], trail_start_sec=ref["trail_start_sec"],
            trail_end_sec=ref["trail_end_sec"],
            trail_color={k: tuple(v) for k, v in ref["trail_color"].items()},
            trail_thickness=ref["trail_thickness"], alpha=ref["alpha"],
            trail_window=ref["trail_window"], smooth_trails=ref["smooth_trails"],
            trails={n: {int(f): tuple(p) for f, p in fd.items()}
                    for n, fd in ref["trails"].items()},
        )
        self.assertEqual(rebuilt, ref)

    def test_extra_fields_do_not_disturb_the_contract(self):
        d = self._build(extra={"tracker": "sam3", "sam3_stats": {}})
        self.assertEqual(d["tracker"], "sam3")
        self.assertTrue(self.REQUIRED.issubset(d))


# ── Mask store ─────────────────────────────────────────────────────────────────

class TestResolveCheckpoint(unittest.TestCase):
    """Weights lookup: explicit path, then $COMET_SAM3_CHECKPOINT, then defaults."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ckpt = Path(self.tmp.name) / "sam3.pt"
        self.ckpt.write_bytes(b"weights")
        self._env = os.environ.pop(CHECKPOINT_ENV, None)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop(CHECKPOINT_ENV, None)
        if self._env is not None:
            os.environ[CHECKPOINT_ENV] = self._env

    def test_explicit_path_wins(self):
        other = Path(self.tmp.name) / "other.pt"
        other.write_bytes(b"w")
        os.environ[CHECKPOINT_ENV] = str(other)
        self.assertEqual(resolve_checkpoint(str(self.ckpt)), str(self.ckpt))

    def test_env_var_used_when_no_explicit_path(self):
        os.environ[CHECKPOINT_ENV] = str(self.ckpt)
        self.assertEqual(resolve_checkpoint(), str(self.ckpt))

    def test_tilde_is_expanded(self):
        os.environ[CHECKPOINT_ENV] = str(self.ckpt).replace(
            str(Path.home()), "~", 1)
        if os.environ[CHECKPOINT_ENV].startswith("~"):
            self.assertEqual(resolve_checkpoint(), str(self.ckpt))

    def test_missing_explicit_path_raises_instead_of_downloading(self):
        # A typo must not silently trigger an 800 MB gated download.
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint(str(Path(self.tmp.name) / "nope.pt"))

    def test_missing_env_path_raises(self):
        os.environ[CHECKPOINT_ENV] = str(Path(self.tmp.name) / "nope.pt")
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint()

    def test_none_when_nothing_found(self):
        with mock.patch("sam3_backend.DEFAULT_CHECKPOINT_PATHS",
                        (str(Path(self.tmp.name) / "absent.pt"),)):
            self.assertIsNone(resolve_checkpoint())

    def test_falls_back_to_default_search_paths(self):
        with mock.patch("sam3_backend.DEFAULT_CHECKPOINT_PATHS",
                        ("/definitely/absent.pt", str(self.ckpt))):
            self.assertEqual(resolve_checkpoint(), str(self.ckpt))


class TestMaskStore(unittest.TestCase):
    def test_rle_roundtrip(self):
        rng = np.random.default_rng(7)
        for shape in ((7, 5), (1, 1), (64, 48)):
            for p in (0.0, 1.0, 0.02, 0.5):
                m = rng.random(shape) < p
                self.assertTrue(
                    np.array_equal(decode_rle(encode_rle(m), shape), m),
                    (shape, p))

    def test_rle_starting_with_foreground(self):
        m = np.ones((3, 3), dtype=bool)
        m[2, 2] = False
        self.assertTrue(np.array_equal(decode_rle(encode_rle(m), (3, 3)), m))

    def test_save_load_roundtrip(self):
        m = np.zeros((40, 60), dtype=bool)
        m[5:15, 10:30] = True
        s = MaskStore(40, 60)
        s.add(100, "cf1", m)
        s.add(101, "cf1", m)
        with tempfile.TemporaryDirectory() as td:
            p = s.save(Path(td) / "x_masks.npz")
            loaded = MaskStore.load(p)
        self.assertTrue(np.array_equal(loaded.get(100, "cf1"), m))
        self.assertIsNone(loaded.get(999, "cf1"))
        self.assertIsNone(loaded.get(100, "nope"))
        self.assertEqual(loaded.names(), ["cf1"])
        self.assertEqual(loaded.frames("cf1"), [100, 101])

    def test_add_is_idempotent_per_frame_and_name(self):
        # Chunk overlap re-emits frames; the index must not gain duplicates.
        m = np.zeros((6, 6), dtype=bool)
        m[1, 1] = True
        s = MaskStore(6, 6)
        s.add(3, "cf1", m)
        s.add(3, "cf1", m)
        self.assertEqual(s.frames("cf1"), [3])
        self.assertEqual(len(s), 1)

    def test_rename_rekeys_data_and_index(self):
        m = np.zeros((6, 6), dtype=bool)
        m[2, 2] = True
        s = MaskStore(6, 6)
        s.add(7, "0", m)
        s.add(8, "1", m)
        s.rename({"0": "cf1", "1": "cf2"})
        self.assertEqual(sorted(s.names()), ["cf1", "cf2"])
        self.assertTrue(np.array_equal(s.get(7, "cf1"), m))
        self.assertIsNone(s.get(7, "0"))

    def test_rename_leaves_unmapped_names_alone(self):
        s = MaskStore(4, 4)
        s.add(1, "keepme", np.ones((4, 4), dtype=bool))
        s.rename({"0": "cf1"})
        self.assertEqual(s.names(), ["keepme"])

    def test_sidecar_path_convention(self):
        self.assertEqual(sidecar_path("output/a_tracking.json").name,
                         "a_masks.npz")


if __name__ == "__main__":
    unittest.main()

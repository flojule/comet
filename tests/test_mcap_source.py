#!/usr/bin/env python3
# tests/test_mcap_source.py
"""Tests for reading camera images out of ROS 2 mcap bags.

Bags are synthesised by tests/make_bag.py, so these exercise the real mcap
reader and CDR decoder end to end — no ROS installation involved.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2                                                  # noqa: E402

from make_bag import (                                      # noqa: E402
    make_split_recording, stair_frame, write_bag,
)
from mcap_source import (                                   # noqa: E402
    TopicInfo,
    extract_to_video,
    find_bags,
    iter_images,
    list_image_topics,
    measure_fps,
)

COLOR_TOPIC = "/camera/color/image_raw"


class TestBagDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_all_splits(self):
        make_split_recording(self.dir, splits=3, frames_per_split=4)
        bags = find_bags(self.dir)
        self.assertEqual(len(bags), 5)          # 3 colour + depth + infra

    def test_single_file_accepted(self):
        make_split_recording(self.dir, splits=1, frames_per_split=2,
                             extra_topics=False)
        f = next(self.dir.glob("*.mcap"))
        self.assertEqual(find_bags(f), [f])

    def test_splits_ordered_by_time_not_filename(self):
        # 12 splits: lexicographic order would put rec_10 before rec_2.
        period = 33_000_000
        for s in range(12):
            write_bag(self.dir / f"rec_{s}.mcap",
                      [stair_frame(32, 24, s)], start_ns=s * period,
                      period_ns=period)
        order = [p.name for p in find_bags(self.dir)]
        self.assertEqual(order[:3], ["rec_0.mcap", "rec_1.mcap", "rec_2.mcap"])
        self.assertEqual(order[-1], "rec_11.mcap")

    def test_missing_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            find_bags(self.dir / "nope")

    def test_folder_without_bags_raises(self):
        with self.assertRaises(FileNotFoundError):
            find_bags(self.dir)


class TestTopicListing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        make_split_recording(self.dir, splits=2, frames_per_split=5,
                             w=64, h=48)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lists_every_image_topic(self):
        topics = {t.topic: t for t in list_image_topics(self.dir)}
        self.assertEqual(sorted(topics), [
            COLOR_TOPIC,
            "/camera/depth/image_rect_raw",
            "/camera/infra1/image_rect_raw",
        ])

    def test_counts_are_summed_across_splits(self):
        topics = {t.topic: t for t in list_image_topics(self.dir)}
        self.assertEqual(topics[COLOR_TOPIC].count, 10)     # 2 splits x 5

    def test_probe_reports_resolution_and_encoding(self):
        topics = {t.topic: t for t in list_image_topics(self.dir, probe=True)}
        c = topics[COLOR_TOPIC]
        self.assertEqual((c.width, c.height), (64, 48))
        self.assertEqual(c.encoding, "rgb8")
        self.assertEqual(topics["/camera/depth/image_rect_raw"].encoding, "mono16")

    def test_probe_can_be_skipped(self):
        topics = {t.topic: t for t in list_image_topics(self.dir, probe=False)}
        self.assertEqual(topics[COLOR_TOPIC].width, 0)

    def test_describe_is_human_readable(self):
        line = TopicInfo("/a", "sensor_msgs/msg/Image", 7, 64, 48, "rgb8").describe()
        self.assertIn("/a", line)
        self.assertIn("64x48", line)
        self.assertIn("Image", line)


class TestFrameDecoding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _one(self, **kw) -> np.ndarray:
        src = stair_frame(64, 48, 0)
        write_bag(self.dir / "b.mcap", [src], 0, 33_000_000, **kw)
        _, frame, _ = next(iter_images(self.dir, kw.get("topic", COLOR_TOPIC)))
        return frame

    def test_rgb8_channel_order_is_corrected(self):
        # The bag stores RGB; a naive read would come back with R and B swapped.
        src = np.zeros((8, 8, 3), dtype=np.uint8)
        src[:, :, 2] = 255                                   # pure red in BGR
        write_bag(self.dir / "b.mcap", [src], 0, 33_000_000, encoding="rgb8")
        _, frame, _ = next(iter_images(self.dir, COLOR_TOPIC))
        self.assertEqual(list(frame[0, 0]), [0, 0, 255])

    def test_bgr8_passthrough(self):
        src = np.zeros((8, 8, 3), dtype=np.uint8)
        src[:, :, 0] = 255                                   # pure blue in BGR
        write_bag(self.dir / "b.mcap", [src], 0, 33_000_000, encoding="bgr8")
        _, frame, _ = next(iter_images(self.dir, COLOR_TOPIC))
        self.assertEqual(list(frame[0, 0]), [255, 0, 0])

    def test_mono8_becomes_three_channel(self):
        f = self._one(encoding="mono8")
        self.assertEqual(f.shape, (48, 64, 3))
        self.assertTrue((f[:, :, 0] == f[:, :, 2]).all())

    def test_mono16_is_scaled_to_8bit(self):
        f = self._one(encoding="mono16")
        self.assertEqual(f.shape, (48, 64, 3))
        self.assertEqual(f.dtype, np.uint8)

    def test_compressed_jpeg(self):
        f = self._one(compressed=True)
        self.assertEqual(f.shape, (48, 64, 3))

    def test_frames_span_all_splits_in_order(self):
        make_split_recording(self.dir, splits=3, frames_per_split=4,
                             extra_topics=False)
        stamps = [ts for ts, _, _ in iter_images(self.dir, COLOR_TOPIC)]
        self.assertEqual(len(stamps), 12)
        self.assertEqual(stamps, sorted(stamps))

    def test_limit_and_stride(self):
        make_split_recording(self.dir, splits=1, frames_per_split=10,
                             extra_topics=False)
        self.assertEqual(len(list(iter_images(self.dir, COLOR_TOPIC, limit=3))), 3)
        every_other = list(iter_images(self.dir, COLOR_TOPIC, stride=2))
        self.assertEqual(len(every_other), 5)

    def test_unknown_topic_yields_nothing(self):
        make_split_recording(self.dir, splits=1, frames_per_split=2,
                             extra_topics=False)
        self.assertEqual(list(iter_images(self.dir, "/nope")), [])


class TestExtractToVideo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        make_split_recording(self.dir, splits=3, frames_per_split=6,
                             w=64, h=48, fps=30.0)
        self.out = self.dir / "out" / "color.mp4"

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_every_frame_across_splits(self):
        meta = extract_to_video(self.dir, COLOR_TOPIC, self.out)
        self.assertEqual(meta["frames"], 18)
        self.assertTrue(self.out.exists())
        cap = cv2.VideoCapture(str(self.out))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertEqual(n, 18)

    def test_metadata_describes_the_stream(self):
        meta = extract_to_video(self.dir, COLOR_TOPIC, self.out)
        self.assertEqual((meta["width"], meta["height"]), (64, 48))
        self.assertEqual(meta["topic"], COLOR_TOPIC)
        self.assertEqual(len(meta["timestamps_ns"]), 18)
        # Every .mcap in the folder is scanned, including the ones carrying
        # only the depth and infra topics — any file could hold the topic.
        self.assertEqual(len(meta["bags"]), 5)
        self.assertAlmostEqual(meta["fps"], 30.0, delta=0.5)

    def test_fps_is_measured_from_message_timestamps(self):
        d2 = Path(self.tmp.name) / "slow"
        make_split_recording(d2, splits=1, frames_per_split=8, w=64, h=48,
                             fps=15.0, extra_topics=False)
        meta = extract_to_video(d2, COLOR_TOPIC, d2 / "v.mp4")
        self.assertAlmostEqual(meta["fps"], 15.0, delta=0.5)

    def test_explicit_fps_overrides_measurement(self):
        meta = extract_to_video(self.dir, COLOR_TOPIC, self.out, fps=10.0)
        self.assertEqual(meta["fps"], 10.0)

    def test_max_frames_caps_output(self):
        meta = extract_to_video(self.dir, COLOR_TOPIC, self.out, max_frames=5)
        self.assertEqual(meta["frames"], 5)

    def test_empty_topic_raises(self):
        with self.assertRaises(ValueError):
            extract_to_video(self.dir, "/not/a/topic", self.out)

    def test_depth_topic_extracts_too(self):
        meta = extract_to_video(self.dir, "/camera/depth/image_rect_raw",
                                self.out)
        self.assertGreater(meta["frames"], 0)


class TestMeasureFps(unittest.TestCase):
    def test_regular_spacing(self):
        stamps = [i * 33_333_333 for i in range(10)]
        self.assertAlmostEqual(measure_fps(stamps), 30.0, delta=0.1)

    def test_median_ignores_a_dropout(self):
        stamps = [0, 33_000_000, 66_000_000, 900_000_000, 933_000_000]
        self.assertAlmostEqual(measure_fps(stamps), 30.3, delta=1.0)

    def test_too_few_samples(self):
        self.assertIsNone(measure_fps([5]))
        self.assertIsNone(measure_fps([]))


if __name__ == "__main__":
    unittest.main()

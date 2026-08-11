#!/usr/bin/env python3
# tests/make_bag.py
"""Synthesise ROS 2 mcap bags that look like a RealSense recording.

Used by the tests, and useful by hand for trying the GUI without real data:

    python tests/make_bag.py /tmp/fakebag --splits 3 --frames 30
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np

IMAGE_MSGDEF = """
std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""

COMPRESSED_MSGDEF = """
std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def stair_frame(w: int, h: int, i: int, angle_deg: float = 0.0) -> np.ndarray:
    """A crude staircase: parallel nosing lines at a known angle, plus texture.

    The known angle is what tests/test_stairs.py checks the orientation
    estimator against.
    """
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    rng = np.random.default_rng(i)
    img = np.clip(img + rng.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)

    cx, cy = w // 2, h // 2
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)      # along a nosing line
    nx, ny = -np.sin(theta), np.cos(theta)     # across the flight
    span = max(w, h)

    for k in range(-4, 5):
        off = k * (h // 12) + (i % 3)          # slight drift, like camera motion
        px, py = cx + nx * off, cy + ny * off
        p0 = (int(px - dx * span), int(py - dy * span))
        p1 = (int(px + dx * span), int(py + dy * span))
        shade = 200 if k % 2 == 0 else 150
        cv2.line(img, p0, p1, (shade, shade, shade), 3)
    return img


def write_bag(
    path: Path,
    frames: list[np.ndarray],
    start_ns: int,
    period_ns: int,
    topic: str = "/camera/color/image_raw",
    encoding: str = "rgb8",
    compressed: bool = False,
) -> None:
    from mcap_ros2.writer import Writer

    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "wb") as fh:
        w = Writer(fh)
        if compressed:
            sch = w.register_msgdef("sensor_msgs/msg/CompressedImage",
                                    COMPRESSED_MSGDEF)
        else:
            sch = w.register_msgdef("sensor_msgs/msg/Image", IMAGE_MSGDEF)

        for n, bgr in enumerate(frames):
            ts = start_ns + n * period_ns
            header = {"stamp": {"sec": ts // 1_000_000_000,
                                "nanosec": ts % 1_000_000_000},
                      "frame_id": "camera_color_optical_frame"}
            if compressed:
                ok, enc = cv2.imencode(".jpg", bgr)
                assert ok
                msg = {"header": header, "format": "jpeg",
                       "data": enc.tobytes()}
            else:
                h, wd = bgr.shape[:2]
                if encoding == "rgb8":
                    payload = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    ch = 3
                elif encoding == "bgr8":
                    payload, ch = bgr, 3
                elif encoding == "mono8":
                    payload, ch = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 1
                elif encoding == "mono16":
                    payload = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype("<u2") * 257
                    ch = 2
                else:
                    raise ValueError(f"make_bag cannot emit {encoding}")
                msg = {"header": header, "height": h, "width": wd,
                       "encoding": encoding, "is_bigendian": 0,
                       "step": wd * ch, "data": payload.tobytes()}
            w.write_message(topic=topic, schema=sch, message=msg,
                            log_time=ts, publish_time=ts)
        w.finish()


def make_split_recording(
    folder: Path,
    splits: int = 3,
    frames_per_split: int = 10,
    w: int = 160,
    h: int = 120,
    fps: float = 30.0,
    angle_deg: float = 20.0,
    extra_topics: bool = True,
) -> Path:
    """A recording split across N files, plus a couple of decoy topics."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    period = int(1e9 / fps)
    n = 0
    for s in range(splits):
        imgs = [stair_frame(w, h, n + k, angle_deg) for k in range(frames_per_split)]
        # Deliberately named so lexicographic order differs from real order
        # once the index passes 9.
        write_bag(folder / f"rec_{s}.mcap", imgs,
                  start_ns=n * period, period_ns=period)
        n += frames_per_split

    if extra_topics:
        imgs = [stair_frame(w, h, k, angle_deg) for k in range(4)]
        write_bag(folder / "rec_depth.mcap", imgs, 0, period,
                  topic="/camera/depth/image_rect_raw", encoding="mono16")
        write_bag(folder / "rec_ir.mcap", imgs, 0, period,
                  topic="/camera/infra1/image_rect_raw", encoding="mono8")
    return folder


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--angle", type=float, default=20.0)
    a = ap.parse_args()
    out = make_split_recording(Path(a.folder), a.splits, a.frames,
                               a.width, a.height, angle_deg=a.angle)
    print(f"Wrote a {a.splits}-file recording to {out}")

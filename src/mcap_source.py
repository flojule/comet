#!/usr/bin/env python3
# mcap_source.py
"""Read camera images out of ROS 2 mcap bags — no ROS installation required.

rosbag2 embeds each message's schema in the mcap file, so `mcap-ros2-support`
can decode CDR messages standalone.  That means this runs anywhere Python does,
including a laptop with no ROS on it.

Handles the two shapes a RealSense driver publishes:
  * sensor_msgs/msg/Image            — raw pixels plus an `encoding` string
  * sensor_msgs/msg/CompressedImage  — JPEG/PNG bytes

A recording split across several files (rosbag2's `_0.mcap`, `_1.mcap`, …) is
read as one continuous stream, ordered by each file's first message timestamp
rather than by filename, since plain lexicographic sorting puts `_10` before
`_9`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SCHEMAS = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}


def _require_mcap():
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError as e:
        raise ImportError(
            f"mcap support is not installed ({e}). "
            "pip install -r requirements-mcap.txt"
        ) from e
    return make_reader, DecoderFactory


# ── Pixel conversion ───────────────────────────────────────────────────────────

# ROS encoding string → the cv2 conversion that lands it in BGR.
_DIRECT_CONVERSIONS = {
    "rgb8":   cv2.COLOR_RGB2BGR,
    "rgba8":  cv2.COLOR_RGBA2BGR,
    "bgra8":  cv2.COLOR_BGRA2BGR,
    "mono8":  cv2.COLOR_GRAY2BGR,
    "8uc1":   cv2.COLOR_GRAY2BGR,
    "yuv422": cv2.COLOR_YUV2BGR_UYVY,
    "uyvy":   cv2.COLOR_YUV2BGR_UYVY,
    "yuv422_yuy2": cv2.COLOR_YUV2BGR_YUYV,
    "yuyv":   cv2.COLOR_YUV2BGR_YUYV,
    "bayer_rggb8": cv2.COLOR_BayerBG2BGR,
    "bayer_bggr8": cv2.COLOR_BayerRG2BGR,
    "bayer_gbrg8": cv2.COLOR_BayerGR2BGR,
    "bayer_grbg8": cv2.COLOR_BayerGB2BGR,
}

# Channel count per encoding, for reshaping the flat byte buffer.
_CHANNELS = {
    "rgb8": 3, "bgr8": 3, "8uc3": 3,
    "rgba8": 4, "bgra8": 4, "8uc4": 4,
    "mono8": 1, "8uc1": 1,
    "mono16": 1, "16uc1": 1,
    "yuv422": 2, "uyvy": 2, "yuv422_yuy2": 2, "yuyv": 2,
    "bayer_rggb8": 1, "bayer_bggr8": 1, "bayer_gbrg8": 1, "bayer_grbg8": 1,
}


def image_msg_to_bgr(msg) -> np.ndarray:
    """Decode a sensor_msgs/msg/Image into a BGR uint8 array."""
    enc = str(msg.encoding).lower()
    h, w = int(msg.height), int(msg.width)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if enc in ("mono16", "16uc1"):
        # 16-bit single channel — depth, or a mono camera in high bit depth.
        # Scaled to 8 bits for display; use the raw stream if you need metres.
        px = buf.view("<u2" if not msg.is_bigendian else ">u2")
        px = px[:h * w].reshape(h, w)
        vis = cv2.normalize(px, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    ch = _CHANNELS.get(enc)
    if ch is None:
        raise ValueError(
            f"Unsupported image encoding {msg.encoding!r}. "
            f"Known: {', '.join(sorted(_CHANNELS))}"
        )

    expected = h * w * ch
    if buf.size < expected:
        raise ValueError(
            f"Truncated image: {buf.size} bytes for {w}x{h}x{ch} "
            f"({expected} expected)"
        )
    img = buf[:expected].reshape(h, w, ch) if ch > 1 else buf[:expected].reshape(h, w)

    if enc in ("bgr8", "8uc3"):
        return img
    if enc in ("8uc4",):
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    conv = _DIRECT_CONVERSIONS.get(enc)
    if conv is None:
        raise ValueError(f"No BGR conversion for encoding {msg.encoding!r}")
    return cv2.cvtColor(img, conv)


def compressed_msg_to_bgr(msg) -> np.ndarray:
    """Decode a sensor_msgs/msg/CompressedImage (JPEG/PNG) into BGR."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"cv2 could not decode a CompressedImage with format={getattr(msg, 'format', '?')!r}"
        )
    return img


def msg_to_bgr(schema_name: str, msg) -> np.ndarray:
    if schema_name == "sensor_msgs/msg/CompressedImage":
        return compressed_msg_to_bgr(msg)
    return image_msg_to_bgr(msg)


# ── Bag discovery ──────────────────────────────────────────────────────────────

def _natural_key(p: Path):
    """Sort key that orders rosbag2 splits _2 before _10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def find_bags(folder_or_file: str | Path) -> list[Path]:
    """All .mcap files for a recording, in playback order.

    Ordered by each file's first message timestamp where the summary provides
    one, falling back to a natural filename sort.
    """
    p = Path(folder_or_file)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"No such file or folder: {p}")

    files = sorted(p.rglob("*.mcap"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No .mcap files under {p}")

    make_reader, DecoderFactory = _require_mcap()
    stamped: list[tuple[int, Path]] = []
    for f in files:
        start = None
        try:
            with open(f, "rb") as fh:
                summary = make_reader(fh).get_summary()
            if summary and summary.statistics:
                start = summary.statistics.message_start_time
        except Exception as e:                  # noqa: BLE001 - ordering only
            logging.debug(f"No summary for {f.name} ({e}); ordering by name")
        stamped.append((start if start is not None else -1, f))

    if all(s >= 0 for s, _ in stamped):
        stamped.sort(key=lambda sp: sp[0])
        return [f for _, f in stamped]
    return files


@dataclass
class TopicInfo:
    topic: str
    schema: str
    count: int
    width: int = 0
    height: int = 0
    encoding: str = ""

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.width else "?"

    def describe(self) -> str:
        return (f"{self.topic}  [{self.schema.split('/')[-1]}]  "
                f"{self.count} msgs  {self.resolution}  {self.encoding}")


def list_image_topics(folder_or_file: str | Path,
                      probe: bool = True) -> list[TopicInfo]:
    """Every image topic in the bag, with message counts and a sampled frame.

    `probe` decodes one message per topic to report resolution and encoding,
    which is what tells you whether a topic is colour, depth or infrared.
    """
    make_reader, DecoderFactory = _require_mcap()
    bags = find_bags(folder_or_file)

    totals: dict[str, TopicInfo] = {}
    for bag in bags:
        with open(bag, "rb") as fh:
            summary = make_reader(fh).get_summary()
            if summary is None:
                continue
            counts = (summary.statistics.channel_message_counts
                      if summary.statistics else {})
            for cid, ch in summary.channels.items():
                schema = summary.schemas.get(ch.schema_id)
                if schema is None or schema.name not in IMAGE_SCHEMAS:
                    continue
                info = totals.get(ch.topic)
                if info is None:
                    totals[ch.topic] = TopicInfo(ch.topic, schema.name,
                                                 int(counts.get(cid, 0)))
                else:
                    info.count += int(counts.get(cid, 0))

    if probe:
        for topic, info in totals.items():
            try:
                for _, frame, msg in iter_images(bags, topic, limit=1):
                    info.height, info.width = frame.shape[:2]
                    info.encoding = str(getattr(msg, "encoding", None)
                                        or getattr(msg, "format", "")).strip()
            except Exception as e:              # noqa: BLE001 - probe only
                info.encoding = f"<undecodable: {e}>"

    return sorted(totals.values(), key=lambda i: i.topic)


# ── Frame iteration ────────────────────────────────────────────────────────────

def iter_images(bags: list[Path] | str | Path, topic: str,
                limit: int | None = None, stride: int = 1):
    """Yield (timestamp_ns, bgr_frame, raw_msg) for `topic`, across all bags."""
    make_reader, DecoderFactory = _require_mcap()
    if not isinstance(bags, list):
        bags = find_bags(bags)

    emitted = 0
    seen = 0
    for bag in bags:
        with open(bag, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            for schema, channel, message, msg in reader.iter_decoded_messages(
                    topics=[topic]):
                if seen % stride:
                    seen += 1
                    continue
                seen += 1
                yield message.log_time, msg_to_bgr(schema.name, msg), msg
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def extract_to_video(
    folder_or_file: str | Path,
    topic: str,
    out_path: str | Path,
    *,
    fps: float | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    progress=None,
) -> dict:
    """Write a topic's images to an mp4 and return metadata.

    The rest of the pipeline consumes plain video files, so turning a bag into
    one mp4 keeps the mcap dependency isolated to this module.  Frame timestamps
    are preserved in the returned dict (and by the caller in a sidecar), since
    the mp4 itself only carries a constant frame rate.
    """
    bags = find_bags(folder_or_file)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    stamps: list[int] = []
    size = None

    for ts, frame, _msg in iter_images(bags, topic, stride=stride):
        if writer is None:
            size = (frame.shape[1], frame.shape[0])
            # Real fps comes from the message timestamps; until we have two we
            # cannot know it, so open the writer with a placeholder and rewrite
            # the container at the end if it differs materially.
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), fps or 30.0, size)
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open {out_path} for writing")
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        stamps.append(int(ts))
        if progress and len(stamps) % 25 == 0:
            progress(len(stamps))
        if max_frames is not None and len(stamps) >= max_frames:
            break

    if writer is None:
        raise ValueError(f"No decodable images on topic {topic!r}")
    writer.release()

    measured = measure_fps(stamps)
    if fps is None and measured and abs(measured - 30.0) > 0.5:
        # Rewrite the container so playback speed matches the recording.
        _rewrite_fps(out_path, measured, size)

    return {
        "video": str(out_path),
        "topic": topic,
        "frames": len(stamps),
        "width": size[0],
        "height": size[1],
        "fps": fps or measured or 30.0,
        "timestamps_ns": stamps,
        "bags": [str(b) for b in bags],
    }


def measure_fps(stamps_ns: list[int]) -> float | None:
    """Median frame rate implied by message timestamps."""
    if len(stamps_ns) < 2:
        return None
    deltas = np.diff(np.asarray(stamps_ns, dtype=np.float64))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None
    return float(1e9 / np.median(deltas))


def _rewrite_fps(path: Path, fps: float, size: tuple[int, int]) -> None:
    """Re-encode `path` at `fps`.  Frame data is unchanged; only the rate is."""
    tmp = path.with_suffix(".retimed.mp4")
    cap = cv2.VideoCapture(str(path))
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        cap.release()
        logging.warning(f"Could not retime {path} to {fps:.2f} fps; leaving as-is")
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    tmp.replace(path)
    logging.info(f"Video timed at {fps:.2f} fps from message timestamps")

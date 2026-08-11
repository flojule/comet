#!/usr/bin/env python3
# media.py
"""Normalise any visual input into a video file the rest of the pipeline reads.

Accepts a video, a single photo, a folder of photos, or a folder of mcap bags,
and always hands back one mp4.  Everything downstream (FrameWindow, SAM 3,
render) then has exactly one input shape to care about.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Media:
    """A resolved video plus how it was produced."""
    video: Path
    kind: str                       # video | image | images | mcap
    frames: int
    width: int
    height: int
    fps: float
    detail: dict
    _tmp: str | None = None

    def cleanup(self) -> None:
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def __enter__(self) -> Media:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


def classify(path: str | Path) -> str:
    """What kind of input this is: video | image | images | mcap."""
    p = Path(path)
    if p.is_file():
        s = p.suffix.lower()
        if s in VIDEO_SUFFIXES:
            return "video"
        if s in IMAGE_SUFFIXES:
            return "image"
        if s == ".mcap":
            return "mcap"
        raise ValueError(f"Unrecognised file type: {p.name}")
    if not p.is_dir():
        raise FileNotFoundError(f"No such file or folder: {p}")
    if any(p.rglob("*.mcap")):
        return "mcap"
    if any(q.suffix.lower() in IMAGE_SUFFIXES for q in p.iterdir() if q.is_file()):
        return "images"
    raise ValueError(f"{p} holds no images and no .mcap files")


def _images_in(folder: Path) -> list[Path]:
    return sorted(
        (q for q in folder.iterdir()
         if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda q: q.name,
    )


def _write_video(frames, out: Path, fps: float) -> tuple[int, int, int]:
    writer, size, n = None, None, 0
    for img in frames:
        if writer is None:
            size = (img.shape[1], img.shape[0])
            writer = cv2.VideoWriter(
                str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open {out} for writing")
        if (img.shape[1], img.shape[0]) != size:
            # A folder of mixed-size photos would otherwise produce a video
            # where every off-size frame is silently dropped by the writer.
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        writer.write(img)
        n += 1
    if writer is None:
        raise ValueError("No frames to write")
    writer.release()
    return n, size[0], size[1]


def resolve(
    source: str | Path,
    *,
    topic: str | None = None,
    fps: float = 30.0,
    still_frames: int = 8,
    max_frames: int | None = None,
    workdir: str | Path | None = None,
    progress=None,
) -> Media:
    """Turn `source` into a Media with a real video file behind it.

    `topic` is required for mcap input.  `still_frames` controls how many times
    a single photo is repeated: SAM 3's video path wants a sequence, and a
    handful of identical frames also gives its memory bank something stable to
    settle on.
    """
    src = Path(source)
    kind = classify(src)

    if kind == "video":
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        v = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.release()
        return Media(src, "video", n, w, h, v, {"source": str(src)})

    tmp = None
    if workdir is None:
        tmp = tempfile.mkdtemp(prefix="comet_media_")
        out_dir = Path(tmp)
    else:
        out_dir = Path(workdir)
        out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "input.mp4"

    try:
        if kind == "mcap":
            if not topic:
                raise ValueError(
                    "mcap input needs a topic — run `python src/stairs_pipeline.py "
                    f"--list {src}` to see what is in the bag"
                )
            from mcap_source import extract_to_video
            meta = extract_to_video(src, topic, out, max_frames=max_frames,
                                    progress=progress)
            return Media(out, "mcap", meta["frames"], meta["width"],
                         meta["height"], meta["fps"], meta, _tmp=tmp)

        if kind == "image":
            img = cv2.imread(str(src))
            if img is None:
                raise ValueError(f"Cannot read image: {src}")
            n, w, h = _write_video([img] * max(1, still_frames), out, fps)
            logging.info(f"Single image → {n}-frame clip ({w}x{h})")
            return Media(out, "image", n, w, h, fps,
                         {"source": str(src), "repeated": n}, _tmp=tmp)

        files = _images_in(src)
        if not files:
            raise ValueError(f"No images in {src}")
        if max_frames:
            files = files[:max_frames]

        def _read():
            for i, f in enumerate(files):
                img = cv2.imread(str(f))
                if img is None:
                    logging.warning(f"Skipping unreadable image {f.name}")
                    continue
                if progress and i % 25 == 0:
                    progress(i)
                yield img

        n, w, h = _write_video(_read(), out, fps)
        logging.info(f"{n} images → clip ({w}x{h}) at {fps} fps")
        return Media(out, "images", n, w, h, fps,
                     {"source": str(src), "files": [f.name for f in files]},
                     _tmp=tmp)
    except Exception:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        raise

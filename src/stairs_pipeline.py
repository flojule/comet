#!/usr/bin/env python3
# stairs_pipeline.py
"""Segment a named thing in a recording and measure its orientation.

    mcap / video / photos ──▶ SAM 3 (text prompt) ──▶ masks
                                                      ├──▶ overlay video
                                                      └──▶ orientation JSON

Built for static structure — stairs, doorways, ramps — where the camera moves
and the subject does not.  That is why it renders a per-frame overlay instead
of the motion trails track.py and render.py produce: a trail drawn through a
static object encodes the camera's egomotion, not the object's, which is
misleading rather than useful.

    python src/stairs_pipeline.py --list /path/to/bags
    python src/stairs_pipeline.py /path/to/bags \\
        --topic /camera/color/image_raw --prompt stairs --out output/stairs

Everything except the SAM 3 call itself runs without a GPU, so the mcap
reading, overlay and orientation stages are testable on any machine.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import stair_orientation as so
from maskstore import MaskStore
from media import Media, resolve
from sam3_backend import (
    FrameWindow,
    Sam3Config,
    build_predictor,
    discover_id_names,
    track_window_chunked,
)

OVERLAY_ALPHA = 0.45
MASK_COLORS = [
    (  0, 220, 255),   # amber
    (255, 120,   0),   # azure
    (  0, 255, 120),   # green
    (255,   0, 200),   # magenta
]


@dataclass
class StairsResult:
    video: str
    overlay: str | None
    data: str
    frames: int
    objects: dict[int, str]
    coverage: dict[str, float] = field(default_factory=dict)
    orientation: dict = field(default_factory=dict)


def _color(i: int) -> tuple[int, int, int]:
    return MASK_COLORS[i % len(MASK_COLORS)]


def render_overlay(
    media: Media,
    window: FrameWindow,
    masks: MaskStore,
    id_to_name: dict[int, str],
    orientations: list[so.Orientation] | None,
    out_path: str | Path,
    *,
    draw_orientation: bool = True,
) -> str | None:
    """Tint each object's mask over the frame, and annotate the angle."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), media.fps,
        (window.width, window.height))
    if not writer.isOpened():
        logging.error(f"Cannot open overlay video for writing: {out_path}")
        return None

    by_frame = {o.frame: o for o in (orientations or [])}
    names = list(id_to_name.values())

    for local in range(window.n_frames):
        abs_idx = window.to_absolute(local)
        frame = window.read_frame(abs_idx)
        if frame is None:
            continue

        for i, name in enumerate(names):
            m = masks.get(abs_idx, name)
            if m is None or m.shape != frame.shape[:2]:
                continue
            col = np.array(_color(i), dtype=np.float32)
            frame[m] = (frame[m] * (1 - OVERLAY_ALPHA)
                        + col * OVERLAY_ALPHA).astype(np.uint8)
            # Outline makes the mask boundary legible where the tint is subtle.
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cont, -1, _color(i), 2)

        o = by_frame.get(abs_idx)
        if draw_orientation and o is not None:
            frame = so.annotate(frame, o)

        present = [n for i, n in enumerate(names)
                   if masks.get(abs_idx, n) is not None]
        info = f"frame {abs_idx}  {', '.join(present) if present else 'NOTHING DETECTED'}"
        cv2.putText(frame, info, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4)
        cv2.putText(frame, info, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (230, 230, 230), 1)
        writer.write(frame)

    writer.release()
    return str(out_path)


def run(
    source: str | Path,
    *,
    prompt: str = "stairs",
    topic: str | None = None,
    out_stem: str | Path = "output/stairs",
    max_frames: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    checkpoint: str | None = None,
    min_score: float = 0.0,
    reprompt_every: int = 0,
    orientation: bool = True,
    overlay: bool = True,
    orientation_cfg: so.OrientationConfig | None = None,
    predictor=None,
    progress=None,
    log=print,
) -> StairsResult:
    """Full run.  `progress`/`log` are hooks so a GUI can follow along."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    log(f"Resolving input: {source}")
    media = resolve(source, topic=topic, max_frames=max_frames,
                    progress=progress)
    window: FrameWindow | None = None
    try:
        log(f"  {media.kind}: {media.frames} frames "
            f"{media.width}x{media.height} @ {media.fps:.1f} fps")

        last = end_frame if end_frame is not None else media.frames - 1
        window = FrameWindow(str(media.video), start_frame, last)
        window.extract()
        log(f"  extracted {window.n_frames} frames")

        cfg = Sam3Config(
            keep_masks=True,
            min_score=min_score,
            chunk_size=max(0, reprompt_every),
            checkpoint=checkpoint,
        )
        masks = MaskStore(window.height, window.width)
        cfg.mask_sink = lambda f, oid, m: masks.add(f, str(oid), m)

        own = predictor is None
        predictor = predictor or build_predictor(cfg)
        log(f"Running SAM 3 with prompt {prompt!r} …")
        try:
            results = track_window_chunked(window, cfg, predictor=predictor,
                                           text=prompt)
        finally:
            if own and hasattr(predictor, "shutdown"):
                try:
                    predictor.shutdown()
                except Exception as e:      # noqa: BLE001 - cleanup only
                    logging.warning(f"predictor.shutdown() failed: {e}")

        id_to_name = discover_id_names(results, prefix=_slug(prompt))
        if not id_to_name:
            log(f"SAM 3 found nothing matching {prompt!r}. "
                f"Try a different wording, or check the topic is the colour "
                f"camera rather than depth/infra.")
        else:
            log(f"  found {len(id_to_name)}: " +
                ", ".join(sorted(id_to_name.values())))
        masks.rename({str(o): n for o, n in id_to_name.items()})

        span = sorted(results)
        coverage = {
            name: round(sum(1 for f in span if oid in results[f]) / max(1, len(span)), 4)
            for oid, name in id_to_name.items()
        }
        for name, cov in coverage.items():
            (log if cov >= 0.9 else
             (lambda m: log("WARNING: " + m)))(f"  {name}: seen in {100*cov:.0f}% of frames")

        # ── Orientation ───────────────────────────────────────────────────────
        ori: list[so.Orientation] = []
        ori_summary: dict = {}
        if orientation and id_to_name:
            cfgo = orientation_cfg or so.OrientationConfig()
            primary = _largest_object(results, id_to_name)
            log(f"Measuring orientation of {primary!r} …")
            for local in range(window.n_frames):
                abs_idx = window.to_absolute(local)
                frame = window.read_frame(abs_idx)
                if frame is None:
                    continue
                ori.append(so.estimate_frame(
                    frame, masks.get(abs_idx, primary), abs_idx, cfgo))
            ori = so.smooth_series(ori, cfgo.smooth_window)
            ori_summary = so.summarise(ori)
            ori_summary["object"] = primary
            if ori_summary.get("angle_deg_mean") is None:
                log("  no dominant edge direction found in any frame")
            else:
                log(f"  mean angle {ori_summary['angle_deg_mean']:.1f}deg "
                    f"(sd {ori_summary['angle_deg_std']:.1f}, "
                    f"{100*ori_summary['coverage']:.0f}% of frames)")

        # ── Outputs ───────────────────────────────────────────────────────────
        overlay_path = None
        if overlay and id_to_name:
            overlay_path = render_overlay(
                media, window, masks, id_to_name, ori,
                out_stem.with_name(out_stem.name + "_overlay.mp4"))
            if overlay_path:
                log(f"Overlay → {overlay_path}")

        mask_path = out_stem.with_name(out_stem.name + "_masks.npz")
        masks.save(mask_path)

        data = {
            "source": str(source),
            "topic": topic,
            "prompt": prompt,
            "media": {k: v for k, v in media.detail.items()
                      if k != "timestamps_ns"},
            "fps": media.fps,
            "width": window.width,
            "height": window.height,
            "start_frame": start_frame,
            "end_frame": last,
            "frames": window.n_frames,
            "objects": {str(o): n for o, n in id_to_name.items()},
            "coverage": coverage,
            "orientation": ori_summary,
            "masks": str(mask_path),
        }
        if media.kind == "mcap":
            data["timestamps_ns"] = media.detail.get("timestamps_ns", [])

        data_path = out_stem.with_name(out_stem.name + "_stairs.json")
        data_path.write_text(json.dumps(data, indent=2))
        log(f"Data → {data_path}")
        log(f"Masks → {mask_path}")

        return StairsResult(
            video=str(media.video), overlay=overlay_path, data=str(data_path),
            frames=window.n_frames, objects=id_to_name, coverage=coverage,
            orientation=ori_summary,
        )
    finally:
        if window is not None:
            window.cleanup()
        media.cleanup()


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in text.strip().lower())
    return keep.strip("_") or "obj"


def _largest_object(results, id_to_name: dict[int, str]) -> str:
    """The object with the most mask area — the staircase, not a stray step."""
    totals: dict[int, float] = {}
    for per_obj in results.values():
        for oid, obs in per_obj.items():
            totals[oid] = totals.get(oid, 0.0) + obs.area
    if not totals:
        return next(iter(id_to_name.values()))
    return id_to_name[max(totals, key=totals.get)]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?",
                   help="mcap folder/file, video, photo, or folder of photos")
    p.add_argument("--list", action="store_true",
                   help="list image topics in an mcap source and exit")
    p.add_argument("--topic", default=None, help="camera topic (mcap input)")
    p.add_argument("--prompt", default="stairs", help="what to segment")
    p.add_argument("--out", default="output/stairs", help="output path stem")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--reprompt-every", type=int, default=0)
    p.add_argument("--no-orientation", action="store_true")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    if not args.source:
        print("Give a source. Try --help.")
        return 2

    if args.list:
        from mcap_source import list_image_topics
        topics = list_image_topics(args.source)
        if not topics:
            print(f"No image topics found in {args.source}")
            return 1
        print(f"Image topics in {args.source}:\n")
        for t in topics:
            print("  " + t.describe())
        return 0

    run(args.source, prompt=args.prompt, topic=args.topic, out_stem=args.out,
        max_frames=args.max_frames, start_frame=args.start_frame,
        end_frame=args.end_frame, checkpoint=args.checkpoint,
        min_score=args.min_score, reprompt_every=args.reprompt_every,
        orientation=not args.no_orientation, overlay=not args.no_overlay)
    return 0


if __name__ == "__main__":
    sys.exit(main())

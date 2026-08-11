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
from media import resolve
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


def _render_overlay_frames(
    frames: list,
    start_frame: int,
    fps: float,
    masks: MaskStore,
    id_to_name: dict[int, str],
    orientations: list[so.Orientation] | None,
    out_path: str | Path,
    *,
    draw_orientation: bool = True,
) -> str | None:
    """Tint each object's mask over the frame, and annotate the angle.

    Takes decoded frames rather than a live FrameWindow, so this runs from the
    stage-1 artifacts alone — no bag, no GPU, no re-extraction.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        logging.error("No frames to render")
        return None
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        logging.error(f"Cannot open overlay video for writing: {out_path}")
        return None

    by_frame = {o.frame: o for o in (orientations or [])}
    names = list(id_to_name.values())

    for local, frame in enumerate(frames):
        abs_idx = start_frame + local
        frame = frame.copy()

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


def segment(
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
    keep_video: bool = True,
    predictor=None,
    progress=None,
    log=print,
) -> dict:
    """Stage 1 — run SAM 3 and persist masks.  THIS IS THE STAGE THAT NEEDS A GPU.

    Writes `<stem>_masks.npz`, `<stem>_frames.mp4` and `<stem>_stairs.json`.
    Those three files are a complete handoff: `analyse()` needs nothing else, so
    the model can run on the GPU machine and everything downstream elsewhere.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    log(f"Resolving input: {source}")
    media = resolve(source, topic=topic, max_frames=max_frames, progress=progress)
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

        # Area per object, so analyse() can pick the primary one without the
        # per-frame observations, which are not part of the handoff.
        areas: dict[str, float] = {}
        for per_obj in results.values():
            for oid, obs in per_obj.items():
                n = id_to_name.get(oid)
                if n:
                    areas[n] = areas.get(n, 0.0) + obs.area

        mask_path = out_stem.with_name(out_stem.name + "_masks.npz")
        masks.save(mask_path)

        frames_path = None
        if keep_video:
            frames_path = out_stem.with_name(out_stem.name + "_frames.mp4")
            _persist_window(window, media.fps, frames_path)
            log(f"Frames → {frames_path}")

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
            "areas": areas,
            "orientation": {},
            "masks": str(mask_path),
            "frames_video": str(frames_path) if frames_path else None,
        }
        if media.kind == "mcap":
            data["timestamps_ns"] = media.detail.get("timestamps_ns", [])

        data_path = out_stem.with_name(out_stem.name + "_stairs.json")
        data_path.write_text(json.dumps(data, indent=2))
        log(f"Data → {data_path}")
        log(f"Masks → {mask_path}")
        return data
    finally:
        if window is not None:
            window.cleanup()
        media.cleanup()


def _persist_window(window: FrameWindow, fps: float, out: Path) -> None:
    """Copy the extracted frames into an mp4 that outlives the temp dir."""
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps,
                             (window.width, window.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open {out} for writing")
    for local in range(window.n_frames):
        frame = window.read_frame(window.to_absolute(local))
        if frame is not None:
            writer.write(frame)
    writer.release()


def analyse(
    out_stem: str | Path = "output/stairs",
    *,
    orientation: bool = True,
    overlay: bool = True,
    orientation_cfg: so.OrientationConfig | None = None,
    frames_video: str | Path | None = None,
    log=print,
) -> StairsResult:
    """Stage 2 — orientation and overlay from stage 1's output.  NO GPU NEEDED.

    Re-runnable: retuning orientation costs seconds here rather than a full
    pass of the model.
    """
    out_stem = Path(out_stem)
    data_path = out_stem.with_name(out_stem.name + "_stairs.json")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found — run the segment stage first "
            f"(python src/stairs_pipeline.py <source> --stage segment)"
        )
    data = json.loads(data_path.read_text())

    mask_path = Path(data.get("masks") or
                     out_stem.with_name(out_stem.name + "_masks.npz"))
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask sidecar missing: {mask_path}")
    masks = MaskStore.load(mask_path)

    # Guard the unset case separately: Path("") is Path("."), which exists, so
    # a bare exists() check would sail past a missing record and fail later
    # with an unhelpful "cannot open ." from VideoCapture.
    recorded = frames_video or data.get("frames_video")
    if not recorded:
        raise FileNotFoundError(
            "No frames video recorded by the segment stage — re-run it with "
            "--keep-video (the default), or point --frames-video at the clip."
        )
    video = Path(recorded)
    if not video.exists():
        raise FileNotFoundError(
            f"Frames video missing: {video}. Copy it across from the machine "
            f"that ran the segment stage, or pass --frames-video."
        )

    id_to_name = {int(k): v for k, v in data["objects"].items()}
    start = int(data["start_frame"])
    n_frames = int(data["frames"])
    fps = float(data["fps"])

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    frames: list = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < n_frames:
        log(f"WARNING: {video.name} has {len(frames)} frames, expected {n_frames}")
        n_frames = len(frames)

    ori: list[so.Orientation] = []
    ori_summary: dict = {}
    if orientation and id_to_name:
        cfgo = orientation_cfg or so.OrientationConfig()
        primary = _primary_from_areas(data.get("areas") or {}, id_to_name)
        log(f"Measuring orientation of {primary!r} …")
        for local in range(n_frames):
            abs_idx = start + local
            ori.append(so.estimate_frame(
                frames[local], masks.get(abs_idx, primary), abs_idx, cfgo))
        ori = so.smooth_series(ori, cfgo.smooth_window)
        ori_summary = so.summarise(ori)
        ori_summary["object"] = primary
        if ori_summary.get("angle_deg_mean") is None:
            log("  no dominant edge direction found in any frame")
        else:
            log(f"  mean angle {ori_summary['angle_deg_mean']:.1f}deg "
                f"(sd {ori_summary['angle_deg_std']:.1f}, "
                f"{100*ori_summary['coverage']:.0f}% of frames)")

    overlay_path = None
    if overlay and id_to_name:
        overlay_path = _render_overlay_frames(
            frames[:n_frames], start, fps, masks, id_to_name, ori,
            out_stem.with_name(out_stem.name + "_overlay.mp4"))
        if overlay_path:
            log(f"Overlay → {overlay_path}")

    data["orientation"] = ori_summary
    data_path.write_text(json.dumps(data, indent=2))

    return StairsResult(
        video=str(video), overlay=overlay_path, data=str(data_path),
        frames=n_frames, objects=id_to_name,
        coverage=data.get("coverage", {}), orientation=ori_summary,
    )


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
    keep_video: bool = True,
    predictor=None,
    progress=None,
    log=print,
) -> StairsResult:
    """Both stages back to back, for when one machine does everything."""
    data = segment(
        source, prompt=prompt, topic=topic, out_stem=out_stem,
        max_frames=max_frames, start_frame=start_frame, end_frame=end_frame,
        checkpoint=checkpoint, min_score=min_score,
        reprompt_every=reprompt_every, keep_video=keep_video,
        predictor=predictor, progress=progress, log=log,
    )
    if not data["objects"]:
        # Nothing was found; there is nothing for stage 2 to measure or draw.
        return StairsResult(
            video=data.get("frames_video") or "", overlay=None,
            data=str(Path(out_stem).with_name(Path(out_stem).name + "_stairs.json")),
            frames=int(data["frames"]), objects={},
            coverage=data.get("coverage", {}), orientation={},
        )
    return analyse(out_stem, orientation=orientation, overlay=overlay,
                   orientation_cfg=orientation_cfg, log=log)


def _primary_from_areas(areas: dict, id_to_name: dict[int, str]) -> str:
    """Biggest object by accumulated mask area — the staircase, not a stray step."""
    if areas:
        return max(areas, key=areas.get)
    return next(iter(id_to_name.values()))


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in text.strip().lower())
    return keep.strip("_") or "obj"


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
    p.add_argument("--stage", choices=("all", "segment", "analyse"), default="all",
                   help="'segment' runs SAM 3 and needs a CUDA GPU; 'analyse' "
                        "does orientation and overlay from its output and needs "
                        "none, so the two can run on different machines "
                        "(default: all)")
    p.add_argument("--frames-video", default=None,
                   help="override the frames mp4 the analyse stage reads")
    p.add_argument("--no-keep-video", action="store_true",
                   help="do not persist the frames mp4 (analyse then cannot run "
                        "separately)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    # The analyse stage reads stage 1's artifacts, so it takes no source at all.
    if args.stage == "analyse":
        analyse(args.out, orientation=not args.no_orientation,
                overlay=not args.no_overlay, frames_video=args.frames_video)
        return 0

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

    if args.stage == "segment":
        segment(args.source, prompt=args.prompt, topic=args.topic,
                out_stem=args.out, max_frames=args.max_frames,
                start_frame=args.start_frame, end_frame=args.end_frame,
                checkpoint=args.checkpoint, min_score=args.min_score,
                reprompt_every=args.reprompt_every,
                keep_video=not args.no_keep_video)
        print("\nSegment stage done. Copy these to wherever you want to analyse:")
        stem = Path(args.out)
        for suffix in ("_stairs.json", "_masks.npz", "_frames.mp4"):
            p = stem.with_name(stem.name + suffix)
            if p.exists():
                print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")
        print(f"\nThen: python src/stairs_pipeline.py --stage analyse --out {args.out}")
        return 0

    run(args.source, prompt=args.prompt, topic=args.topic, out_stem=args.out,
        max_frames=args.max_frames, start_frame=args.start_frame,
        end_frame=args.end_frame, checkpoint=args.checkpoint,
        min_score=args.min_score, reprompt_every=args.reprompt_every,
        orientation=not args.no_orientation, overlay=not args.no_overlay,
        keep_video=not args.no_keep_video)
    return 0


if __name__ == "__main__":
    sys.exit(main())

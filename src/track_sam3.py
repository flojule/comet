#!/usr/bin/env python3
# track_sam3.py
"""Track objects with SAM 3, emitting Comet's standard tracking JSON.

A drop-in alternative to track.py.  Same output contract, so render.py and
to_webm.py are unchanged and the two trackers can be compared on one clip:

    python src/track_sam3.py --out output/crazyflo_sam3.mp4
    python src/render.py     output/crazyflo_sam3_tracking.json

Two ways to say what to track:

    --from-rois          (default) seed one object per box in src/rois.json.
                         Keeps your cf1/cf2/cf3/payload identities.
    --text "drone"       let the SAM 3 detector find every matching object.
                         Names are assigned by first appearance: obj0, obj1, …
                         Rename with --name 0=cf1 --name 1=cf2.

Needs a CUDA GPU and the SAM 3 package — run `python src/sam3_preflight.py`
first.  Everything except the model call is importable without torch.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from maskstore import MaskStore, sidecar_path
from sam3_backend import (
    FrameWindow,
    Observation,
    PromptSpec,
    Sam3Config,
    build_predictor,
    discover_id_names,
    observations_to_trails,
    track_object_zoom,
    track_window_chunked,
)
from trails import (
    build_palette,
    build_tracking_data,
    fill_gaps_bidirectional,
    write_tracking_json,
)

# ── Configuration ──────────────────────────────────────────────────────────────
VIDEO_IN  = "input/crazyflo.mp4"
ROIS_FILE = "src/rois.json"
VIDEO_OUT = "output/crazyflo_sam3.mp4"

# Reused from track.py so both trackers render in the same colours.
TRAIL_COLOR: dict[str, tuple[int, int, int]] = {
    "cf1":     (  0,   0, 255),   # red
    "cf2":     (  0, 255,   0),   # green
    "cf3":     (255,   0,   0),   # blue
    "payload": ( 50,  50,  50),
}
TRAIL_THICKNESS = 3
DOT_RADIUS      = 5
ALPHA           = 0.6
TRAIL_START_SEC = 2
TRAIL_END_SEC   = 2
TRAIL_WINDOW    = 200
SMOOTH_TRAILS   = True

LOG_TO_FILE = True
LOG_FILE    = "track_sam3.log"


# ── ROI loading ────────────────────────────────────────────────────────────────

def load_rois(path: str, required: bool = True
              ) -> tuple[int, int | None, dict[str, list[int]]]:
    """Read rois.json.  In text mode the file is optional — a concept prompt
    needs no seed boxes, so "point at a video and name the thing" must work
    with no picking step at all."""
    if not required and not Path(path).exists():
        return 0, None, {}
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "rois" in raw:
        return raw.get("start_frame", 0), raw.get("end_frame"), raw["rois"]
    return 0, None, raw


# ── Quality control ────────────────────────────────────────────────────────────

def report_quality(
    results: dict[int, dict[int, Observation]],
    id_to_name: dict[int, str],
) -> dict[str, dict]:
    """Log per-object coverage, score and area stats; return them for the JSON.

    Coverage is the number that matters: SAM 3 silently returning nothing for an
    object is the failure mode that a glance at the trail video does not reveal,
    because gap fill will have drawn a plausible straight line through it.
    """
    span = sorted(results)
    if not span:
        logging.warning("SAM 3 returned no frames at all")
        return {}

    stats: dict[str, dict] = {}
    for oid, name in sorted(id_to_name.items(), key=lambda kv: kv[1]):
        frames = [f for f in span if oid in results[f]]
        scores = [results[f][oid].score for f in frames]
        areas  = [results[f][oid].area  for f in frames]
        cover  = len(frames) / len(span) if span else 0.0

        # Longest run of consecutive tracked frames missing this object.
        worst_gap = 0
        run = 0
        present = set(frames)
        for f in span:
            run = 0 if f in present else run + 1
            worst_gap = max(worst_gap, run)

        stats[name] = {
            "obj_id":        oid,
            "frames_seen":   len(frames),
            "frames_total":  len(span),
            "coverage":      round(cover, 4),
            "longest_gap":   worst_gap,
            "score_mean":    round(float(np.mean(scores)), 4) if scores else 0.0,
            "score_min":     round(float(np.min(scores)),  4) if scores else 0.0,
            "area_median":   int(np.median(areas)) if areas else 0,
            "area_min":      int(np.min(areas))    if areas else 0,
            "area_max":      int(np.max(areas))    if areas else 0,
        }
        level = logging.WARNING if cover < 0.9 or worst_gap > 15 else logging.INFO
        logging.log(
            level,
            f"{name}: coverage {100 * cover:.1f}% ({len(frames)}/{len(span)}), "
            f"longest gap {worst_gap} frames, score min/mean "
            f"{stats[name]['score_min']:.2f}/{stats[name]['score_mean']:.2f}, "
            f"area median {stats[name]['area_median']}px"
        )
        if stats[name]["area_median"] and stats[name]["area_max"] > 20 * stats[name]["area_median"]:
            logging.warning(
                f"{name}: mask area spikes to {stats[name]['area_max']}px vs median "
                f"{stats[name]['area_median']}px — likely latched onto the background"
            )
    return stats


# ── Debug video ────────────────────────────────────────────────────────────────

def write_debug_video(
    out_path: str,
    window: FrameWindow,
    results: dict[int, dict[int, Observation]],
    id_to_name: dict[int, str],
    color: dict[str, tuple[int, int, int]],
    masks: MaskStore | None,
    fps: float,
) -> None:
    """Annotated video: mask tint, bbox, label, score, and the trail so far."""
    frames = sorted(results)
    if not frames:
        logging.warning("No results — skipping debug video")
        return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter.fourcc(*"mp4v"), fps,
        (window.width, window.height),
    )
    # VideoWriter reports failure only here — it does not raise, and every
    # subsequent write() is silently discarded.
    if not writer.isOpened():
        logging.error(f"Cannot open debug video for writing: {out_path}")
        return
    history: dict[str, list[tuple[int, int]]] = {n: [] for n in id_to_name.values()}

    for abs_idx in frames:
        frame = window.read_frame(abs_idx)
        if frame is None:
            continue
        per_obj = results[abs_idx]

        for oid, obs in sorted(per_obj.items()):
            name = id_to_name.get(oid, f"id{oid}")
            col  = color.get(name, (255, 255, 255))
            history.setdefault(name, []).append((obs.cx, obs.cy))

            m = masks.get(abs_idx, name) if masks is not None else obs.mask
            if m is not None and m.shape[:2] == frame.shape[:2]:
                frame[m] = (frame[m] * 0.45 + np.array(col) * 0.55).astype(np.uint8)

            x, y, w, h = obs.bbox
            cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 0), 4)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            cv2.circle(frame, (obs.cx, obs.cy), DOT_RADIUS + 3, (0, 0, 0), -1)
            cv2.circle(frame, (obs.cx, obs.cy), DOT_RADIUS + 1, col, -1)

            label = f"{name} {obs.score:.2f}"
            pos   = (obs.cx + 10, obs.cy - 10)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 1)

        for name, pts in history.items():
            if len(pts) >= 2:
                cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False,
                              color.get(name, (255, 255, 255)), 2)

        missing = [n for oid, n in id_to_name.items() if oid not in per_obj]
        info = f"Frame {abs_idx}  SAM3  {len(per_obj)} obj"
        if missing:
            info += f"  MISSING: {','.join(sorted(missing))}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 1)
        writer.write(frame)

    writer.release()
    logging.info(f"Debug → {out_path}")


# ── Borrowed agents (hybrid mode) ──────────────────────────────────────────────

def borrow_trails(
    src_json: str, names: list[str],
) -> dict[str, dict[int, tuple[int, int]]]:
    """Lift named agents out of another tracking JSON (e.g. track.py's output).

    Lets SAM 3 handle the well-defined objects while the background-subtraction
    tracker keeps the one it is better at — the small low-contrast payload.
    """
    with open(src_json) as f:
        d = json.load(f)
    out: dict[str, dict[int, tuple[int, int]]] = {}
    for name in names:
        if name not in d.get("trails", {}):
            raise KeyError(
                f"--borrow-agents {name}: not in {src_json} "
                f"(has {sorted(d.get('trails', {}))})"
            )
        out[name] = {int(f): tuple(p) for f, p in d["trails"][name].items()}
        logging.info(f"Borrowed {name} from {src_json} ({len(out[name])} frames)")
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", default=VIDEO_IN)
    p.add_argument("--rois",  default=ROIS_FILE)
    p.add_argument("--out",   default=VIDEO_OUT, help="output path stem")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--from-rois", action="store_true", default=True,
                      help="seed objects from rois.json boxes (default)")
    mode.add_argument("--text", metavar="PROMPT",
                      help='concept prompt, e.g. "drone" — discovers objects')

    p.add_argument("--name", action="append", default=[], metavar="ID=NAME",
                   help="rename a discovered object id (repeatable, --text mode)")
    p.add_argument("--agents", default=None,
                   help="comma-separated subset of rois.json agents to track")

    p.add_argument("--start-frame", type=int, default=None,
                   help="override rois.json start_frame")
    p.add_argument("--end-frame", type=int, default=None,
                   help="override rois.json end_frame")

    p.add_argument("--prompt-shape", choices=("box", "point"), default="box",
                   help="seed with the ROI box or with its centre point")
    p.add_argument("--box-key", default=None,
                   help="force the SAM 3 box-prompt keyword instead of probing")
    p.add_argument("--norm-boxes", action="store_true",
                   help="send prompt coordinates normalised to 0-1")
    p.add_argument("--point-mode", choices=("centroid", "bbox"), default="centroid",
                   help="reduce each mask to its barycentre or its bbox centre")

    p.add_argument("--reprompt-every", type=int, default=0, metavar="N",
                   help="re-seed every N frames; also caps memory-bank VRAM")
    p.add_argument("--chunk-overlap", type=int, default=8)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--min-area", type=int, default=1)
    p.add_argument("--mask-threshold", type=float, default=0.0)

    p.add_argument("--zoom-agents", default=None, metavar="NAMES",
                   help="comma-separated agents to re-track through an upscaled "
                        "crop (for objects too small to survive downscaling)")
    p.add_argument("--zoom-crop", type=int, default=384)
    p.add_argument("--zoom-upscale", type=int, default=3)
    p.add_argument("--zoom-segment", type=int, default=60)

    p.add_argument("--borrow-agents", default=None, metavar="NAMES",
                   help="comma-separated agents to take from --borrow-from instead")
    p.add_argument("--borrow-from", default=None, metavar="JSON",
                   help="tracking JSON to borrow those agents from")

    p.add_argument("--no-gap-fill", action="store_true",
                   help="leave dropouts as holes instead of interpolating")
    p.add_argument("--min-gap-frames", type=int, default=5)
    p.add_argument("--save-masks", action="store_true",
                   help="write a *_masks.npz sidecar for render.py --mask-mode")
    p.add_argument("--no-debug-video", action="store_true")
    p.add_argument("--frames-dir", default=None,
                   help="keep extracted frames here instead of a temp dir")
    p.add_argument("--gpus", default=None, help="comma-separated CUDA device ids")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="SAM 3 weights file. Defaults to $COMET_SAM3_CHECKPOINT, "
                        "then ~/ws/models/weights/sam3.pt; if none is found, "
                        "upstream downloads the gated checkpoint from HF")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _split(csv: str | None) -> list[str]:
    return [s.strip() for s in csv.split(",") if s.strip()] if csv else []


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args([])

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if LOG_TO_FILE:
        handlers.append(logging.FileHandler(LOG_FILE, mode="a"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, handlers=handlers,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    start_frame, end_frame, rois = load_rois(args.rois, required=not args.text)
    if args.start_frame is not None:
        start_frame = args.start_frame
    if args.end_frame is not None:
        end_frame = args.end_frame

    wanted = _split(args.agents)
    if wanted:
        missing = [n for n in wanted if n not in rois]
        if missing:
            sys.exit(f"--agents: {missing} not in {args.rois} (has {sorted(rois)})")
        rois = {n: rois[n] for n in wanted}

    borrow_names = _split(args.borrow_agents)
    if borrow_names and not args.borrow_from:
        sys.exit("--borrow-agents requires --borrow-from")
    # Borrowed agents are supplied by the other tracker, so don't seed them here.
    sam_rois = {n: b for n, b in rois.items() if n not in borrow_names}
    if not sam_rois and args.text is None:
        sys.exit("Nothing left for SAM 3 to track after --borrow-agents")

    # Masks are needed for the sidecar and for the debug overlay.  They are
    # never held as raw arrays: the sink run-length encodes each one on arrival
    # and the Observation drops it, which is the difference between a few MB and
    # several GB of resident memory on a 1080p clip.
    want_masks = args.save_masks or not args.no_debug_video

    cfg = Sam3Config(
        normalize_coords = args.norm_boxes,
        box_key          = args.box_key,
        point_mode       = args.point_mode,
        mask_threshold   = args.mask_threshold,
        min_score        = args.min_score,
        min_area         = args.min_area,
        chunk_size       = max(0, args.reprompt_every),
        chunk_overlap    = args.chunk_overlap,
        gpus             = [int(g) for g in _split(args.gpus)] or None,
        checkpoint       = args.checkpoint,
        keep_masks       = want_masks,
    )

    window = FrameWindow(args.video, start_frame, end_frame)
    try:
        window.extract(args.frames_dir)

        # Keyed by object id while tracking runs; renamed once the id → name
        # mapping is known (text mode only settles it after propagation).
        masks = MaskStore(window.height, window.width) if want_masks else None
        if masks is not None:
            cfg.mask_sink = lambda f, oid, m: masks.add(f, str(oid), m)

        predictor = build_predictor(cfg)
        try:
            if args.text:
                results = track_window_chunked(
                    window, cfg, predictor=predictor, text=args.text,
                )
                name_map = {}
                for spec in args.name:
                    if "=" not in spec:
                        sys.exit(f"--name expects ID=NAME, got {spec!r}")
                    k, v = spec.split("=", 1)
                    name_map[int(k)] = v
                id_to_name = discover_id_names(results, name_map=name_map)
                logging.info(
                    f"Discovered {len(id_to_name)} object(s): "
                    + ", ".join(f"{o}→{n}" for o, n in sorted(id_to_name.items()))
                )
            else:
                prompts = [
                    PromptSpec(name=name, obj_id=i, box=tuple(float(v) for v in box))
                    for i, (name, box) in enumerate(sorted(sam_rois.items()))
                ]
                id_to_name = {p.obj_id: p.name for p in prompts}
                results = track_window_chunked(
                    window, cfg, predictor=predictor, prompts=prompts,
                    prompt_shape=args.prompt_shape,
                )

            # ── Zoom pass for objects too small to track at full frame ────────
            for name in _split(args.zoom_agents):
                oid = next((o for o, n in id_to_name.items() if n == name), None)
                if oid is None:
                    sys.exit(f"--zoom-agents: unknown agent {name!r}")
                if name not in rois:
                    sys.exit(f"--zoom-agents {name}: no seed box in {args.rois}")
                logging.info(f"Zoom-tracking {name} (crop {args.zoom_crop}px "
                             f"×{args.zoom_upscale})")
                zoom = track_object_zoom(
                    window, cfg, tuple(float(v) for v in rois[name]), oid,
                    predictor=predictor, crop_size=args.zoom_crop,
                    upscale=args.zoom_upscale, segment_len=args.zoom_segment,
                )
                replaced = 0
                for abs_idx, obs in zoom.items():
                    results.setdefault(abs_idx, {})[oid] = obs
                    replaced += 1
                logging.info(f"Zoom pass supplied {replaced} frames for {name}")
        finally:
            if hasattr(predictor, "shutdown"):
                try:
                    predictor.shutdown()
                except Exception as e:          # noqa: BLE001 - cleanup only
                    logging.warning(f"predictor.shutdown() failed: {e}")

        stats = report_quality(results, id_to_name)

        if masks is not None:
            masks.rename({str(oid): name for oid, name in id_to_name.items()})

        # ── Debug video ───────────────────────────────────────────────────────
        if not args.no_debug_video:
            debug_out = str(Path(args.out).with_name(
                Path(args.out).stem + "_debug.mp4"))
            write_debug_video(debug_out, window, results, id_to_name,
                              build_palette(list(id_to_name.values()), TRAIL_COLOR),
                              masks, window.fps)

        # ── Trails ────────────────────────────────────────────────────────────
        trails = observations_to_trails(results, id_to_name)
        if borrow_names:
            trails.update(borrow_trails(args.borrow_from, borrow_names))

        if not args.no_gap_fill:
            before = {n: len(t) for n, t in trails.items()}
            # No detection corridor to search: SAM 3 gives us tracked objects,
            # not a pool of candidate blobs, so holes resolve to interpolation
            # between the frames on either side.
            trails = fill_gaps_bidirectional(
                trails, {}, min_gap_frames=args.min_gap_frames,
            )
            for n, t in trails.items():
                if len(t) > before.get(n, 0):
                    logging.info(
                        f"{n}: interpolated {len(t) - before[n]} gap frames"
                    )

        color = build_palette(sorted(trails), TRAIL_COLOR)
        data = build_tracking_data(
            video_in        = args.video,
            fps             = window.fps,
            total_frames    = window.total_frames,
            width           = window.width,
            height          = window.height,
            start_frame     = start_frame,
            end_frame       = end_frame,
            trail_start_sec = TRAIL_START_SEC,
            trail_end_sec   = TRAIL_END_SEC,
            trail_color     = color,
            trail_thickness = TRAIL_THICKNESS,
            alpha           = ALPHA,
            trail_window    = TRAIL_WINDOW,
            smooth_trails   = SMOOTH_TRAILS,
            trails          = trails,
            extra           = {
                "tracker":     "sam3",
                "sam3_prompt": {"text": args.text} if args.text
                               else {"rois": sorted(sam_rois)},
                "sam3_stats":  stats,
            },
        )
        data_out = str(Path(args.out).with_name(
            Path(args.out).stem + "_tracking.json"))
        write_tracking_json(data_out, data)
        logging.info(f"Tracking data → {data_out}")

        if masks is not None and args.save_masks:
            mp = masks.save(sidecar_path(data_out))
            logging.info(f"Masks → {mp} ({len(masks)} masks)")
    finally:
        if args.frames_dir is None:
            window.cleanup()


if __name__ == "__main__":
    main(parse_args())

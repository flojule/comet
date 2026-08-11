"""render.py — post-process trail videos from tracking data exported by track.py.

Usage:
    python src/render.py                              # uses default data file
    python src/render.py output/crazyflo_path_tracking.json
    python src/render.py output/crazyflo_sam3_tracking.json --mask-mode occlude

Outputs (MP4, next to the JSON file):
    <stem>_persistent.mp4
    <stem>_transient.mp4

Any trail property can be overridden via the RENDER_* env vars or by editing
the OVERRIDES dict at the top of this file.

Mask modes (need a *_masks.npz sidecar, written by
`track_sam3.py --save-masks`; ignored when there is none):

    off       trails drawn straight over the frame, as before
    occlude   trails pass BEHIND the objects — each object's own pixels are
              restored on top of the trail canvas
    glow      occlude, plus a coloured halo around each object's silhouette
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from maskstore import MaskStore, sidecar_path
from trails import blend_trail, smooth_pts

# ── Optional per-run overrides (None = use value from JSON) ────────────────────
OVERRIDES: dict = {
    # "trail_thickness": 3,
    # "alpha": 0.6,
    # "trail_window": 200,
    # "smooth_trails": True,
    # "trail_color": {"cf1": [0, 0, 255], "cf2": [0, 255, 0], "cf3": [255, 0, 0], "payload": [50, 50, 50]},
}

DEFAULT_DATA_FILE = "output/crazyflo_path_tracking.json"

GLOW_RADIUS = 9    # px — halo thickness in "glow" mode
GLOW_ALPHA  = 0.7


def composite_masks(
    base: np.ndarray,
    blended: np.ndarray,
    masks: dict[str, np.ndarray],
    color: dict[str, tuple[int, int, int]],
    mode: str,
) -> np.ndarray:
    """Put objects back in front of the trail canvas.

    `blended` already has trails painted over `base`.  Restoring the original
    pixels wherever an object's mask sits makes the trail read as passing behind
    the object instead of being smeared across it.
    """
    if mode == "off" or not masks:
        return blended

    out = blended
    if mode == "glow":
        halo = np.zeros_like(base)
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (GLOW_RADIUS * 2 + 1, GLOW_RADIUS * 2 + 1))
        for name, m in masks.items():
            u = m.astype(np.uint8)
            ring = cv2.dilate(u, kern) & (1 - u)   # dilated minus the object
            halo[ring.astype(bool)] = color.get(name, (255, 255, 255))
        halo = cv2.GaussianBlur(halo, (0, 0), GLOW_RADIUS / 2.0)
        out = blend_trail(out, halo, GLOW_ALPHA)

    union = np.zeros(base.shape[:2], dtype=bool)
    for m in masks.values():
        union |= m
    out = out.copy()
    out[union] = base[union]
    return out


def load_masks(data_file: str, want: bool) -> MaskStore | None:
    if not want:
        return None
    p = sidecar_path(data_file)
    if not p.exists():
        print(f"  no mask sidecar at {p} — falling back to --mask-mode off")
        return None
    store = MaskStore.load(p)
    print(f"  masks: {len(store)} from {p}")
    return store


def render(data_file: str = DEFAULT_DATA_FILE, mask_mode: str = "off") -> None:
    with open(data_file) as f:
        d = json.load(f)

    # Apply overrides
    for k, v in OVERRIDES.items():
        if v is not None:
            d[k] = v

    video_in        = d["video_in"]
    fps             = float(d["fps"])
    total           = int(d["total_frames"])
    W               = int(d["width"])
    H               = int(d["height"])
    trail_start_sec = float(d["trail_start_sec"])
    trail_end_sec   = float(d["trail_end_sec"])
    trail_color     = {k: tuple(v) for k, v in d["trail_color"].items()}
    thickness       = int(d["trail_thickness"])
    alpha           = float(d["alpha"])
    trail_window    = int(d["trail_window"])
    smooth          = bool(d["smooth_trails"])

    # Trails: {agent: [(frame_idx, (x, y)), ...]} sorted by frame
    trails: dict[str, list[tuple[int, tuple[int, int]]]] = {
        name: sorted((int(fi), tuple(pt)) for fi, pt in pts.items())
        for name, pts in d["trails"].items()
    }

    trail_start_frame = int(trail_start_sec * fps)
    trail_end_frame   = total - int(trail_end_sec * fps)

    # Dynamic transient window from payload speed
    payload_seq = trails.get("payload", [])
    if len(payload_seq) >= 10:
        steps = [
            ((payload_seq[k+1][1][0] - payload_seq[k][1][0])**2
             + (payload_seq[k+1][1][1] - payload_seq[k][1][1])**2)**0.5
            for k in range(len(payload_seq) - 1)
        ]
        valid = [s for s in steps if s > 0.1]
        if valid:
            trail_window = max(20, int(W / float(np.median(valid)) * 0.45))

    stem   = Path(data_file).stem.removesuffix("_tracking")
    parent = Path(data_file).parent
    out_p  = parent / (stem + "_persistent.mp4")
    out_t  = parent / (stem + "_transient.mp4")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    head   = {name: 0 for name in trails}

    store = load_masks(data_file, mask_mode != "off")
    if store is None:
        mask_mode = "off"

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        sys.exit(f"Cannot open {video_in}")

    wr_p = cv2.VideoWriter(str(out_p), fourcc, fps, (W, H))
    wr_t = cv2.VideoWriter(str(out_t), fourcc, fps, (W, H))

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        show_trail  = trail_start_frame <= fi <= trail_end_frame
        drain_trail = (not show_trail) and (fi <= trail_end_frame + trail_window)

        if not show_trail and not drain_trail:
            wr_p.write(frame)
            wr_t.write(frame)
            continue

        # Advance head pointers only while actively tracking
        if show_trail:
            for name, seq in trails.items():
                while head[name] < len(seq) and seq[head[name]][0] <= fi:
                    head[name] += 1

        # Masks for this frame, if a sidecar is loaded.
        frame_masks: dict[str, np.ndarray] = {}
        if mask_mode != "off":
            for name in trails:
                m = store.get(fi, name)
                if m is not None and m.shape == (H, W):
                    frame_masks[name] = m

        # ── Persistent ────────────────────────────────────────────────────────
        if show_trail:
            cp = np.zeros((H, W, 3), dtype=np.uint8)
            for name, seq in trails.items():
                pts  = [pt for f, pt in seq[:head[name]] if f >= trail_start_frame]
                draw = smooth_pts(pts) if (smooth and len(pts) >= 2) else pts
                if len(draw) >= 2:
                    cv2.polylines(cp, [np.array(draw, dtype=np.int32)],
                                  False, trail_color[name], thickness)
            wr_p.write(composite_masks(
                frame, blend_trail(frame, cp, alpha),
                frame_masks, trail_color, mask_mode))
        else:
            wr_p.write(frame)

        # ── Transient (shooting star: thick bright head → thin faded tail) ───
        # During drain_trail, heads are frozen so the window scrolls the tail
        # away at the normal rate — no sudden disappearance.
        ct = np.zeros((H, W, 3), dtype=np.uint8)
        for name, seq in trails.items():
            window = [(f, pt) for f, pt in seq[:head[name]]
                      if fi - trail_window <= f]
            if len(window) < 2:
                continue
            draw = smooth_pts([pt for _, pt in window]) if smooth else [pt for _, pt in window]
            for k in range(len(draw) - 1):
                age = fi - window[k + 1][0]
                w   = max(0.0, 1.0 - age / trail_window)
                col = tuple(int(c * w) for c in trail_color[name])
                lw  = max(1, round(thickness * w ** 0.5))
                cv2.line(ct, draw[k], draw[k + 1], col, lw)
        wr_t.write(composite_masks(
            frame, blend_trail(frame, ct, alpha),
            frame_masks, trail_color, mask_mode))

        if fi % 30 == 0:
            sys.stdout.write(f"\r  {100*fi/total:.0f}%")
            sys.stdout.flush()

    sys.stdout.write("\r  100%\n")
    sys.stdout.flush()
    cap.release()
    wr_p.release()
    wr_t.release()

    print(f"Done.\n  {out_p}\n  {out_t}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("data_file", nargs="?", default=DEFAULT_DATA_FILE)
    p.add_argument("--mask-mode", choices=("off", "occlude", "glow"), default="off",
                   help="use the *_masks.npz sidecar to composite objects over "
                        "the trails (default: off)")
    return p.parse_args(argv)


if __name__ == "__main__":
    _a = parse_args()
    render(_a.data_file, _a.mask_mode)

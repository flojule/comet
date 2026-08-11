#!/usr/bin/env python3
# trails.py
"""Shared trail geometry, smoothing, gap-fill and tracking-JSON I/O.

Extracted from track.py so that every tracker backend (background subtraction
in track.py, SAM 3 in track_sam3.py) produces byte-compatible tracking data and
render.py has a single implementation to consume.

The tracking JSON contract — every backend must emit exactly this:

    {
      "video_in": str, "fps": float, "total_frames": int,
      "width": int, "height": int,
      "start_frame": int, "end_frame": int | null,
      "trail_start_sec": float, "trail_end_sec": float,
      "trail_color": {agent: [b, g, r]},
      "trail_thickness": int, "alpha": float, "trail_window": int,
      "smooth_trails": bool,
      "trails": {agent: {"<frame_idx>": [cx, cy]}}
    }

`trails` is keyed by ABSOLUTE frame index in the source video (a string, since
JSON object keys are strings), not by an index into any extracted sub-range.
render.py walks `range(total_frames)` of the original video and matches on
these keys, so an off-by-start_frame error here silently desynchronises every
trail from the footage.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


# ── Geometry ───────────────────────────────────────────────────────────────────

def box_center(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return x + w // 2, y + h // 2


def safe_append(trail: list, pt: tuple[int, int], jump_threshold: float) -> None:
    """Append `pt` unless it is a teleport away from the last point."""
    if not trail:
        trail.append(pt)
        return
    lx, ly = trail[-1]
    if ((lx - pt[0]) ** 2 + (ly - pt[1]) ** 2) ** 0.5 > jump_threshold:
        return
    trail.append(pt)


def filter_trail(trail: list, max_dev: int = 120) -> list:
    if not trail:
        return []
    out = [trail[0]]
    for p in trail[1:]:
        lx, ly = out[-1]
        if ((p[0] - lx) ** 2 + (p[1] - ly) ** 2) ** 0.5 <= max_dev:
            out.append(p)
    return out


# ── Smoothing ──────────────────────────────────────────────────────────────────

def smooth_pts(pts: list) -> list:
    """Savitzky-Golay smooth for real-time trail rendering."""
    n = len(pts)
    if n < 5:
        return list(pts)
    win = min(15, n if n % 2 == 1 else n - 1)  # odd, <= n, >= 5
    xs = savgol_filter([p[0] for p in pts], win, 3)
    ys = savgol_filter([p[1] for p in pts], win, 3)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def spline_smooth(pts: list) -> list:
    """
    Global smoothing spline fitted to the full trail.
    Used in post-processing where the complete trajectory is known.
    UnivariateSpline (s > 0) does not interpolate exactly — it finds a smooth
    curve that minimises squared residuals, which is exactly what we want for
    noisy tracking data.
    """
    n = len(pts)
    if n < 4:
        return list(pts)
    t    = np.arange(n, dtype=float)
    xs   = np.array([p[0] for p in pts], dtype=float)
    ys   = np.array([p[1] for p in pts], dtype=float)
    s    = n * 4.0   # smoothing factor — larger → smoother curve
    sp_x = UnivariateSpline(t, xs, s=s, k=3)
    sp_y = UnivariateSpline(t, ys, s=s, k=3)
    return [(int(float(sp_x(ti))), int(float(sp_y(ti)))) for ti in t]


# ── Compositing ────────────────────────────────────────────────────────────────

def blend_trail(frame: np.ndarray, canvas: np.ndarray, alpha: float) -> np.ndarray:
    mask = canvas.any(axis=2)
    out  = frame.copy()
    out[mask] = cv2.addWeighted(frame, 1 - alpha, canvas, alpha, 0)[mask]
    return out


# ── Gap fill ───────────────────────────────────────────────────────────────────

def fill_gaps_bidirectional(
    full_trail_log:      dict[str, dict[int, tuple[int, int]]],
    full_det_log:        dict[int, list],
    corridor_half_width: float = 80.0,
    min_gap_frames:      int   = 5,
    y_bands:             dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[int, tuple[int, int]]]:
    """
    Post-process gap fill using both the gap start and end positions.

    For each gap of ≥ min_gap_frames consecutive missing frames in an agent's
    trail, we:
      1. Define a corridor: points within `corridor_half_width` px of the
         straight line between the gap's anchor and recovery positions.
      2. For every gap frame, pick the nearest detection inside that corridor.
      3. If ≥ 25 % of gap frames found a detection, use them (+ linear interp
         for the remainder).  Otherwise fall back to pure linear interpolation.

    `full_det_log` maps frame index → list of (x, y, w, h) candidate boxes.  A
    SAM 3 backend can pass an empty dict, in which case every gap resolves to
    linear interpolation between the known endpoints.

    Returns a dict {agent_name: {frame_idx: (cx, cy)}} that covers all frames
    from the first to last tracked frame, gaps filled in.
    """
    result: dict[str, dict[int, tuple[int, int]]] = {}
    for name, frame_pts in full_trail_log.items():
        if not frame_pts:
            result[name] = {}
            continue

        frames = sorted(frame_pts.keys())
        filled: dict[int, tuple[int, int]] = dict(frame_pts)

        for i in range(len(frames) - 1):
            fa = frames[i]
            fb = frames[i + 1]
            gap_len = fb - fa - 1
            if gap_len < min_gap_frames:
                continue

            start_pt = frame_pts[fa]
            end_pt   = frame_pts[fb]
            sx, sy   = start_pt
            ex, ey   = end_pt
            seg_dx   = ex - sx
            seg_dy   = ey - sy
            seg_len2 = seg_dx ** 2 + seg_dy ** 2

            sub: dict[int, tuple[int, int]] = {}
            y_lo, y_hi = (y_bands[name] if y_bands and name in y_bands
                          else (-float("inf"), float("inf")))
            for gf in range(fa + 1, fb):
                dets    = full_det_log.get(gf, [])
                best_d  = float("inf")
                best_pt: tuple[int, int] | None = None
                for dx, dy, dw, dh in dets:
                    dcx = dx + dw / 2
                    dcy = dy + dh / 2
                    if not (y_lo <= dcy <= y_hi):
                        continue
                    # Perpendicular distance from detection to line segment
                    if seg_len2 < 1.0:
                        d_line = ((dcx - sx) ** 2 + (dcy - sy) ** 2) ** 0.5
                    else:
                        t = ((dcx - sx) * seg_dx + (dcy - sy) * seg_dy) / seg_len2
                        t = max(0.0, min(1.0, t))
                        d_line = ((dcx - (sx + t * seg_dx)) ** 2
                                  + (dcy - (sy + t * seg_dy)) ** 2) ** 0.5
                    if d_line < corridor_half_width and d_line < best_d:
                        best_d  = d_line
                        best_pt = (int(dcx), int(dcy))
                if best_pt:
                    sub[gf] = best_pt

            gap_frames = list(range(fa + 1, fb))
            if len(sub) >= max(1, len(gap_frames) * 0.25):
                for gf in gap_frames:
                    if gf in sub:
                        filled[gf] = sub[gf]
                    else:
                        t = (gf - fa) / (fb - fa)
                        filled[gf] = (int(sx + seg_dx * t), int(sy + seg_dy * t))
            else:
                for gf in gap_frames:
                    t = (gf - fa) / (fb - fa)
                    filled[gf] = (int(sx + seg_dx * t), int(sy + seg_dy * t))

            logging.debug(
                f"Gap-fill {name}: {fa}→{fb} ({gap_len} frames), "
                f"det hits={len(sub)}/{len(gap_frames)}"
            )

        result[name] = filled
    return result


# ── Palette ────────────────────────────────────────────────────────────────────

# Distinct BGR colours for agents that have no explicit entry (text-prompt mode
# discovers objects at runtime, so their names are not known in advance).
FALLBACK_PALETTE: list[tuple[int, int, int]] = [
    (  0,   0, 255),   # red
    (  0, 255,   0),   # green
    (255,   0,   0),   # blue
    (  0, 255, 255),   # yellow
    (255,   0, 255),   # magenta
    (255, 255,   0),   # cyan
    (  0, 128, 255),   # orange
    (128,   0, 255),   # pink
    (255, 128,   0),   # azure
    (128, 255,   0),   # spring
]


def build_palette(
    names: list[str],
    explicit: dict[str, tuple[int, int, int]] | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Colour for every name, preferring `explicit` and cycling the fallback.

    render.py indexes `trail_color[name]` unconditionally, so a name missing
    from this dict is a KeyError at render time rather than a missing trail.
    """
    explicit = explicit or {}
    out: dict[str, tuple[int, int, int]] = {}
    nxt = 0
    for name in names:
        if name in explicit:
            out[name] = tuple(explicit[name])
        else:
            out[name] = FALLBACK_PALETTE[nxt % len(FALLBACK_PALETTE)]
            nxt += 1
    return out


# ── Tracking JSON I/O ──────────────────────────────────────────────────────────

def build_tracking_data(
    *,
    video_in:        str,
    fps:             float,
    total_frames:    int,
    width:           int,
    height:          int,
    start_frame:     int,
    end_frame:       int | None,
    trail_start_sec: float,
    trail_end_sec:   float,
    trail_color:     dict[str, tuple[int, int, int]],
    trail_thickness: int,
    alpha:           float,
    trail_window:    int,
    smooth_trails:   bool,
    trails:          dict[str, dict[int, tuple[int, int]]],
    extra:           dict | None = None,
) -> dict:
    """Assemble the tracking dict in the exact schema render.py expects."""
    data = {
        "video_in":        video_in,
        "fps":             fps,
        "total_frames":    total_frames,
        "width":           width,
        "height":          height,
        "start_frame":     start_frame,
        "end_frame":       end_frame,
        "trail_start_sec": trail_start_sec,
        "trail_end_sec":   trail_end_sec,
        "trail_color":     {k: list(v) for k, v in trail_color.items()},
        "trail_thickness": trail_thickness,
        "alpha":           alpha,
        "trail_window":    trail_window,
        "smooth_trails":   smooth_trails,
        "trails": {
            name: {str(fi): list(pt) for fi, pt in sorted(fd.items())}
            for name, fd in trails.items()
        },
    }
    if extra:
        data.update(extra)
    return data


def write_tracking_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return path

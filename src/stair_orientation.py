#!/usr/bin/env python3
# stair_orientation.py
"""Estimate stair orientation from a per-frame segmentation mask.

A staircase in an image is a set of near-parallel lines — the tread nosings.
Their shared direction is the quantity of interest, so the estimator finds
straight edges inside the stair mask and reports the dominant direction.

    mask ──▶ edges inside mask ──▶ Hough segments ──▶ angle histogram ──▶ angle

WHAT THIS IS AND IS NOT
-----------------------
The output is an angle in the IMAGE PLANE, in degrees, measured from the
positive x-axis and wrapped to [0, 180) because a line has no direction — 10°
and 190° are the same line.

It is NOT the stairs' 3D orientation relative to the robot.  Recovering that
needs camera intrinsics and either the vanishing point of the nosing lines or
the depth stream; this module deliberately stops short of that, because an
angle that silently mixes image-plane and world-frame conventions is worse
than one with a documented meaning.  `angle_deg` is directly useful for
alignment ("is the robot square to the flight?") and as the input to that
later 3D step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class OrientationConfig:
    canny_lo: int = 50
    canny_hi: int = 150
    # Hough segment length as a fraction of the mask's smaller bbox side, so
    # the threshold scales with how much of the frame the stairs occupy.
    min_line_frac: float = 0.25
    max_line_gap: int = 10
    hough_threshold: int = 30
    # Angle histogram resolution.  5° trades a little precision for stability
    # against noisy single segments.
    bin_deg: float = 5.0
    # Reject a frame whose dominant bin holds less than this share of the
    # total segment length — that means there is no consistent direction.
    min_dominant_share: float = 0.30
    min_segments: int = 3
    smooth_window: int = 9      # frames; odd, 0 disables


@dataclass
class Orientation:
    """One frame's estimate."""
    frame: int
    angle_deg: float | None            # [0, 180), None when indeterminate
    confidence: float                  # share of segment length in the peak bin
    n_segments: int
    total_length: float
    segments: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.angle_deg is not None


def _wrap180(deg: float) -> float:
    """Fold an angle into [0, 180) — a line and its reverse are the same line."""
    return float(deg % 180.0)


def circular_mean_180(angles: list[float], weights: list[float] | None = None) -> float:
    """Mean of angles that live on a 180° circle.

    Averaging 179° and 1° arithmetically gives 90°, which is perpendicular to
    both.  Doubling the angles maps them onto a full circle where the vector
    mean behaves, then halving brings the result back.
    """
    if not angles:
        raise ValueError("no angles")
    a = np.deg2rad(np.asarray(angles, dtype=float) * 2.0)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    x = float((w * np.cos(a)).sum())
    y = float((w * np.sin(a)).sum())
    if x == 0.0 and y == 0.0:
        return _wrap180(float(np.mean(angles)))
    return _wrap180(np.rad2deg(np.arctan2(y, x)) / 2.0)


def estimate_frame(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    frame_idx: int = 0,
    cfg: OrientationConfig | None = None,
) -> Orientation:
    """Dominant edge direction inside `mask`."""
    cfg = cfg or OrientationConfig()
    if mask is None or not mask.any():
        return Orientation(frame_idx, None, 0.0, 0, 0.0)

    ys, xs = np.nonzero(mask)
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    min_len = max(8, int(min(bw, bh) * cfg.min_line_frac))

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.canny_lo, cfg.canny_hi)

    # Restrict to the stairs.  Eroding first drops the mask outline itself,
    # which is a strong edge in every direction and would otherwise dominate
    # the histogram with the silhouette rather than the nosings.
    m8 = mask.astype(np.uint8)
    inner = cv2.erode(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edges = cv2.bitwise_and(edges, edges, mask=inner)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=cfg.hough_threshold,
        minLineLength=min_len, maxLineGap=cfg.max_line_gap,
    )
    if lines is None or len(lines) < cfg.min_segments:
        n = 0 if lines is None else len(lines)
        return Orientation(frame_idx, None, 0.0, n, 0.0)

    # OpenCV ≤4 returns (N, 1, 4); OpenCV 5 returns (N, 4).  Normalise both.
    lines = np.asarray(lines).reshape(-1, 4)

    segs, angles, lengths = [], [], []
    for x1, y1, x2, y2 in lines:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        segs.append((int(x1), int(y1), int(x2), int(y2)))
        angles.append(_wrap180(np.rad2deg(np.arctan2(dy, dx))))
        lengths.append(length)

    if len(segs) < cfg.min_segments:
        return Orientation(frame_idx, None, 0.0, len(segs), float(sum(lengths)))

    # Length-weighted histogram: a long nosing is better evidence than a short
    # speck of texture.
    nbins = int(round(180.0 / cfg.bin_deg))
    hist = np.zeros(nbins)
    for a, ln in zip(angles, lengths):
        hist[int(a // cfg.bin_deg) % nbins] += ln

    # Include the neighbouring bins so a peak straddling a boundary is not split.
    peak = int(np.argmax(hist))
    neigh = [(peak - 1) % nbins, peak, (peak + 1) % nbins]
    total = float(hist.sum())
    share = float(hist[neigh].sum() / total) if total else 0.0

    sel = [(a, ln) for a, ln in zip(angles, lengths)
           if int(a // cfg.bin_deg) % nbins in neigh]
    angle = circular_mean_180([a for a, _ in sel], [ln for _, ln in sel])

    if share < cfg.min_dominant_share:
        return Orientation(frame_idx, None, share, len(segs), total, segs)
    return Orientation(frame_idx, angle, share, len(segs), total, segs)


def smooth_series(results: list[Orientation], window: int = 9) -> list[Orientation]:
    """Median-style smoothing over the 180° circle, skipping bad frames.

    Applied after estimation rather than during it, so a frame that genuinely
    had no answer stays `None` instead of inheriting its neighbours'.
    """
    if window < 3 or len(results) < 3:
        return results
    half = window // 2
    out: list[Orientation] = []
    for i, r in enumerate(results):
        if not r.ok:
            out.append(r)
            continue
        lo, hi = max(0, i - half), min(len(results), i + half + 1)
        nearby = [(x.angle_deg, x.confidence) for x in results[lo:hi] if x.ok]
        if len(nearby) < 2:
            out.append(r)
            continue
        ang = circular_mean_180([a for a, _ in nearby], [c for _, c in nearby])
        out.append(Orientation(r.frame, ang, r.confidence, r.n_segments,
                               r.total_length, r.segments))
    return out


def annotate(frame_bgr: np.ndarray, o: Orientation,
             color: tuple[int, int, int] = (0, 220, 255),
             draw_segments: bool = True) -> np.ndarray:
    """Draw the detected segments and an angle readout onto a copy of a frame."""
    out = frame_bgr.copy()
    if draw_segments:
        for x1, y1, x2, y2 in o.segments:
            cv2.line(out, (x1, y1), (x2, y2), color, 1)

    if o.ok:
        h, w = out.shape[:2]
        cx, cy = w // 2, h // 2
        th = np.deg2rad(o.angle_deg)
        r = int(min(w, h) * 0.3)
        p0 = (int(cx - r * np.cos(th)), int(cy - r * np.sin(th)))
        p1 = (int(cx + r * np.cos(th)), int(cy + r * np.sin(th)))
        cv2.line(out, p0, p1, (0, 0, 0), 5)
        cv2.line(out, p0, p1, color, 2)
        label = f"stair angle {o.angle_deg:5.1f}deg  conf {o.confidence:.2f}"
    else:
        label = f"stair angle --  ({o.n_segments} segments, no dominant direction)"

    cv2.putText(out, label, (10, out.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(out, label, (10, out.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return out


def summarise(results: list[Orientation]) -> dict:
    """Aggregate stats for the tracking JSON."""
    good = [r for r in results if r.ok]
    if not good:
        logging.warning("Stair orientation: no frame produced an estimate")
        return {"frames": len(results), "frames_with_estimate": 0,
                "coverage": 0.0, "angle_deg_mean": None,
                "angle_deg_std": None, "confidence_mean": 0.0}

    angles = [r.angle_deg for r in good]
    mean = circular_mean_180(angles)
    # Spread about the circular mean, folded to ±90.
    dev = [((a - mean + 90.0) % 180.0) - 90.0 for a in angles]
    return {
        "frames": len(results),
        "frames_with_estimate": len(good),
        "coverage": round(len(good) / len(results), 4),
        "angle_deg_mean": round(mean, 2),
        "angle_deg_std": round(float(np.std(dev)), 2),
        "confidence_mean": round(float(np.mean([r.confidence for r in good])), 3),
        "per_frame": {str(r.frame): (round(r.angle_deg, 2) if r.ok else None)
                      for r in results},
    }

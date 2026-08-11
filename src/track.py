#!/usr/bin/env python3
# track.py
"""Track objects using foreground detection + global Hungarian assignment."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment as _scipy_lsa

from trails import (
    blend_trail,
    box_center,
    build_tracking_data,
    smooth_pts,
    write_tracking_json,
)
from trails import safe_append as _safe_append


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    r, c = _scipy_lsa(cost)
    return list(zip(r.tolist(), c.tolist()))


# ── Configuration ──────────────────────────────────────────────────────────────
LOG_TO_FILE           = True
REWIND_FRAMES         = 150  # frames of detection history to keep for gap-fill
SMOOTH_TRAILS         = True

VIDEO_IN  = "input/crazyflo.mp4"
ROIS_FILE = "src/rois.json"
VIDEO_OUT = "output/crazyflo_path.mp4"

TRAIL_COLOR: dict[str, tuple[int, int, int]] = {
    "cf1":     (  0,   0, 255),   # red
    "cf2":     (  0, 255,   0),   # green
    "cf3":     (255,   0,   0),   # blue
    "payload": ( 50,  50,  50),
}
TRAIL_THICKNESS = 3
DOT_RADIUS      = 5
ALPHA           = 0.6
trail_start_sec = 2
trail_end_sec   = 2
TRAIL_WINDOW    = 200   # frames kept in fading transient clean output

# Detection & assignment
MIN_BLOB_AREA   = 20    # px²  — ignore tiny contours
MERGE_RADIUS    = 50    # px   — merge nearby blobs into one before assignment
MAX_ASSIGN_DIST = 260   # px   — reject assignments farther than this
APPEAR_WEIGHT   = 0.07  # contribution of histogram distance to cost matrix
SIZE_WEIGHT     = 0.20  # contribution of bbox-area deviation to cost matrix
VELOCITY_WINDOW = 6     # trail points used to estimate velocity
VELOCITY_DECAY  = 0.85  # velocity multiplier per lost frame (0 < x < 1)

# Lost / recovery
LOST_THRESHOLD  = 20    # consecutive unmatched frames before logging a warning
JUMP_THRESHOLD  = 220   # px — skip trail append if position jumps this far
MAX_CLAMP_DIST  = 200   # px — max distance for CLAMP_WHEN_LOST fallback assignment

# Per-agent vertical (Y) clamp: maximum allowed |Δy| from last known position.
# Applied in the primary assignment pass — keeps the payload from stealing a drone blob.
AGENT_Y_CLAMP: dict[str, float] = {
    "payload": 120.0,   # generous enough for normal motion, blocks drone-level blobs
}

# Agents that fall back to nearest-unclaimed-blob when the primary assignment
# leaves them unmatched.  No distance, size, or Y constraints — runs AFTER drones
# have already claimed their detections, so it can't cause identity swaps.
CLAMP_WHEN_LOST: set[str] = {"payload"}


# ── Pure helpers ───────────────────────────────────────────────────────────────

def safe_append(trail: list, pt: tuple[int, int]) -> None:
    """Module-local binding of the shared helper at this tracker's threshold."""
    _safe_append(trail, pt, JUMP_THRESHOLD)


def estimate_velocity(trail: list) -> tuple[float, float]:
    if len(trail) < 2:
        return (0.0, 0.0)
    seg   = trail[-min(VELOCITY_WINDOW + 1, len(trail)):]
    steps = len(seg) - 1
    return (
        (seg[-1][0] - seg[0][0]) / steps,
        (seg[-1][1] - seg[0][1]) / steps,
    )


def extrapolate(trail: list, vel: tuple[float, float]) -> tuple[float, float] | None:
    if not trail:
        return None
    return (trail[-1][0] + vel[0], trail[-1][1] + vel[1])


def calc_hist(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    x, y, w, h = (max(0, int(v)) for v in bbox)
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def hist_dist(h1: np.ndarray | None, h2: np.ndarray | None) -> float:
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(
        h1.reshape(-1, 1).astype(np.float32),
        h2.reshape(-1, 1).astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    ))


def merge_detections(dets: list[tuple]) -> list[tuple]:
    """Union-find merge of bounding boxes whose centres are within MERGE_RADIUS."""
    if not dets:
        return []
    n      = len(dets)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    centres = [(x + w / 2, y + h / 2) for x, y, w, h in dets]
    for i in range(n):
        for j in range(i + 1, n):
            d = ((centres[i][0] - centres[j][0]) ** 2
                 + (centres[i][1] - centres[j][1]) ** 2) ** 0.5
            if d < MERGE_RADIUS:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        xs  = [dets[k][0]                for k in idxs]
        ys  = [dets[k][1]                for k in idxs]
        x2s = [dets[k][0] + dets[k][2]  for k in idxs]
        y2s = [dets[k][1] + dets[k][3]  for k in idxs]
        mx, my = min(xs), min(ys)
        merged.append((mx, my, max(x2s) - mx, max(y2s) - my))
    return merged


def build_fg_mask(frame: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Standard mask — tuned for drone-sized blobs."""
    diff = cv2.absdiff(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(bg,    cv2.COLOR_BGR2GRAY),
    )
    _, m = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m    = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kern)
    m    = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
    return m


def build_fg_mask_sensitive(frame: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """High-sensitivity mask — lower threshold + small open kernel so thin/small
    payload blobs survive the morphological pass."""
    diff = cv2.absdiff(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(bg,    cv2.COLOR_BGR2GRAY),
    )
    _, m = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    kern_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kern_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern_close)
    return m


def get_detections(fg: np.ndarray) -> list[tuple]:
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = [tuple(cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) >= MIN_BLOB_AREA]
    return merge_detections(raw)


def global_assign(
    names:       list[str],
    preds:       dict[str, tuple | None],
    dets:        list[tuple],
    hists:       dict[str, np.ndarray | None],
    det_hists:   list[np.ndarray | None],
    last_ys:     dict[str, float] | None = None,
    agent_areas: dict[str, float] | None = None,
) -> list[tuple[str, int]]:
    """
    Build a cost matrix and run Hungarian assignment.
    Returns a list of (agent_name, detection_index) pairs.
    Each agent and each detection appears at most once.

    last_ys:     last-known Y centre for agents with an AGENT_Y_CLAMP entry.
    agent_areas: expected bbox area (px²) per agent; used to penalise size changes.
    """
    n_a, n_d = len(names), len(dets)
    if n_a == 0 or n_d == 0:
        return []

    INF  = 1e6
    cost = np.full((n_a, n_d), INF, dtype=np.float32)

    for i, name in enumerate(names):
        p = preds[name]
        if p is None:
            continue
        px, py   = p
        y_clamp  = AGENT_Y_CLAMP.get(name)
        anchor_y = (last_ys or {}).get(name, py)
        area_exp = (agent_areas or {}).get(name, 0.0)

        for j, (dx, dy, dw, dh) in enumerate(dets):
            dcx = dx + dw / 2
            dcy = dy + dh / 2

            # Hard vertical gate
            if y_clamp is not None and abs(dcy - anchor_y) > y_clamp:
                continue

            dist = ((dcx - px) ** 2 + (dcy - py) ** 2) ** 0.5
            if dist > MAX_ASSIGN_DIST:
                continue

            # Appearance cost
            app = hist_dist(hists.get(name), det_hists[j])

            # Size-consistency cost: penalise detections whose area differs from
            # the agent's running average.  Uses sqrt(area) so the scale is linear
            # in side-length rather than area.  Capped at 1.5× to stay bounded.
            if area_exp > 0:
                area_det  = float(dw * dh)
                size_diff = abs(area_det ** 0.5 - area_exp ** 0.5) / max(area_exp ** 0.5, 1.0)
                size_cost = min(size_diff, 1.5)
            else:
                size_cost = 0.0

            cost[i, j] = (dist
                          + APPEAR_WEIGHT * MAX_ASSIGN_DIST * app
                          + SIZE_WEIGHT   * MAX_ASSIGN_DIST * size_cost)

    return [
        (names[r], c)
        for r, c in _hungarian(cost)
        if cost[r, c] < INF
    ]


def _replay_trail(
    name:        str,
    state:       dict,
    lost_since:  int,
    recovery_pt: tuple[int, int],
    det_buffer:  list,
    all_states:  dict,
) -> None:
    """
    Fill the trail gap for `name` between lost_since and the current frame.

    Strategy:
      1. Walk forward through the buffered per-frame detections.
      2. At each step, predict where the agent should be (velocity extrapolation).
      3. Pick the nearest unclaimed detection — "claimed" means too close to
         any other agent's current trail endpoint (used as a spatial proxy).
      4. If ≥25 % of gap frames matched a detection, use them.
         Otherwise fall back to a straight-line interpolation from the last
         known position to recovery_pt.
    """
    gap = [(fi, ds) for fi, ds in det_buffer if lost_since <= fi]
    if not gap:
        return

    # Other agents' latest positions — proxy for "this detection is taken"
    other_pos = [
        st["trail"][-1]
        for n, st in all_states.items()
        if n != name and st["trail"]
    ]

    anchor = state["trail"][-1]   # last good position before loss
    vel    = state["velocity"]    # velocity at time of loss
    sub: list[tuple[int, int]] = []

    for _, frame_dets in gap:
        prev = sub[-1] if sub else anchor
        px   = prev[0] + vel[0]
        py   = prev[1] + vel[1]

        best_dist: float         = MAX_ASSIGN_DIST
        best_pt: tuple | None    = None

        for dx, dy, dw, dh in frame_dets:
            dcx = dx + dw / 2
            dcy = dy + dh / 2

            # Vertical clamp (e.g. payload height stability)
            if name in AGENT_Y_CLAMP and abs(dcy - prev[1]) > AGENT_Y_CLAMP[name]:
                continue

            # Skip detections near other agents' current positions
            if any(
                ((dcx - ocx) ** 2 + (dcy - ocy) ** 2) ** 0.5 < MERGE_RADIUS * 2
                for ocx, ocy in other_pos
            ):
                continue

            dist = ((dcx - px) ** 2 + (dcy - py) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_pt   = (int(dcx), int(dcy))

        if best_pt:
            sub.append(best_pt)
            prev2 = sub[-2] if len(sub) >= 2 else anchor
            vx = best_pt[0] - prev2[0]
            vy = best_pt[1] - prev2[1]
            vel = (vx * 0.4 + vel[0] * 0.6, vy * 0.4 + vel[1] * 0.6)
        else:
            vel = (vel[0] * VELOCITY_DECAY, vel[1] * VELOCITY_DECAY)

    if len(sub) >= max(1, len(gap) * 0.25):
        for pt in sub:
            safe_append(state["trail"], pt)
        logging.debug(f"Gap-fill: {name} matched {len(sub)}/{len(gap)} frames from detection replay")
    else:
        # Too sparse — linear interpolation from anchor to recovery_pt
        n = len(gap)
        for k in range(1, n + 1):
            t = k / (n + 1)
            safe_append(state["trail"], (
                int(anchor[0] + (recovery_pt[0] - anchor[0]) * t),
                int(anchor[1] + (recovery_pt[1] - anchor[1]) * t),
            ))
        logging.debug(f"Gap-fill: {name} interpolated {n} frames (only {len(sub)} detections found)")


def render_trails(canvas: np.ndarray, trails: dict[str, list], show: bool) -> None:
    if not show:
        return
    canvas[:] = 0
    for agent, pts in trails.items():
        if len(pts) < 2:
            continue
        draw = smooth_pts(pts) if SMOOTH_TRAILS else pts
        cv2.polylines(
            canvas,
            [np.array(draw, dtype=np.int32)],
            False,
            TRAIL_COLOR[agent],
            TRAIL_THICKNESS,
        )


def draw_debug(
    frame:     np.ndarray,
    states:    dict,
    frame_idx: int,
    total:     int,
    fps:       float,
) -> np.ndarray:
    out = frame.copy()
    for name, st in states.items():
        if not st["trail"]:
            continue
        cx, cy = st["trail"][-1]
        color  = TRAIL_COLOR.get(name, (255, 255, 255))

        # EMA-smoothed bbox — thick border with black outline for contrast
        bbox = st.get("bbox_disp") or st.get("bbox")
        if bbox:
            x, y, bw, bh = (int(v) for v in bbox)
            cv2.rectangle(out, (x - 2, y - 2), (x + bw + 2, y + bh + 2), (0, 0, 0), 5)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 3)

        # Centre dot with black outline
        cv2.circle(out, (int(cx), int(cy)), DOT_RADIUS + 4, (0, 0, 0), -1)
        cv2.circle(out, (int(cx), int(cy)), DOT_RADIUS + 2, color, -1)

        # Label with black stroke for legibility
        label    = name if st["lost"] == 0 else f"{name} LOST:{st['lost']}"
        lx, ly   = int(cx) + 10, int(cy) - 10
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    # Frame counter
    info = f"Frame {frame_idx}/{total}   {int(fps)} fps"
    cv2.putText(out, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 1)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-display", action="store_true",
                   help="run without a GUI window (required on headless machines)")
    p.add_argument("--video", default=VIDEO_IN, help=f"input video (default: {VIDEO_IN})")
    p.add_argument("--rois", default=ROIS_FILE, help=f"ROI file (default: {ROIS_FILE})")
    p.add_argument("--out", default=VIDEO_OUT, help=f"output path stem (default: {VIDEO_OUT})")
    return p.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args([])
    video_in  = args.video
    rois_file = args.rois
    video_out = args.out
    display   = not args.no_display

    # ── Logging ────────────────────────────────────────────────────────────────
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if LOG_TO_FILE:
        handlers.append(logging.FileHandler("track_and_annotate.log", mode="a"))
    logging.basicConfig(
        level=logging.INFO, handlers=handlers,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # ── Load ROIs ──────────────────────────────────────────────────────────────
    with open(rois_file) as f:
        raw = json.load(f)
    start_frame: int       = raw.get("start_frame", 0)   if isinstance(raw, dict) else 0
    end_frame:   int | None = raw.get("end_frame",   None) if isinstance(raw, dict) else None
    rois: dict              = raw["rois"] if isinstance(raw, dict) and "rois" in raw else raw

    # ── Optional input re-encode ───────────────────────────────────────────────
    proc_video = video_in
    temp_mp4   = None
    if not video_in.lower().endswith(".mp4"):
        try:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_mp4 = tmpf.name
            tmpf.close()
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", video_in,
                 "-c:v", "libx264", "-crf", "18", "-preset", "fast", temp_mp4],
                capture_output=True, timeout=300,
            )
            if r.returncode == 0:
                proc_video = temp_mp4
                logging.info(f"Re-encoded input → {temp_mp4}")
        except Exception as e:
            logging.warning(f"Input re-encode failed: {e}")
            temp_mp4 = None

    cap   = cv2.VideoCapture(proc_video)
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Read first frame (needed before background model) ─────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read start_frame={start_frame}")

    # ── Background model ───────────────────────────────────────────────────────
    # Train ONLY on frames 0 → start_frame (pre-takeoff).
    # This ensures the drones on the ground become part of the background, so
    # when they lift off at start_frame they appear cleanly as foreground.
    # Training on the full video would bake flying-drone positions into the
    # background, creating ghost blobs at ground positions during tracking.
    logging.debug(f"Building background model from frames 0→{start_frame} (pre-takeoff) …")
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=min(2000, start_frame + 1), varThreshold=25, detectShadows=False,
    )
    step = max(1, int(fps_v // 2))
    for i in range(0, start_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        r, f = cap.read()
        if not r:
            break
        bg_sub.apply(f, learningRate=0.01)
    bg_img = bg_sub.getBackgroundImage()
    if bg_img is None:
        logging.warning("Background image not available — using first frame as fallback")
        bg_img = first_frame.copy()
    logging.debug("Background model built")

    # ── Save debug background images ────────────────────────────────────────────
    dbg_dir = Path(video_out).parent
    cv2.imwrite(str(dbg_dir / "debug_background.png"), bg_img)
    cv2.imwrite(str(dbg_dir / "debug_first_frame.png"), first_frame)
    # Foreground mask at start_frame — shows exactly which blobs the detector fires on
    fg_sample = build_fg_mask(first_frame, bg_img)
    fg_vis    = cv2.cvtColor(fg_sample, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(dbg_dir / "debug_fg_mask_start.png"), fg_vis)
    # Scaled diff: |first_frame − background| ×3 — good for spotting weak contrast
    diff_vis = cv2.absdiff(first_frame, bg_img)
    diff_vis = cv2.convertScaleAbs(diff_vis, alpha=3.0)
    cv2.imwrite(str(dbg_dir / "debug_diff_start.png"), diff_vis)
    # Overlay: first_frame with FG mask tinted red — shows detected blobs in context
    overlay = first_frame.copy()
    overlay[fg_sample == 255] = (overlay[fg_sample == 255] * 0.4 + np.array([0, 0, 200]) * 0.6).astype(np.uint8)
    cv2.imwrite(str(dbg_dir / "debug_fg_overlay_start.png"), overlay)
    logging.debug(f"Debug images → {dbg_dir}/")

    fourcc_mp4 = cv2.VideoWriter.fourcc(*"mp4v")

    # Reset main capture to start_frame + 1 (first_frame already consumed)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 1)

    # ── Per-agent state ────────────────────────────────────────────────────────
    # trail:      [(cx, cy), …] accumulated centroid path
    # bbox:       last-known (x, y, w, h)
    # lost:       consecutive frames without a detection match
    # velocity:   (vx, vy) px/frame, decays when lost
    # hist:       EMA colour histogram (HSV) of agent appearance
    # last_seen:  frame index of last successful match
    states: dict[str, dict] = {}
    for name, roi in rois.items():
        cx, cy = box_center(*roi)
        states[name] = dict(
            trail     = [(cx, cy)],
            bbox      = tuple(roi),
            bbox_disp = tuple(roi),             # EMA-smoothed bbox for display
            avg_area  = float(roi[2] * roi[3]), # EMA bbox area for size-consistency
            avg_y     = float(cy),              # slow EMA of Y — stable height anchor
            lost      = 0,
            velocity  = (0.0, 0.0),
            hist      = calc_hist(first_frame, roi),
            last_seen = start_frame,
        )

    trail_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    # det_buffer: rolling window of (frame_idx, dets) for gap-fill on recovery
    det_buffer: list[tuple[int, list]] = []
    # lost_since: frame when each agent first went unmatched (cleared on recovery)
    lost_since: dict[str, int] = {}
    # full_det_log: ALL per-frame detections — used by bidirectional gap-fill
    full_det_log: dict[int, list] = {}
    # full_trail_log: direct (non-gap-fill) matches per agent, keyed by frame
    full_trail_log: dict[str, dict[int, tuple[int, int]]] = {n: {} for n in states}

    frame_idx = start_frame + 1
    if display:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 1600, 900)
    logging.debug(f"Tracking frames {start_frame}→{end_frame or total - 1}")

    # Debug writer covers only the tracked segment (start_frame → end_frame)
    debug_out    = str(Path(video_out).with_name(Path(video_out).stem + "_debug.mp4"))
    debug_writer = cv2.VideoWriter(debug_out, fourcc_mp4, fps_v, (W, H))

    while True:
        if end_frame is not None and frame_idx > end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. Detect foreground blobs ─────────────────────────────────────────
        fg        = build_fg_mask(frame, bg_img)
        dets      = get_detections(fg)

        # ── Payload supplemental detection ────────────────────────────────────
        # The standard 5×5 open can erase the payload's thin blob.  Re-run with
        # a sensitive mask restricted to the payload's expected Y-band and inject
        # any new blobs not already represented in `dets`.
        if "payload" in states and states["payload"]["trail"]:
            pay_st   = states["payload"]
            pay_y    = pay_st.get("avg_y", float(pay_st["trail"][-1][1]))
            y_clamp  = AGENT_Y_CLAMP.get("payload", 120.0)
            y_lo     = max(0, int(pay_y - y_clamp))
            y_hi     = min(frame.shape[0], int(pay_y + y_clamp))
            sens_fg  = build_fg_mask_sensitive(frame, bg_img)
            band_fg  = np.zeros_like(sens_fg)
            band_fg[y_lo:y_hi, :] = sens_fg[y_lo:y_hi, :]
            pay_dets = get_detections(band_fg)
            for pd in pay_dets:
                pcx = pd[0] + pd[2] / 2
                pcy = pd[1] + pd[3] / 2
                if not any(
                    abs((d[0] + d[2] / 2) - pcx) < MERGE_RADIUS and
                    abs((d[1] + d[3] / 2) - pcy) < MERGE_RADIUS
                    for d in dets
                ):
                    dets.append(pd)

        det_hists = [calc_hist(frame, tuple(d)) for d in dets]

        if frame_idx % 120 == 0 and not dets:
            logging.warning(f"[{frame_idx}] No foreground detections — check background model or lighting")

        # Rolling detection buffer (bboxes only — no pixels needed for gap-fill)
        det_buffer.append((frame_idx, list(dets)))
        if len(det_buffer) > REWIND_FRAMES:
            det_buffer.pop(0)
        # Full detection log — kept for the post-process bidirectional gap fill
        full_det_log[frame_idx] = list(dets)

        # ── 2. Predict next positions from velocity ────────────────────────────
        preds = {
            name: extrapolate(st["trail"], st["velocity"])
            for name, st in states.items()
        }
        hists = {name: st["hist"] for name, st in states.items()}

        # ── 3. Global, mutually-exclusive assignment ───────────────────────────
        #   Each detection is matched to at most one agent, and vice-versa.
        #   This prevents identity swaps when drones cross paths.
        # Use slow-EMA avg_y as Y anchor — more stable than last trail point
        last_ys = {
            name: st.get("avg_y", float(st["trail"][-1][1]))
            for name, st in states.items()
            if name in AGENT_Y_CLAMP and st["trail"]
        }
        agent_areas = {name: st["avg_area"] for name, st in states.items()}
        matches        = global_assign(list(states.keys()), preds, dets, hists, det_hists, last_ys, agent_areas)
        matched_agents = {name for name, _ in matches}

        # ── 4. Update matched agents ───────────────────────────────────────────
        for name, di in matches:
            det    = dets[di]
            cx, cy = box_center(*det)
            st     = states[name]

            # Gap-fill: if this agent was lost, rewind through buffered detections
            if st["lost"] > 0 and name in lost_since:
                _replay_trail(name, st, lost_since[name], (cx, cy), det_buffer, states)
                del lost_since[name]

            safe_append(st["trail"], (cx, cy))
            st["bbox"] = det
            bx, by, bw, bh = det
            # EMA-smooth the displayed bbox to eliminate frame-to-frame jitter
            if st.get("bbox_disp"):
                ox, oy, ow, oh = st["bbox_disp"]
                A = 0.3
                st["bbox_disp"] = (int(ox*(1-A) + bx*A), int(oy*(1-A) + by*A),
                                   int(ow*(1-A) + bw*A), int(oh*(1-A) + bh*A))
            else:
                st["bbox_disp"] = (bx, by, bw, bh)
            # Slow EMA on expected area — resists sudden size jumps from noise
            st["avg_area"] = st["avg_area"] * 0.85 + float(bw * bh) * 0.15
            st["lost"]      = 0
            st["velocity"]  = estimate_velocity(st["trail"])
            # EMA histogram update — slow adaptation keeps appearance stable
            new_h = det_hists[di]
            if new_h is not None and st["hist"] is not None:
                st["hist"] = 0.7 * st["hist"] + 0.3 * new_h
            elif new_h is not None:
                st["hist"] = new_h
            st["last_seen"] = frame_idx
            st["avg_y"]     = st["avg_y"] * 0.98 + float(cy) * 0.02
            full_trail_log[name][frame_idx] = (cx, cy)

        # ── 4b. Nearest-blob clamp for payload-type agents ────────────────────
        # Runs after all primary assignments are committed.  Picks the closest
        # unclaimed detection with no distance/size/Y constraints so the payload
        # is never left dark when any moving blob exists near its last position.
        claimed_dets = {di for _, di in matches}
        for name in CLAMP_WHEN_LOST:
            if name in matched_agents or name not in states:
                continue
            st = states[name]
            if not st["trail"]:
                continue
            last_cx, last_cy = st["trail"][-1]
            best_dist: float        = float("inf")
            best_di:   int | None   = None
            clamp_anchor_y = st.get("avg_y", float(last_cy))
            for di, (dx, dy, dw, dh) in enumerate(dets):
                if di in claimed_dets:
                    continue
                dcx  = dx + dw / 2
                dcy  = dy + dh / 2
                # Apply the same Y gate as the primary pass, using stable avg_y
                if name in AGENT_Y_CLAMP and abs(dcy - clamp_anchor_y) > AGENT_Y_CLAMP[name]:
                    continue
                dist = ((dcx - last_cx) ** 2 + (dcy - last_cy) ** 2) ** 0.5
                if dist < best_dist and dist <= MAX_CLAMP_DIST:
                    best_dist = dist
                    best_di   = di
            if best_di is not None:
                det    = dets[best_di]
                cx, cy = box_center(*det)
                if st["lost"] > 0 and name in lost_since:
                    _replay_trail(name, st, lost_since[name], (cx, cy), det_buffer, states)
                    del lost_since[name]
                safe_append(st["trail"], (cx, cy))
                st["bbox"] = det
                bx, by, bw, bh = det
                if st.get("bbox_disp"):
                    ox, oy, ow, oh = st["bbox_disp"]
                    A = 0.3
                    st["bbox_disp"] = (int(ox*(1-A) + bx*A), int(oy*(1-A) + by*A),
                                       int(ow*(1-A) + bw*A), int(oh*(1-A) + bh*A))
                else:
                    st["bbox_disp"] = (bx, by, bw, bh)
                st["avg_area"]  = st["avg_area"] * 0.85 + float(bw * bh) * 0.15
                st["lost"]      = 0
                st["velocity"]  = estimate_velocity(st["trail"])
                new_h = det_hists[best_di]
                if new_h is not None and st["hist"] is not None:
                    st["hist"] = 0.7 * st["hist"] + 0.3 * new_h
                elif new_h is not None:
                    st["hist"] = new_h
                st["last_seen"] = frame_idx
                st["avg_y"]     = st["avg_y"] * 0.98 + float(cy) * 0.02
                matched_agents.add(name)
                full_trail_log[name][frame_idx] = (cx, cy)
                logging.debug(f"[{frame_idx}] {name} clamped to nearest blob at dist={best_dist:.0f}px")

        # ── 5. Handle unmatched (lost) agents ─────────────────────────────────
        for name, st in states.items():
            if name in matched_agents:
                continue

            # Record the first frame of this loss so _replay_trail knows where to start
            if name not in lost_since:
                lost_since[name] = frame_idx

            st["lost"] += 1
            # Decay velocity so stale predictions don't drift far from last position
            vx, vy = st["velocity"]
            st["velocity"] = (vx * VELOCITY_DECAY, vy * VELOCITY_DECAY)

            if st["lost"] == LOST_THRESHOLD:
                logging.info(f"[{frame_idx}] {name} lost for {LOST_THRESHOLD} frames — waiting for re-detection")

        # ── 6. Draw trails and write frame ─────────────────────────────────────
        current_sec = frame_idx / fps_v
        show_trail  = (
            current_sec >= trail_start_sec
            and current_sec <= (total / fps_v - trail_end_sec)
        )
        trails = {n: st["trail"] for n, st in states.items()}
        render_trails(trail_canvas, trails, show_trail)

        annotated = blend_trail(frame, trail_canvas, ALPHA) if show_trail else frame.copy()
        dbg = draw_debug(annotated, states, frame_idx, total, fps_v)
        debug_writer.write(dbg)
        if display:
            cv2.imshow("Tracking", dbg)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                logging.info("Quit early")
                break

        if frame_idx % 30 == 0:
            sys.stdout.write(f"\r  tracking {100 * frame_idx / total:.0f}%")
            sys.stdout.flush()

        frame_idx += 1

    sys.stdout.write("\r  tracking 100%\n")
    sys.stdout.flush()

    cap.release()
    debug_writer.release()
    if display:
        cv2.destroyAllWindows()
    logging.info(f"Debug → {debug_out}")

    # ── Temp cleanup ───────────────────────────────────────────────────────────
    if temp_mp4:
        try:
            Path(temp_mp4).unlink()
        except Exception:
            pass

    # ── Export tracking data for post-processing ───────────────────────────────
    data_out = str(Path(video_out).with_name(Path(video_out).stem + "_tracking.json"))
    tracking_data = build_tracking_data(
        video_in        = video_in,
        fps             = fps_v,
        total_frames    = total,
        width           = W,
        height          = H,
        start_frame     = start_frame,
        end_frame       = end_frame,
        trail_start_sec = trail_start_sec,
        trail_end_sec   = trail_end_sec,
        trail_color     = TRAIL_COLOR,
        trail_thickness = TRAIL_THICKNESS,
        alpha           = ALPHA,
        trail_window    = TRAIL_WINDOW,
        smooth_trails   = SMOOTH_TRAILS,
        trails          = full_trail_log,
    )
    write_tracking_json(data_out, tracking_data)
    logging.info(f"Tracking data → {data_out}")


if __name__ == "__main__":
    main(parse_args())

#!/usr/bin/env python3
# sam3_backend.py
"""SAM 3 video-tracking backend for Comet.

Wraps Meta's `Sam3VideoPredictor` session API and reduces its per-frame mask
output to the (frame_index → point) form the Comet render pipeline consumes.

────────────────────────────────────────────────────────────────────────────────
API ASSUMPTIONS
────────────────────────────────────────────────────────────────────────────────
Everything this module assumes about the upstream SAM 3 package is collected in
`_ADAPTER` below and in `normalize_frame_outputs()`.  The session verbs
(`start_session` / `add_prompt` / `propagate_in_video` / `close_session`) and
the text- and point-prompt payloads are taken from the official example
notebook:

    https://github.com/facebookresearch/sam3/blob/main/examples/
        sam3_video_predictor_example.ipynb

Two things that notebook does NOT pin down, and which this module therefore
probes at runtime rather than hard-codes:

  * the keyword for a BOX prompt — tried in order from `BOX_PROMPT_KEYS`;
  * the exact shape of `response["outputs"]` — `normalize_frame_outputs()`
    accepts every plausible layout and raises a loud, descriptive error if it
    meets one it does not recognise.

If upstream changes, fix it here; nothing else in the repo touches SAM 3.

Nothing in this module imports torch or sam3 at module scope, so the rest of
the pipeline (and the test suite) still runs on a machine without them.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ── Adapter constants ──────────────────────────────────────────────────────────

# Candidate keyword names for a box prompt, tried in order until one is accepted.
BOX_PROMPT_KEYS: tuple[str, ...] = ("box", "boxes", "bbox", "input_boxes")

# Keys searched when pulling masks / ids / scores out of a frame's outputs.
_MASK_KEYS  = ("pred_masks", "masks", "mask", "pred_mask", "masklets")
_ID_KEYS    = ("obj_ids", "object_ids", "obj_id", "ids", "objects")
_SCORE_KEYS = ("scores", "score", "obj_scores", "pred_scores", "confidences")

_ADAPTER = {
    "start":     "start_session",
    "prompt":    "add_prompt",
    "propagate": "propagate_in_video",
    "reset":     "reset_session",
    "close":     "close_session",
    "remove":    "remove_object",
}


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Sam3Config:
    """Knobs for a SAM 3 tracking run."""

    # Prompting
    normalize_coords: bool = False   # send box/point coords as 0-1 instead of px
    box_key: str | None    = None    # force a box-prompt keyword; None = probe

    # Mask → point reduction
    point_mode: str = "centroid"     # "centroid" (mask barycentre) or "bbox"
    mask_threshold: float = 0.0      # logit threshold; probabilities use 0.5

    # Filtering
    min_score: float = 0.0           # drop observations below this score
    min_area: int    = 1             # drop masks smaller than this many px

    # Long-video handling
    chunk_size: int = 0              # 0 = single pass over the whole window
    chunk_overlap: int = 8           # frames of context re-prompted per chunk

    # Runtime
    gpus: list[int] | None = None    # None = every visible CUDA device
    keep_masks: bool = False         # decode full masks at all
    # Called as sink(absolute_frame, obj_id, mask) for every mask, which is then
    # dropped from the Observation.  Masks are big — a 1080p boolean mask is
    # ~2 MB, so a 680-frame run with four objects would be ~5 GB if they were
    # all held in memory.  The sink lets the caller compress as they arrive.
    mask_sink: object | None = None
    extra_builder_kwargs: dict = field(default_factory=dict)


# ── Frame window ───────────────────────────────────────────────────────────────

class FrameWindow:
    """A [start_frame, end_frame] slice of a video, extracted as JPEG frames.

    SAM 3 indexes the frames it is given from 0.  Comet's tracking JSON is keyed
    by ABSOLUTE frame index in the source video, because render.py walks the
    original footage.  This class owns that mapping — every index crossing the
    boundary goes through `to_absolute` / `to_local` and nowhere else.
    """

    def __init__(self, video: str, start_frame: int, end_frame: int | None):
        self.video       = video
        self.start_frame = int(start_frame)
        self.end_frame   = end_frame
        self.dir: Path | None = None
        self._tmp: str | None = None
        self.n_frames = 0
        self.width    = 0
        self.height   = 0
        self.fps      = 30.0
        self.total_frames = 0

    # -- index mapping --------------------------------------------------------
    def to_absolute(self, local_idx: int) -> int:
        return self.start_frame + int(local_idx)

    def to_local(self, abs_idx: int) -> int:
        return int(abs_idx) - self.start_frame

    # -- extraction -----------------------------------------------------------
    def extract(self, dest: str | Path | None = None, quality: int = 95) -> Path:
        """Write the window's frames as zero-padded JPEGs and return the dir."""
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video}")

        self.fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        last = self.end_frame if self.end_frame is not None else self.total_frames - 1
        last = min(last, self.total_frames - 1)
        if last < self.start_frame:
            cap.release()
            raise ValueError(
                f"Empty frame window: start_frame={self.start_frame} > end_frame={last}"
            )

        if dest is None:
            self._tmp = tempfile.mkdtemp(prefix="comet_sam3_frames_")
            out_dir = Path(self._tmp)
        else:
            out_dir = Path(dest)
            out_dir.mkdir(parents=True, exist_ok=True)

        # Advance by grabbing rather than setting CAP_PROP_POS_FRAMES.  That
        # property's frame accuracy depends on the container and codec, while
        # sequential decode is exact for all of them — and it has to be exact,
        # because render.py reads the source from frame 0 sequentially, so any
        # discrepancy offsets every trail against the footage.  grab() skips the
        # decode of the frames we discard, so the scan stays cheap.
        for _ in range(self.start_frame):
            if not cap.grab():
                cap.release()
                raise ValueError(
                    f"Video has fewer than start_frame={self.start_frame} frames"
                )

        written = 0
        for abs_idx in range(self.start_frame, last + 1):
            ok, frame = cap.read()
            if not ok:
                logging.warning(
                    f"Video ended early at frame {abs_idx} (expected through {last})"
                )
                break
            cv2.imwrite(
                str(out_dir / f"{written:06d}.jpg"), frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            written += 1
        cap.release()

        if written == 0:
            raise RuntimeError(f"No frames extracted from {self.video}")

        self.n_frames = written
        self.dir = out_dir
        logging.info(
            f"Extracted {written} frames "
            f"({self.start_frame}→{self.to_absolute(written - 1)}) → {out_dir}"
        )
        return out_dir

    def read_frame(self, abs_idx: int) -> np.ndarray | None:
        """Read one extracted frame by absolute index (for debug rendering)."""
        if self.dir is None:
            return None
        p = self.dir / f"{self.to_local(abs_idx):06d}.jpg"
        if not p.exists():
            return None
        return cv2.imread(str(p))

    def cleanup(self) -> None:
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
            self.dir  = None

    def __enter__(self) -> FrameWindow:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


# ── Observations ───────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """One object, on one frame."""
    obj_id: int
    cx: int
    cy: int
    bbox: tuple[int, int, int, int]
    area: int
    score: float
    mask: np.ndarray | None = None   # bool HxW, only when keep_masks


def _as_tensor(data, dtype: str):
    """Build a float32/int32 tensor, falling back to numpy when torch is absent.

    Real runs always have torch — the predictor wants tensors.  The fallback
    exists so this module stays importable, and the prompt paths stay testable,
    on a machine with no GPU stack installed.
    """
    try:
        import torch
    except ImportError:
        return np.asarray(data, dtype=np.float32 if dtype == "float32" else np.int32)
    return torch.tensor(
        data, dtype=torch.float32 if dtype == "float32" else torch.int32,
    )


def _to_numpy(x):
    """Duck-typed tensor → ndarray, so this module never imports torch."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(x, attr):
            break
    else:
        return np.asarray(x)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def binarize_mask(raw, threshold: float = 0.0) -> np.ndarray:
    """Coerce a SAM mask of unknown dtype/shape into a 2-D bool array.

    SAM decoders emit logits (any real value, positive = foreground); some
    wrappers pre-apply a sigmoid, and some return bools already.  Guessing wrong
    turns a whole frame into foreground, so the three cases are separated
    explicitly rather than thresholded uniformly.
    """
    m = _to_numpy(raw)
    if m is None:
        raise ValueError("mask is None")
    m = np.squeeze(m)
    if m.ndim > 2:
        # e.g. [1, H, W] leftovers or a multi-hypothesis axis — take the first.
        m = m.reshape(-1, *m.shape[-2:])[0]
    if m.ndim != 2:
        raise ValueError(f"Cannot reduce mask of shape {np.shape(raw)} to 2-D")

    if m.dtype == np.bool_:
        return m
    if np.issubdtype(m.dtype, np.integer):
        return m > 0
    lo, hi = float(np.min(m)), float(np.max(m))
    if lo >= 0.0 and hi <= 1.0:
        return m >= 0.5          # probabilities
    return m > threshold         # logits


def mask_to_observation(
    mask: np.ndarray,
    obj_id: int,
    score: float,
    point_mode: str = "centroid",
    keep_mask: bool = False,
) -> Observation | None:
    """Reduce a boolean mask to a trail point + bbox.  None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    if point_mode == "bbox":
        cx = x0 + bbox[2] // 2
        cy = y0 + bbox[3] // 2
    else:
        # Mask barycentre: steadier than the bbox centre when an object is
        # partially occluded, since a clipped edge shifts the box but only
        # reweights the centroid.
        cx = int(round(float(xs.mean())))
        cy = int(round(float(ys.mean())))

    return Observation(
        obj_id = int(obj_id),
        cx     = cx,
        cy     = cy,
        bbox   = bbox,
        area   = int(xs.size),
        score  = float(score),
        mask   = mask if keep_mask else None,
    )


def _first_key(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def normalize_frame_outputs(
    outputs,
    frame_shape: tuple[int, int],
    cfg: Sam3Config,
) -> dict[int, Observation]:
    """Turn one frame's `response["outputs"]` into {obj_id: Observation}.

    Accepts the layouts SAM 3 / SAM 2 wrappers are known to emit:
      * {"pred_masks": [N,1,H,W], "obj_ids": [...], "scores": [...]}
      * {"masks": ..., "object_ids": ...}
      * {obj_id: mask}
      * {obj_id: {"mask": ..., "score": ...}}
      * an object exposing any of the above as attributes
    """
    H, W = frame_shape
    if outputs is None:
        return {}

    # Attribute-style container → dict of its interesting attributes.
    if not isinstance(outputs, dict):
        collected = {
            k: getattr(outputs, k)
            for k in (*_MASK_KEYS, *_ID_KEYS, *_SCORE_KEYS)
            if hasattr(outputs, k)
        }
        if not collected:
            raise TypeError(
                f"Unrecognised SAM 3 frame output of type {type(outputs)!r}; "
                f"extend normalize_frame_outputs() in sam3_backend.py"
            )
        outputs = collected

    masks  = _first_key(outputs, _MASK_KEYS)
    ids    = _first_key(outputs, _ID_KEYS)
    scores = _first_key(outputs, _SCORE_KEYS)

    pairs: list[tuple[int, object, float]] = []

    if masks is not None:
        arr = _to_numpy(masks)
        if arr is None:
            return {}
        # Collapse a singleton channel axis: [N,1,H,W] → [N,H,W]
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim == 2:                 # single unbatched mask
            arr = arr[None]
        if ids is None:
            ids = list(range(len(arr)))
        ids = [int(i) for i in _to_numpy(ids).tolist()] if not isinstance(ids, list) \
            else [int(i) for i in ids]
        if scores is None:
            score_list = [1.0] * len(arr)
        else:
            s = _to_numpy(scores)
            score_list = [float(v) for v in np.asarray(s).reshape(-1).tolist()]
            score_list += [1.0] * (len(arr) - len(score_list))
        if len(ids) != len(arr):
            raise ValueError(
                f"SAM 3 returned {len(arr)} masks but {len(ids)} object ids"
            )
        pairs = list(zip(ids, list(arr), score_list))
    else:
        # Mapping form: {obj_id: mask} or {obj_id: {"mask":..., "score":...}}
        for k, v in outputs.items():
            try:
                oid = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict):
                m = _first_key(v, _MASK_KEYS)
                s = _first_key(v, _SCORE_KEYS)
                pairs.append((oid, m, float(s) if s is not None else 1.0))
            else:
                pairs.append((oid, v, 1.0))
        if not pairs:
            raise TypeError(
                f"No masks found in SAM 3 frame output with keys {list(outputs)}; "
                f"extend normalize_frame_outputs() in sam3_backend.py"
            )

    obs: dict[int, Observation] = {}
    for oid, raw_mask, score in pairs:
        if raw_mask is None:
            continue
        mask = binarize_mask(raw_mask, cfg.mask_threshold)
        if mask.shape != (H, W):
            # SAM works at its own resolution; nearest-neighbour keeps it binary.
            mask = cv2.resize(
                mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        o = mask_to_observation(mask, oid, score, cfg.point_mode, cfg.keep_masks)
        if o is None or o.area < cfg.min_area or o.score < cfg.min_score:
            continue
        obs[o.obj_id] = o
    return obs


# ── Predictor session ──────────────────────────────────────────────────────────

class Sam3Session:
    """Context manager around one SAM 3 video session."""

    def __init__(self, predictor, resource_path: str | Path, cfg: Sam3Config):
        self.predictor = predictor
        self.resource_path = str(resource_path)
        self.cfg = cfg
        self.session_id = None
        self._box_key: str | None = cfg.box_key

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self) -> Sam3Session:
        resp = self.predictor.handle_request(request=dict(
            type=_ADAPTER["start"], resource_path=self.resource_path,
        ))
        self.session_id = resp["session_id"]
        logging.info(f"SAM 3 session {self.session_id} on {self.resource_path}")
        return self

    def __exit__(self, *exc) -> None:
        if self.session_id is not None:
            try:
                self.predictor.handle_request(request=dict(
                    type=_ADAPTER["close"], session_id=self.session_id,
                ))
            except Exception as e:              # noqa: BLE001 - cleanup only
                logging.warning(f"close_session failed: {e}")
            self.session_id = None

    def reset(self) -> None:
        """Clear all prompts.  Required before re-prompting the same session."""
        self.predictor.handle_request(request=dict(
            type=_ADAPTER["reset"], session_id=self.session_id,
        ))

    # -- prompts --------------------------------------------------------------
    def _scale(self, xs: list[float], ys: list[float], w: int, h: int):
        if not self.cfg.normalize_coords:
            return xs, ys
        return [x / w for x in xs], [y / h for y in ys]

    def add_text_prompt(self, text: str, frame_index: int = 0) -> dict:
        resp = self.predictor.handle_request(request=dict(
            type=_ADAPTER["prompt"], session_id=self.session_id,
            frame_index=int(frame_index), text=text,
        ))
        return resp.get("outputs")

    def add_point_prompt(
        self, points: list[tuple[float, float]], labels: list[int],
        obj_id: int, frame_index: int, frame_size: tuple[int, int],
    ) -> dict:
        h, w = frame_size
        xs, ys = self._scale([p[0] for p in points], [p[1] for p in points], w, h)
        pts = _as_tensor(list(zip(xs, ys)), "float32")
        lbl = _as_tensor(labels, "int32")
        resp = self.predictor.handle_request(request=dict(
            type=_ADAPTER["prompt"], session_id=self.session_id,
            frame_index=int(frame_index), points=pts, point_labels=lbl,
            obj_id=int(obj_id),
        ))
        return resp.get("outputs")

    def add_box_prompt(
        self, box_xywh: tuple[float, float, float, float],
        obj_id: int, frame_index: int, frame_size: tuple[int, int],
    ) -> dict:
        """Add an (x, y, w, h) box prompt, probing for the accepted keyword.

        The upstream example notebook only documents point and text prompts, so
        the box keyword is discovered on first use and then reused.
        """
        h, w = frame_size
        x, y, bw, bh = box_xywh
        xs, ys = self._scale([x, x + bw], [y, y + bh], w, h)
        box = _as_tensor([[xs[0], ys[0], xs[1], ys[1]]], "float32")   # xyxy

        keys = (self._box_key,) if self._box_key else BOX_PROMPT_KEYS
        errors: list[str] = []
        for key in keys:
            req = dict(
                type=_ADAPTER["prompt"], session_id=self.session_id,
                frame_index=int(frame_index), obj_id=int(obj_id),
            )
            req[key] = box
            try:
                resp = self.predictor.handle_request(request=req)
            except Exception as e:              # noqa: BLE001 - probing
                errors.append(f"{key}: {type(e).__name__}: {e}")
                continue
            if self._box_key != key:
                self._box_key = key
                logging.info(f"SAM 3 box prompts use keyword '{key}'")
            return resp.get("outputs")

        raise RuntimeError(
            "No accepted box-prompt keyword. Tried "
            f"{list(keys)}. Pass --box-key, or fall back to --prompt-shape point.\n"
            + "\n".join(errors)
        )

    # -- propagation ----------------------------------------------------------
    def propagate(self):
        """Yield (local_frame_index, raw_outputs) for every frame."""
        for response in self.predictor.handle_stream_request(request=dict(
            type=_ADAPTER["propagate"], session_id=self.session_id,
        )):
            yield int(response["frame_index"]), response.get("outputs")


def build_predictor(cfg: Sam3Config):
    """Import and construct the upstream predictor.  Fails loudly and usefully."""
    try:
        import torch
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as e:
        raise ImportError(
            f"SAM 3 is not installed in this environment ({e}). "
            "See requirements-sam3.txt and run `python src/sam3_preflight.py`."
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "SAM 3 video tracking needs a CUDA GPU; torch.cuda.is_available() is "
            "False. Run `python src/sam3_preflight.py` for details."
        )

    gpus = cfg.gpus if cfg.gpus is not None else list(range(torch.cuda.device_count()))
    logging.info(f"Building SAM 3 video predictor on GPUs {gpus}")
    return build_sam3_video_predictor(gpus_to_use=gpus, **cfg.extra_builder_kwargs)


# ── Orchestration ──────────────────────────────────────────────────────────────

@dataclass
class PromptSpec:
    """One thing to track: a name, and how to seed it on the first frame."""
    name: str
    obj_id: int
    box: tuple[float, float, float, float] | None = None   # (x, y, w, h)
    point: tuple[float, float] | None = None


def track_window(
    window: FrameWindow,
    cfg: Sam3Config,
    *,
    predictor=None,
    text: str | None = None,
    prompts: list[PromptSpec] | None = None,
    prompt_shape: str = "box",
    progress: bool = True,
) -> dict[int, dict[int, Observation]]:
    """Run SAM 3 over `window` and return {absolute_frame: {obj_id: Observation}}.

    Exactly one of `text` (concept prompt, objects discovered by the detector)
    or `prompts` (one seeded object per spec, identities fixed by the caller)
    must be given.
    """
    if (text is None) == (prompts is None):
        raise ValueError("Pass exactly one of text= or prompts=")
    if window.dir is None:
        raise RuntimeError("Call FrameWindow.extract() before track_window()")

    own_predictor = predictor is None
    predictor = predictor or build_predictor(cfg)
    frame_size = (window.height, window.width)

    results: dict[int, dict[int, Observation]] = {}
    try:
        with Sam3Session(predictor, window.dir, cfg) as sess:
            if text is not None:
                sess.add_text_prompt(text, frame_index=0)
                logging.info(f"Text prompt {text!r} on local frame 0")
            else:
                for spec in prompts:
                    if prompt_shape == "point" or spec.box is None:
                        pt = spec.point
                        if pt is None and spec.box is not None:
                            x, y, w, h = spec.box
                            pt = (x + w / 2, y + h / 2)
                        if pt is None:
                            raise ValueError(f"{spec.name}: no box and no point")
                        sess.add_point_prompt([pt], [1], spec.obj_id, 0, frame_size)
                    else:
                        sess.add_box_prompt(spec.box, spec.obj_id, 0, frame_size)
                    logging.info(
                        f"Seeded {spec.name} as obj_id={spec.obj_id} "
                        f"({'point' if prompt_shape == 'point' else 'box'})"
                    )

            for local_idx, raw in sess.propagate():
                abs_idx = window.to_absolute(local_idx)
                per_obj = normalize_frame_outputs(raw, frame_size, cfg)
                if cfg.mask_sink is not None:
                    for oid, obs in per_obj.items():
                        if obs.mask is not None:
                            cfg.mask_sink(abs_idx, oid, obs.mask)
                            obs.mask = None      # freed as soon as it is stored
                results[abs_idx] = per_obj
                if progress and local_idx % 30 == 0 and window.n_frames:
                    pct = 100 * (local_idx + 1) / window.n_frames
                    print(f"\r  sam3 propagating {pct:.0f}%", end="", flush=True)
        if progress:
            print("\r  sam3 propagating 100%")
    finally:
        if own_predictor and hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception as e:              # noqa: BLE001 - cleanup only
                logging.warning(f"predictor.shutdown() failed: {e}")

    return results


def _make_chunk_dir(window: FrameWindow, root: Path, lo: int, hi: int) -> Path:
    """Symlink local frames [lo, hi) into their own directory, renumbered from 0.

    SAM 3 takes a directory of frames as its resource, so a chunk is just a
    cheap alternate view of the frames already on disk — no re-encoding.
    """
    d = root / f"chunk_{lo:06d}"
    d.mkdir(parents=True, exist_ok=True)
    for k, local in enumerate(range(lo, hi)):
        src = window.dir / f"{local:06d}.jpg"
        dst = d / f"{k:06d}.jpg"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)   # filesystems without symlink support
    return d


def track_window_chunked(
    window: FrameWindow,
    cfg: Sam3Config,
    *,
    predictor=None,
    text: str | None = None,
    prompts: list[PromptSpec] | None = None,
    prompt_shape: str = "box",
) -> dict[int, dict[int, Observation]]:
    """Propagate in fixed-length chunks, re-seeding each chunk from the last one.

    Serves two purposes: it bounds the memory bank's VRAM growth on long clips,
    and it re-anchors identities periodically so a single bad frame cannot
    poison the rest of the run.  Object ids stay stable across chunks because
    every chunk after the first is seeded with explicit obj_ids taken from the
    previous chunk's final masks.
    """
    if cfg.chunk_size <= 0:
        return track_window(window, cfg, predictor=predictor, text=text,
                            prompts=prompts, prompt_shape=prompt_shape)
    if window.dir is None:
        raise RuntimeError("Call FrameWindow.extract() before track_window_chunked()")

    own_predictor = predictor is None
    predictor = predictor or build_predictor(cfg)
    # Overlap must leave the window advancing: at chunk_overlap >= chunk_size a
    # chunk would start at or before the previous one and the loop never ends.
    overlap = max(0, min(cfg.chunk_overlap, cfg.chunk_size - 1))
    if overlap != cfg.chunk_overlap:
        logging.warning(
            f"chunk_overlap {cfg.chunk_overlap} >= chunk_size {cfg.chunk_size}; "
            f"clamped to {overlap}"
        )
    root = Path(tempfile.mkdtemp(prefix="comet_sam3_chunks_"))

    merged: dict[int, dict[int, Observation]] = {}
    try:
        lo = 0
        chunk_no = 0
        carry: list[PromptSpec] | None = None
        while lo < window.n_frames:
            hi = min(window.n_frames, lo + cfg.chunk_size)
            chunk_dir = _make_chunk_dir(window, root, lo, hi)
            sub = FrameWindow(window.video, window.to_absolute(lo), None)
            sub.dir      = chunk_dir
            sub.n_frames = hi - lo
            sub.width, sub.height = window.width, window.height

            if chunk_no == 0:
                out = track_window(sub, cfg, predictor=predictor, text=text,
                                   prompts=prompts, prompt_shape=prompt_shape,
                                   progress=False)
            else:
                if not carry:
                    logging.warning(
                        f"Chunk {chunk_no}: nothing to re-seed with — "
                        f"tracking stops at frame {window.to_absolute(lo)}"
                    )
                    break
                out = track_window(sub, cfg, predictor=predictor, prompts=carry,
                                   prompt_shape="box", progress=False)

            # Later chunks win on overlapping frames: they carry fresher context.
            merged.update(out)

            # Seed the next chunk from the most recent frame that saw anything.
            carry = None
            for abs_idx in sorted(out, reverse=True):
                if out[abs_idx]:
                    carry = [
                        PromptSpec(name=str(oid), obj_id=oid, box=obs.bbox)
                        for oid, obs in sorted(out[abs_idx].items())
                    ]
                    break

            print(f"\r  sam3 propagating {100 * hi / window.n_frames:.0f}%",
                  end="", flush=True)
            nxt = hi - overlap if hi < window.n_frames else hi
            if nxt <= lo:                    # belt and braces: never stall
                raise RuntimeError(
                    f"Chunk window did not advance (lo={lo}, next={nxt}, "
                    f"chunk_size={cfg.chunk_size}, overlap={overlap})"
                )
            lo = nxt
            chunk_no += 1
        print("\r  sam3 propagating 100%")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if own_predictor and hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception as e:              # noqa: BLE001 - cleanup only
                logging.warning(f"predictor.shutdown() failed: {e}")

    return merged


# ── Zoom-crop tracking for tiny objects ────────────────────────────────────────

def track_object_zoom(
    window: FrameWindow,
    cfg: Sam3Config,
    seed_box: tuple[float, float, float, float],
    obj_id: int,
    *,
    predictor=None,
    crop_size: int = 384,
    upscale: int = 3,
    segment_len: int = 60,
) -> dict[int, Observation]:
    """Track one small object by following it through an upscaled crop.

    A 26x34 px payload is ~0.04 % of a 1080p frame; after SAM's internal
    downscaling there is very little signal left.  Cropping a `crop_size` window
    around the object and upscaling it by `upscale` puts the object back into
    the size range the model handles well.

    The crop is re-centred every `segment_len` frames from the object's own last
    position, which bounds how far it can drift towards the crop edge.  If the
    object is lost in a segment, tracking stops there rather than silently
    following whatever else is in frame.

    Returns {absolute_frame: Observation} in FULL-FRAME coordinates.
    """
    if window.dir is None:
        raise RuntimeError("Call FrameWindow.extract() before track_object_zoom()")

    own_predictor = predictor is None
    predictor = predictor or build_predictor(cfg)
    half = crop_size // 2
    out: dict[int, Observation] = {}
    root = Path(tempfile.mkdtemp(prefix="comet_sam3_zoom_"))

    # Crop config mirrors the caller's, but masks live in crop space and are
    # translated back below, so never keep them at crop resolution.
    crop_cfg = Sam3Config(
        normalize_coords=cfg.normalize_coords, box_key=cfg.box_key,
        point_mode=cfg.point_mode, mask_threshold=cfg.mask_threshold,
        min_score=cfg.min_score, min_area=1, keep_masks=False,
    )

    try:
        cx = seed_box[0] + seed_box[2] / 2
        cy = seed_box[1] + seed_box[3] / 2
        cur_box = seed_box
        lo = 0
        while lo < window.n_frames:
            hi = min(window.n_frames, lo + segment_len)

            # Clamp the crop so it stays inside the frame.
            x0 = int(max(0, min(window.width  - crop_size, cx - half)))
            y0 = int(max(0, min(window.height - crop_size, cy - half)))
            cw = min(crop_size, window.width  - x0)
            ch = min(crop_size, window.height - y0)

            seg_dir = root / f"zoom_{lo:06d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            for k, local in enumerate(range(lo, hi)):
                img = cv2.imread(str(window.dir / f"{local:06d}.jpg"))
                if img is None:
                    break
                crop = img[y0:y0 + ch, x0:x0 + cw]
                crop = cv2.resize(crop, (cw * upscale, ch * upscale),
                                  interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(seg_dir / f"{k:06d}.jpg"), crop)

            seg = FrameWindow(window.video, window.to_absolute(lo), None)
            seg.dir      = seg_dir
            seg.n_frames = hi - lo
            seg.width, seg.height = cw * upscale, ch * upscale

            # Seed box expressed in upscaled-crop coordinates.
            sb = ((cur_box[0] - x0) * upscale, (cur_box[1] - y0) * upscale,
                  max(1.0, cur_box[2] * upscale), max(1.0, cur_box[3] * upscale))
            seg_res = track_window(
                seg, crop_cfg, predictor=predictor,
                prompts=[PromptSpec(name="zoom", obj_id=obj_id, box=sb)],
                progress=False,
            )

            last_seen = None
            for abs_idx in sorted(seg_res):
                obs = seg_res[abs_idx].get(obj_id)
                if obs is None:
                    continue
                # Crop space → full-frame space.
                fx = x0 + obs.bbox[0] / upscale
                fy = y0 + obs.bbox[1] / upscale
                fw = max(1.0, obs.bbox[2] / upscale)
                fh = max(1.0, obs.bbox[3] / upscale)
                out[abs_idx] = Observation(
                    obj_id = obj_id,
                    cx     = int(round(x0 + obs.cx / upscale)),
                    cy     = int(round(y0 + obs.cy / upscale)),
                    bbox   = (int(fx), int(fy), int(round(fw)), int(round(fh))),
                    area   = max(1, int(obs.area / (upscale ** 2))),
                    score  = obs.score,
                )
                last_seen = out[abs_idx]

            if last_seen is None:
                logging.warning(
                    f"Zoom track lost the object in frames "
                    f"{window.to_absolute(lo)}→{window.to_absolute(hi - 1)}; stopping"
                )
                break
            cx, cy   = last_seen.cx, last_seen.cy
            cur_box  = last_seen.bbox
            shutil.rmtree(seg_dir, ignore_errors=True)
            lo = hi
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if own_predictor and hasattr(predictor, "shutdown"):
            try:
                predictor.shutdown()
            except Exception as e:              # noqa: BLE001 - cleanup only
                logging.warning(f"predictor.shutdown() failed: {e}")

    return out


def observations_to_trails(
    results: dict[int, dict[int, Observation]],
    id_to_name: dict[int, str],
) -> dict[str, dict[int, tuple[int, int]]]:
    """Reduce per-frame observations to the {name: {frame: (cx, cy)}} trail form."""
    trails: dict[str, dict[int, tuple[int, int]]] = {n: {} for n in id_to_name.values()}
    for abs_idx, per_obj in results.items():
        for oid, obs in per_obj.items():
            name = id_to_name.get(oid)
            if name is None:
                continue
            trails[name][abs_idx] = (obs.cx, obs.cy)
    return trails


def discover_id_names(
    results: dict[int, dict[int, Observation]],
    prefix: str = "obj",
    name_map: dict[int, str] | None = None,
) -> dict[int, str]:
    """Name every object id seen during a text-prompt run.

    Ids are ordered by first appearance so the naming is stable across reruns
    of the same clip, rather than following the detector's internal ordering.
    """
    seen: list[int] = []
    for abs_idx in sorted(results):
        for oid in sorted(results[abs_idx]):
            if oid not in seen:
                seen.append(oid)
    out = {}
    for rank, oid in enumerate(seen):
        out[oid] = (name_map or {}).get(oid, f"{prefix}{rank}")
    return out

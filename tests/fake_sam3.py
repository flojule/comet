#!/usr/bin/env python3
# tests/fake_sam3.py
"""A stand-in for Sam3VideoPredictor.

Implements the same request/response protocol as the real predictor so the
plumbing around SAM 3 — index mapping, prompt dispatch, output normalisation,
chunking, trail assembly — can be tested on a machine with no GPU.

It moves each prompted object along a straight line at a fixed velocity and
paints a filled circle for its mask, so the expected trail is exactly
predictable.  `output_layout` selects which of the response shapes SAM 3
wrappers emit, and `drop_frames` simulates dropouts.
"""
from __future__ import annotations

import numpy as np


class FakePredictor:
    def __init__(
        self,
        height: int = 120,
        width: int = 160,
        n_frames: int = 20,
        radius: int = 5,
        velocity: tuple[float, float] = (2.0, 1.0),
        output_layout: str = "arrays",     # arrays | mapping | mapping_dict | attrs
        mask_dtype: str = "bool",          # bool | logits | probs
        drop_frames: dict[int, set[int]] | None = None,   # local_idx → obj_ids
        box_key: str = "box",              # which keyword this build accepts
        mask_scale: float = 1.0,           # emit masks at a different resolution
    ):
        self.height, self.width = height, width
        self.n_frames = n_frames
        self.radius = radius
        self.velocity = velocity
        self.output_layout = output_layout
        self.mask_dtype = mask_dtype
        self.drop_frames = drop_frames or {}
        self.box_key = box_key
        self.mask_scale = mask_scale

        self.sessions: dict[str, dict] = {}
        self._next = 0
        self.shutdown_calls = 0
        self.requests: list[dict] = []

    # -- protocol -------------------------------------------------------------
    def handle_request(self, request: dict):
        self.requests.append(dict(request))
        kind = request["type"]

        if kind == "start_session":
            sid = f"sess{self._next}"
            self._next += 1
            self.sessions[sid] = {"seeds": {}, "resource": request["resource_path"]}
            return {"session_id": sid}

        if kind == "reset_session":
            self.sessions[request["session_id"]]["seeds"] = {}
            return {}

        if kind == "close_session":
            self.sessions.pop(request["session_id"], None)
            return {}

        if kind == "add_prompt":
            sess = self.sessions[request["session_id"]]
            if "text" in request:
                # Concept prompt: the "detector" finds two objects.
                sess["seeds"][0] = (self.width * 0.2, self.height * 0.2)
                sess["seeds"][1] = (self.width * 0.5, self.height * 0.3)
                return {"outputs": None}

            present = [k for k in ("box", "boxes", "bbox", "input_boxes")
                       if k in request]
            if present:
                if self.box_key not in request:
                    raise TypeError(
                        f"unexpected keyword {present[0]!r}; this build wants "
                        f"{self.box_key!r}"
                    )
                b = np.asarray(request[self.box_key]).reshape(-1)
                cx = (float(b[0]) + float(b[2])) / 2
                cy = (float(b[1]) + float(b[3])) / 2
            elif "points" in request:
                p = np.asarray(request["points"]).reshape(-1, 2)
                cx, cy = float(p[0][0]), float(p[0][1])
            else:
                raise TypeError(f"no usable prompt in {sorted(request)}")

            sess["seeds"][int(request["obj_id"])] = (cx, cy)
            return {"outputs": None}

        raise ValueError(f"FakePredictor got unknown request type {kind!r}")

    def handle_stream_request(self, request: dict):
        assert request["type"] == "propagate_in_video"
        sess = self.sessions[request["session_id"]]
        n = self._frames_in(sess["resource"])
        for i in range(n):
            yield {"frame_index": i, "outputs": self._outputs(sess["seeds"], i)}

    def shutdown(self):
        self.shutdown_calls += 1

    # -- helpers --------------------------------------------------------------
    def _frames_in(self, resource) -> int:
        from pathlib import Path
        p = Path(resource)
        if p.is_dir():
            found = len(list(p.glob("*.jpg")))
            if found:
                return found
        return self.n_frames

    def position(self, seed: tuple[float, float], i: int) -> tuple[float, float]:
        # Clamped to keep the whole circle on screen: an object drifting out of
        # frame would produce an empty mask, which reads as a dropout and would
        # make coverage assertions depend on clip length rather than on the
        # behaviour under test.  Real dropouts are injected via drop_frames.
        x = seed[0] + self.velocity[0] * i
        y = seed[1] + self.velocity[1] * i
        r = self.radius
        return (min(max(x, r), self.width - r - 1),
                min(max(y, r), self.height - r - 1))

    def _circle(self, cx: float, cy: float) -> np.ndarray:
        h = int(self.height * self.mask_scale)
        w = int(self.width * self.mask_scale)
        yy, xx = np.mgrid[0:h, 0:w]
        m = ((xx - cx * self.mask_scale) ** 2
             + (yy - cy * self.mask_scale) ** 2) <= (self.radius * self.mask_scale) ** 2
        if self.mask_dtype == "bool":
            return m
        if self.mask_dtype == "probs":
            return np.where(m, 0.9, 0.1).astype(np.float32)
        return np.where(m, 4.0, -4.0).astype(np.float32)     # logits

    def _outputs(self, seeds: dict[int, tuple[float, float]], i: int):
        dropped = self.drop_frames.get(i, set())
        ids, masks, scores = [], [], []
        for oid, seed in sorted(seeds.items()):
            if oid in dropped:
                continue
            cx, cy = self.position(seed, i)
            ids.append(oid)
            masks.append(self._circle(cx, cy))
            scores.append(0.9)

        if self.output_layout == "arrays":
            if not ids:
                return {"pred_masks": np.zeros((0, 1, 1)), "obj_ids": [],
                        "scores": []}
            # [N, 1, H, W] — the channel axis the real decoder emits.
            return {"pred_masks": np.stack(masks)[:, None],
                    "obj_ids": ids, "scores": scores}

        if self.output_layout == "mapping":
            return {oid: m for oid, m in zip(ids, masks)}

        if self.output_layout == "mapping_dict":
            return {oid: {"mask": m, "score": s}
                    for oid, m, s in zip(ids, masks, scores)}

        if self.output_layout == "attrs":
            class _Out:
                pass
            o = _Out()
            o.masks = np.stack(masks) if masks else np.zeros((0, 1, 1))
            o.object_ids = ids
            o.scores = scores
            return o

        raise ValueError(self.output_layout)

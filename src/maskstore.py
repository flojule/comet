#!/usr/bin/env python3
# maskstore.py
"""Run-length mask sidecar for mask-aware rendering.

A 680-frame 1080p run with four objects is ~5.6 GB of raw boolean masks, so
masks are stored COCO-style: column-major run lengths, alternating background
and foreground, starting with a background run (possibly of length 0).  For the
small, compact objects Comet tracks the encoding is a few dozen integers per
mask, and the whole sidecar stays in the low megabytes.

File layout — a single compressed .npz next to the tracking JSON:
    meta                     : json blob (height, width, entries index)
    rle/<frame>/<obj_name>   : int32 run lengths
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def encode_rle(mask: np.ndarray) -> np.ndarray:
    """Boolean mask → int32 run lengths (column-major, background first)."""
    flat = np.asarray(mask, dtype=bool).reshape(-1, order="F")
    if flat.size == 0:
        return np.zeros(0, dtype=np.int32)
    # Positions where the value changes, plus the implicit end boundary.
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds)
    if flat[0]:
        # Sequence starts with foreground — prepend a zero-length background run
        # so the alternation is unambiguous at decode time.
        runs = np.concatenate(([0], runs))
    return runs.astype(np.int32)


def decode_rle(runs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of `encode_rle`."""
    h, w = shape
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, run in enumerate(np.asarray(runs).tolist()):
        if i % 2 == 1 and run:            # odd runs are foreground
            flat[pos:pos + run] = True
        pos += run
    return flat.reshape((h, w), order="F")


class MaskStore:
    """Accumulate masks during tracking, then save; or load them for rendering."""

    def __init__(self, height: int, width: int):
        self.height = int(height)
        self.width  = int(width)
        self._data: dict[str, np.ndarray] = {}
        self._index: dict[str, list[int]] = {}   # name → frames present

    # -- writing --------------------------------------------------------------
    def add(self, frame_idx: int, name: str, mask: np.ndarray) -> None:
        """Store one mask.  Re-adding the same (frame, name) overwrites it —
        chunked tracking re-emits overlapping frames, and the later pass is the
        one to keep."""
        key = f"rle/{int(frame_idx)}/{name}"
        if key not in self._data:
            self._index.setdefault(name, []).append(int(frame_idx))
        self._data[key] = encode_rle(mask)

    def rename(self, mapping: dict[str, str]) -> None:
        """Re-key stored masks, e.g. from object ids to agent names.

        Text-prompt runs only learn an object's name after propagation, so
        masks are collected under the id and renamed once naming is settled.
        """
        renamed: dict[str, np.ndarray] = {}
        for key, runs in self._data.items():
            _, frame, name = key.split("/", 2)
            renamed[f"rle/{frame}/{mapping.get(name, name)}"] = runs
        self._data = renamed

        index: dict[str, list[int]] = {}
        for name, frames in self._index.items():
            index.setdefault(mapping.get(name, name), []).extend(frames)
        self._index = {n: sorted(set(f)) for n, f in index.items()}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({
            "height": self.height,
            "width":  self.width,
            "index":  {n: sorted(f) for n, f in self._index.items()},
        })
        np.savez_compressed(
            path, meta=np.array(meta), **self._data,
        )
        return path

    # -- reading --------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> MaskStore:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            store = cls(meta["height"], meta["width"])
            store._index = {n: list(f) for n, f in meta["index"].items()}
            store._data = {k: z[k] for k in z.files if k.startswith("rle/")}
        return store

    def get(self, frame_idx: int, name: str) -> np.ndarray | None:
        runs = self._data.get(f"rle/{int(frame_idx)}/{name}")
        if runs is None:
            return None
        return decode_rle(runs, (self.height, self.width))

    def names(self) -> list[str]:
        return sorted(self._index)

    def frames(self, name: str) -> list[int]:
        return list(self._index.get(name, []))

    def __len__(self) -> int:
        return len(self._data)


def sidecar_path(tracking_json: str | Path) -> Path:
    """Conventional sidecar location for a tracking JSON."""
    p = Path(tracking_json)
    return p.with_name(p.stem.removesuffix("_tracking") + "_masks.npz")

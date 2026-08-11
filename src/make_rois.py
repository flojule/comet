#!/usr/bin/env python3
# make_rois.py
"""Write rois.json from the command line — the headless twin of pick.py.

    python src/make_rois.py --video input/crazyflo.mp4 \\
        --start-frame 520 --end-frame 1200 \\
        --roi cf1=940,234,87,57 --roi cf2=1619,88,119,84 \\
        --roi cf3=1267,281,76,48 --roi payload=1278,546,26,34

Boxes are `x,y,w,h` in pixels, top-left origin, matching what pick.py writes.
`--from-json` seeds from an existing file so single values can be amended
without retyping every box.

    python src/make_rois.py --from-json src/rois.json --roi payload=1280,540,30,36
    python src/make_rois.py --show                      # print the current file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

DEFAULT_VIDEO = "input/crazyflo.mp4"
DEFAULT_OUT = "src/rois.json"


def parse_roi(spec: str) -> tuple[str, list[int]]:
    """`name=x,y,w,h` → (name, [x, y, w, h])."""
    if "=" not in spec:
        raise ValueError(f"--roi wants name=x,y,w,h, got {spec!r}")
    name, _, nums = spec.partition("=")
    name = name.strip()
    if not name:
        raise ValueError(f"--roi is missing a name: {spec!r}")
    parts = [p.strip() for p in nums.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"--roi {name}: expected 4 numbers x,y,w,h, got {len(parts)}")
    try:
        box = [int(round(float(p))) for p in parts]
    except ValueError as e:
        raise ValueError(f"--roi {name}: {e}") from e
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"--roi {name}: width and height must be positive")
    return name, box


def video_info(path: str) -> tuple[int, int, int]:
    """(total_frames, width, height); zeros when the video cannot be opened."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0, 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, w, h


def validate(rois: dict, start: int, end: int | None,
             total: int, width: int, height: int) -> list[str]:
    """Problems worth refusing to write — a bad box wastes a whole tracking run."""
    errs: list[str] = []
    if end is not None and end < start:
        errs.append(f"end_frame {end} is before start_frame {start}")
    if total and start >= total:
        errs.append(f"start_frame {start} is past the last frame ({total - 1})")
    if total and end is not None and end > total - 1:
        errs.append(f"end_frame {end} is past the last frame ({total - 1})")
    for name, (x, y, w, h) in rois.items():
        if x < 0 or y < 0:
            errs.append(f"{name}: negative origin ({x},{y})")
        if width and x + w > width:
            errs.append(f"{name}: box runs past the right edge "
                        f"({x}+{w} > {width})")
        if height and y + h > height:
            errs.append(f"{name}: box runs past the bottom edge "
                        f"({y}+{h} > {height})")
    return errs


def build(rois: dict, start: int, end: int | None) -> dict:
    return {"start_frame": int(start),
            "end_frame": None if end is None else int(end),
            "rois": {k: list(v) for k, v in rois.items()}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=DEFAULT_VIDEO)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--roi", action="append", default=[], metavar="NAME=X,Y,W,H",
                   help="one object's seed box; repeat per object")
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--from-json", default=None,
                   help="seed from an existing rois.json and amend it")
    p.add_argument("--drop", action="append", default=[], metavar="NAME",
                   help="remove an object (with --from-json)")
    p.add_argument("--show", action="store_true",
                   help="print the resulting file instead of writing it")
    p.add_argument("--force", action="store_true",
                   help="write even if boxes fall outside the frame")
    return p.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()

    rois: dict = {}
    start, end = 0, None
    src = args.from_json or (args.out if args.show and Path(args.out).exists() else None)
    if src:
        if not Path(src).exists():
            print(f"No such file: {src}", file=sys.stderr)
            return 2
        seed = json.loads(Path(src).read_text())
        rois = dict(seed.get("rois", seed if "rois" not in seed else {}))
        start = seed.get("start_frame", 0)
        end = seed.get("end_frame")

    for spec in args.roi:
        try:
            name, box = parse_roi(spec)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        rois[name] = box

    for name in args.drop:
        rois.pop(name, None)

    if args.start_frame is not None:
        start = args.start_frame
    if args.end_frame is not None:
        end = args.end_frame

    if not rois:
        print("No ROIs. Pass --roi NAME=X,Y,W,H (repeatable), or --from-json.",
              file=sys.stderr)
        return 2

    total, width, height = video_info(args.video)
    if total:
        if end is None:
            end = total - 1
        errs = validate(rois, start, end, total, width, height)
        if errs:
            for e in errs:
                print(f"{'warning' if args.force else 'error'}: {e}",
                      file=sys.stderr)
            if not args.force:
                print("Nothing written. Re-check the boxes, or pass --force.",
                      file=sys.stderr)
                return 1
    else:
        print(f"warning: cannot open {args.video}; boxes not checked against it",
              file=sys.stderr)

    data = build(rois, start, end)
    if args.show:
        print(json.dumps(data, indent=2))
        return 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, indent=2))
    span = f"{start}→{end}" if end is not None else f"{start}→end"
    print(f"Wrote {args.out}: {len(rois)} object(s) ({', '.join(rois)}), "
          f"frames {span}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

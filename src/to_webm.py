"""to_webm.py — convert MP4 files to WebM (VP9).

Usage:
    python src/to_webm.py                        # scans output/
    python src/to_webm.py path/to/dir            # a custom directory
    python src/to_webm.py a.mp4 b.mp4            # named files
    python src/to_webm.py --bitrate 4M --out-dir web/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

INPUT_DIR  = Path("output")
BITRATE    = "2M"


def convert(src: Path, dst: Path, bitrate: str = BITRATE,
            timeout: int = 600) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-c:v", "libvpx-vp9", "-b:v", bitrate, str(dst)],
        capture_output=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src.name}: {r.stderr[:300].decode(errors='replace')}")


def collect(paths: list[str]) -> tuple[list[Path], Path]:
    """Resolve arguments into (files, default output dir)."""
    if not paths:
        return sorted(INPUT_DIR.glob("*.mp4")), INPUT_DIR / "webm"
    if len(paths) == 1 and Path(paths[0]).is_dir():
        d = Path(paths[0])
        return sorted(d.glob("*.mp4")), d / "webm"
    files = [Path(p) for p in paths]
    missing = [f for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"No such file(s): {', '.join(map(str, missing))}")
    return files, files[0].parent / "webm"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*",
                   help="MP4 files, or a directory to scan (default: output/)")
    p.add_argument("--bitrate", default=BITRATE)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    mp4s, default_out = collect(args.paths)
    if not mp4s:
        print("No MP4 files found")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in mp4s:
        dst = out_dir / src.with_suffix(".webm").name
        print(f"  {src.name} → {dst.name} …", end=" ", flush=True)
        convert(src, dst, args.bitrate)
        print("done")

    print(f"\nConverted {len(mp4s)} file(s) → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

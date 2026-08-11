# Comet

Track and segment things in video, and render what you find.

Two jobs:

- **Motion trails** — follow moving objects (drones, vehicles, animals) and draw their paths.
- **Static structure** — segment stairs, doorways or ramps from robot recordings and measure their orientation.

Reads plain video, photos, and ROS 2 mcap bags. Runs from the command line or a small desktop GUI.

[debug video](https://github.com/user-attachments/assets/a0e2a55d-9e8b-4190-9f15-e77fb74a519f) ·
[transient trails](https://github.com/user-attachments/assets/e33fe741-0fd7-465e-98d2-c9d5fde95972) ·
[persistent trails](https://github.com/user-attachments/assets/be91b6bf-33b3-4584-b761-8da9fd4e519d)

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements-mcap.txt    # read ROS 2 mcap bags
pip install -r requirements-sam3.txt    # SAM 3 tracking (CUDA GPU)
git clone https://github.com/facebookresearch/sam3 && pip install -e sam3
python src/sam3_preflight.py            # check the setup
```

SAM 3 weights are found at `--checkpoint`, then `$COMET_SAM3_CHECKPOINT`, then
`~/ws/models/weights/sam3.pt`.

---

## Motion trails

```
video ──▶ pick objects ──▶ track ──▶ render
```

**1. Pick objects and frame range.** Interactively:

```bash
python src/pick.py
```

Arrow keys or `a`/`d` to scrub, Space/Enter to confirm each step: start frame,
end frame, then a box around each object. Objects are named in the `AGENTS`
list at the top of the file.

Or from the command line:

```bash
python src/make_rois.py --video input/crazyflo.mp4 \
    --start-frame 520 --end-frame 1200 \
    --roi cf1=940,234,87,57 --roi payload=1278,546,26,34
```

`--from-json` amends an existing file; `--drop NAME` removes an object;
`--show` prints without writing.

**2. Track.** Background subtraction with Hungarian assignment:

```bash
python src/track.py                 # add --no-display to run headless
```

Builds a background model from pre-motion frames, then follows every object
frame by frame, filling gaps with a bidirectional corridor search. Writes
`*_debug.mp4` and `*_tracking.json`.

Or with SAM 3, which needs no background model and so tolerates a moving
camera:

```bash
python src/track_sam3.py --from-rois          # keeps your object names
python src/track_sam3.py --text "drone"       # finds them from a description
```

Useful flags: `--zoom-agents payload` re-tracks a small object through an
upscaled crop; `--borrow-agents payload --borrow-from <json>` takes one object
from another tracker's output; `--reprompt-every N` re-anchors on long clips;
`--save-masks` writes a mask sidecar.

**3. Render.**

```bash
python src/render.py output/crazyflo_path_tracking.json
```

| Output | |
|---|---|
| `*_persistent.mp4` | trails accumulate from start to end |
| `*_transient.mp4` | shooting-star: bright at the head, fading along the tail |

Appearance is settable per run: `--thickness`, `--alpha`, `--trail-window`,
`--color cf1=255,0,0`, `--smooth` / `--no-smooth`. With a mask sidecar,
`--mask-mode occlude` draws trails behind objects and `--mask-mode glow` adds a
silhouette halo.

**4. WebM** (optional):

```bash
python src/to_webm.py                            # scans output/
python src/to_webm.py clip.mp4 --bitrate 4M
```

---

## Static structure

For things that stay put while the camera moves. Renders a per-frame
segmentation overlay and measures the subject's angle.

```
mcap bags / video / photos ──▶ SAM 3 ──▶ masks ──┬──▶ overlay video
                                                 └──▶ orientation JSON
```

**GUI:**

```bash
python src/gui.py
```

Pick a recording, press **Scan bag**, choose a camera topic, type what to look
for, press **Run**. (`sudo apt install python3-tk` if Tk is missing.)

**Command line:**

```bash
python src/stairs_pipeline.py --list /path/to/bags       # what cameras are in there
python src/stairs_pipeline.py /path/to/bags \
    --topic /camera/color/image_raw --prompt stairs --out output/stairs
```

The same command takes a video, a photo, or a folder of photos — those need no
topic:

```bash
python src/stairs_pipeline.py clip.mp4 --prompt stairs
python src/stairs_pipeline.py ./photos/ --prompt "door"
```

| Output | |
|---|---|
| `*_overlay.mp4` | masks tinted and outlined, angle drawn and annotated |
| `*_stairs.json` | per-frame angles, coverage, provenance, bag timestamps |
| `*_masks.npz` | run-length mask sidecar |
| `*_frames.mp4` | decoded frames, so the analysis can re-run on its own |

### Running the model elsewhere

Only SAM 3 needs a GPU, and the pipeline splits at that line:

```bash
# on the GPU machine
python src/stairs_pipeline.py /path/to/bags --topic /camera/color/image_raw \
    --prompt stairs --out output/stairs --stage segment

# copy the three files across, then anywhere:
python src/stairs_pipeline.py --stage analyse --out output/stairs
```

The analyse stage reads only stage 1's output, so retuning orientation costs
seconds rather than another pass of the model.

### Orientation

`angle_deg` is the dominant edge direction in the image plane, in degrees,
wrapped to `[0, 180)`. It comes from Hough segments inside the mask,
length-weighted into an angle histogram. Each frame also reports a
`confidence`: the share of edge length pointing the dominant way.

### mcap bags

Bags decode with no ROS installation. A recording split across `_0.mcap`,
`_1.mcap`, … reads as one stream ordered by message timestamp.
`sensor_msgs/msg/Image` and `CompressedImage` are both handled, across the
encodings RealSense emits (`rgb8`, `bgr8`, `mono8`, `mono16`, YUV, Bayer).
Frame rate is measured from the message timestamps.

---

## Layout

| | |
|---|---|
| `pick.py` / `make_rois.py` | choose objects and frame range, interactively or by flag |
| `track.py` | background-subtraction tracker |
| `track_sam3.py` | SAM 3 tracker |
| `sam3_backend.py` | SAM 3 sessions, frame windowing, masks |
| `sam3_preflight.py` | environment check |
| `stairs_pipeline.py` | recordings → masks → overlay + orientation |
| `stair_orientation.py` | dominant edge angle from a mask |
| `mcap_source.py` | ROS 2 bag reading |
| `media.py` | video / photos / bags → one video |
| `trails.py` | smoothing, gap fill, tracking-JSON format |
| `maskstore.py` | run-length mask sidecar |
| `render.py` | trail videos |
| `to_webm.py` | MP4 → WebM |
| `gui.py` | desktop front end |

## Tests

```bash
python -m unittest discover -s tests
```

204 tests, no GPU required. `python tests/make_bag.py /tmp/fakebag --angle 25`
writes a synthetic RealSense recording to try the pipeline against.

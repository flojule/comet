# Comet

Motion-trail visualisation tool for tracking multiple objects in video — including well-defined objects (e.g. drones, vehicles, animals) and harder-to-see ones (e.g. a small payload). Uses background subtraction + Hungarian assignment to maintain object identities across frames, then renders clean trail videos.

It also does **segmentation of static structure** — stairs, doorways, ramps — from robot recordings: see [Segmenting static structure](#segmenting-static-structure-stairs-doorways-ramps), which reads ROS 2 mcap bags directly and has a small desktop GUI.

[debug video](https://github.com/user-attachments/assets/a0e2a55d-9e8b-4190-9f15-e77fb74a519f)

[transient trails](https://github.com/user-attachments/assets/e33fe741-0fd7-465e-98d2-c9d5fde95972)

[persistent trails](https://github.com/user-attachments/assets/be91b6bf-33b3-4584-b761-8da9fd4e519d)

## Workflow

Two interchangeable trackers write the same `*_tracking.json`, so everything
downstream is shared:

```
input video
    │
    ▼
pick.py          ← interactively select objects and frame range
    │ rois.json
    ├─────────────────────────┐
    ▼                         ▼
track.py                  track_sam3.py    ← SAM 3 (GPU); text prompts optional
background subtraction    promptable segmentation + tracking
    │                         │
    └──────────┬──────────────┘
               │ *_tracking.json + *_debug.mp4  (+ *_masks.npz from SAM 3)
               ▼
        render.py        ← render persistent and transient trail videos
               │ *_persistent.mp4 + *_transient.mp4
               ▼
        to_webm.py       ← (optional) batch-convert output/*.mp4 → output/webm/
```

Which tracker to use:

| | `track.py` | `track_sam3.py` |
|---|---|---|
| Hardware | any laptop | CUDA GPU |
| Camera | must be static | may move |
| Setup | none | SAM 3 install + gated checkpoint |
| Selecting objects | boxes in `rois.json` | boxes **or** a text prompt |
| Output | centroids | centroids + masks |
| Tiny/low-contrast objects | tuned sensitive mask | may need `--zoom-agents` |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For the SAM 3 backend only (see [SAM 3 tracking](#sam-3-tracking-track_sam3py)):

```bash
pip install -r requirements-sam3.txt
git clone https://github.com/facebookresearch/sam3 && pip install -e sam3
python src/sam3_preflight.py     # verifies python/torch/CUDA/sam3/weights
```

**Weights.** `track_sam3.py` looks for a local checkpoint in this order:

1. `--checkpoint /path/to/sam3.pt`
2. `$COMET_SAM3_CHECKPOINT`
3. `~/ws/models/weights/sam3.pt`, `~/ws/models/weights/sam3.1.pt`,
   `models/weights/sam3.pt`

A local file is passed straight to the builder as `checkpoint_path`, which
skips its `load_from_HF` default — so with weights on disk you need no Hugging
Face account at all. Only if nothing is found does SAM 3 fall back to
downloading the gated `facebook/sam3` checkpoint, which requires
[requesting access](https://huggingface.co/facebook/sam3) and
`huggingface-cli login`. A `--checkpoint` or `$COMET_SAM3_CHECKPOINT` that
points at a missing file is an error, not a silent fallback to the download.

## Usage

**1. Pick objects and frame range** — `src/pick.py`

```bash
python src/pick.py
```

Opens an interactive window. Use arrow keys / `a` / `d` to scrub the video, then press Space/Enter to confirm each step:

1. **Start frame** — first frame of interest (objects should be in their starting positions)
2. **End frame** — last frame of interest
3. **Bounding boxes** — draw a box around each object to track (one at a time)

Objects are defined in the `AGENTS` list at the top of `pick.py`. Each agent can represent any moving object — label them to suit your footage. The tool handles two distinct detection modes under the hood:

- **High-contrast objects** (well-lit, distinct from background) — tracked with a standard foreground mask
- **Low-contrast objects** (small, dim, or occluded) — additionally detected with a more sensitive mask restricted to the object's expected spatial band, so they don't get lost when the standard threshold misses them

Saves `src/rois.json`.

**2. Track** — `src/track.py`

```bash
python src/track.py
```

Reads `input/` video + `src/rois.json`. Builds a background model from pre-motion frames, then tracks all objects frame-by-frame. Gap periods (when an object is temporarily undetected) are filled using a bidirectional corridor search.

Writes to `output/`:
- `*_debug.mp4` — annotated video showing bounding boxes, labels, and trails
- `*_tracking.json` — full per-frame coordinate log for all objects

Pass `--no-display` to run without a GUI window (headless machines, remote boxes).
`--video`, `--rois` and `--out` override the paths at the top of the file.

### SAM 3 tracking (`track_sam3.py`)

A drop-in alternative to step 2 that uses Meta's [SAM 3](https://github.com/facebookresearch/sam3)
instead of background subtraction. It needs no background model, so it tolerates
a moving camera and changing light, and it returns masks rather than blobs — so
overlapping objects keep their own centroids.

```bash
python src/sam3_preflight.py                          # check the environment first
python src/track_sam3.py --out output/crazyflo_sam3.mp4
python src/render.py output/crazyflo_sam3_tracking.json
```

Because both trackers write the same JSON, you can render each and compare them
on the same clip.

**Choosing what to track.** Either seed from the boxes you already picked, which
keeps your `cf1`/`cf2`/`cf3`/`payload` names:

```bash
python src/track_sam3.py --from-rois                   # default
```

…or describe the objects and let the detector find them. Discovered objects are
named by first appearance (`obj0`, `obj1`, …) and can be renamed:

```bash
python src/track_sam3.py --text "drone" --name 0=cf1 --name 1=cf2
```

**Key options**

| Option | Purpose |
|---|---|
| `--agents cf1,cf2` | track a subset of `rois.json` |
| `--prompt-shape point` | seed with the ROI's centre point instead of its box |
| `--point-mode bbox` | use bbox centres instead of mask barycentres |
| `--reprompt-every N` | re-seed every N frames; also caps memory-bank VRAM on long clips |
| `--min-score` / `--min-area` | drop weak or implausibly small detections |
| `--zoom-agents payload` | re-track an agent through an upscaled crop (see below) |
| `--borrow-agents payload --borrow-from <json>` | take that agent from another tracker's output |
| `--save-masks` | write a `*_masks.npz` sidecar for mask-aware rendering |
| `--no-gap-fill` | leave dropouts as holes instead of interpolating |
| `--checkpoint PATH` | weights file (see [Setup](#setup) for the search order) |
| `--gpus 0,1` | restrict which CUDA devices are used |

**Quality reporting.** Every run logs per-object coverage, longest dropout,
score range and mask area, and stores them under `sam3_stats` in the tracking
JSON. Coverage is the number to watch: gap fill will happily draw a straight
line through a stretch where the object was never actually found, and the trail
video alone will not show you that.

**Small, low-contrast objects.** A 26×34 px payload is ~0.04 % of a 1080p frame,
which is where SAM 3 is weakest. Two fallbacks, in order of preference:

```bash
# 1. re-track it through a crop that is upscaled before it reaches the model
python src/track_sam3.py --zoom-agents payload --zoom-crop 384 --zoom-upscale 3

# 2. hybrid: SAM 3 for the drones, background subtraction for the payload
python src/track.py --no-display                       # produces the donor JSON
python src/track_sam3.py --borrow-agents payload \
    --borrow-from output/crazyflo_path_tracking.json
```

**Tests.** The plumbing around the model — frame-index mapping, prompt dispatch,
output normalisation, chunking, trail assembly, mask storage — is covered by a
fake predictor and needs no GPU:

```bash
python -m unittest discover -s tests
```

**3. Render trails** — `src/render.py`

```bash
python src/render.py
```

Reads the tracking JSON and produces two clean videos:

| Output | Description |
|---|---|
| `*_persistent.mp4` | Trails accumulate from start to end |
| `*_transient.mp4` | Shooting-star style: thick at the current position, tapering and fading over ~¼ orbit |

Trail appearance (color, thickness, alpha, window length) can be overridden via the `OVERRIDES` dict at the top of the file without re-running tracking.

If a `*_masks.npz` sidecar exists (written by `track_sam3.py --save-masks`), two
extra render modes become available:

```bash
python src/render.py output/crazyflo_sam3_tracking.json --mask-mode occlude
python src/render.py output/crazyflo_sam3_tracking.json --mask-mode glow
```

| Mode | Effect |
|---|---|
| `off` | default — trails drawn straight over the frame |
| `occlude` | trails pass *behind* the objects |
| `glow` | `occlude`, plus a coloured halo around each silhouette |

Without a sidecar these fall back to `off`.

**4. Convert to WebM** — `src/to_webm.py` *(optional)*

```bash
python src/to_webm.py
```

Batch-converts all `output/*.mp4` → `output/webm/*.webm` using VP9. Useful for web embedding.

## Segmenting static structure (stairs, doorways, ramps)

Trails are for things that move. For structure that stays put while the camera
moves, a trail would just encode your own egomotion — so this path renders a
**per-frame segmentation overlay** and measures the subject's **orientation**
instead.

```
mcap bags / video / photos ──▶ SAM 3 (text prompt) ──▶ masks ──┬──▶ overlay video
                                                               └──▶ orientation JSON
```

### GUI

```bash
python src/gui.py
```

Pick a recording folder, press **Scan bag**, choose the camera topic, type what
to look for (`stairs`), press **Run**. Needs Tk — `sudo apt install python3-tk`
if the app reports it missing. Plain videos, single photos and folders of
photos work too; those need no topic.

### Command line

```bash
pip install -r requirements-mcap.txt

# what cameras are in the bag?
python src/stairs_pipeline.py --list /path/to/bags

# run it
python src/stairs_pipeline.py /path/to/bags \
    --topic /camera/color/image_raw --prompt stairs --out output/stairs
```

Works the same on other inputs:

```bash
python src/stairs_pipeline.py clip.mp4 --prompt stairs
python src/stairs_pipeline.py photo.jpg --prompt stairs
python src/stairs_pipeline.py ./photos/ --prompt stairs
```

### Running the model on a different machine

Only SAM 3 needs a GPU. The pipeline splits at that boundary, so the model can
run on the CUDA box while everything else runs wherever you like:

```bash
# on the GPU machine
python src/stairs_pipeline.py /path/to/bags \
    --topic /camera/color/image_raw --prompt stairs \
    --out output/stairs --stage segment

# copy output/stairs_{stairs.json,masks.npz,frames.mp4} across, then anywhere:
python src/stairs_pipeline.py --stage analyse --out output/stairs
```

Those three files are a complete handoff — the analyse stage never touches the
bags, the model or CUDA. The masks are run-length encoded, so the bulk is just
the frames video.

This is worth using on a single machine too: retuning orientation costs seconds
in the analyse stage instead of a full pass of the model.

**Outputs** (`--out` stem):

| File | Contents |
|---|---|
| `*_overlay.mp4` | frames with masks tinted and outlined, angle drawn and annotated |
| `*_stairs.json` | per-frame angles, coverage, provenance, mcap timestamps |
| `*_masks.npz` | RLE mask sidecar |
| `*_frames.mp4` | the decoded frames, so the analyse stage can re-run standalone |

### Hardware

SAM 3 needs **CUDA**. The upstream package will not run on Apple Silicon — it
calls `.cuda()` unconditionally, gates on `device.type == "cuda"`, and needs
Triton kernels plus the `torch_generic_nms` CUDA extension
([sam3#164](https://github.com/facebookresearch/sam3/issues/164)). `sam3_preflight.py`
detects this and says so rather than reporting a missing GPU.

Everything else — bag reading, orientation, overlay, the GUI, the test suite —
needs no GPU and runs anywhere.

### Reading mcap bags

rosbag2 embeds message schemas in the file, so bags decode with **no ROS
installation**. A recording split across `_0.mcap`, `_1.mcap`, … is read as one
continuous stream, ordered by message timestamp rather than filename (plain
sorting puts `_10` before `_9`). `sensor_msgs/msg/Image` and `CompressedImage`
are both handled, across the encodings RealSense emits (`rgb8`, `bgr8`,
`mono8`, `mono16`, YUV, Bayer).

To try it without real data:

```bash
python tests/make_bag.py /tmp/fakebag --splits 3 --frames 30 --angle 25
```

### What the orientation number means

`angle_deg` is the dominant edge direction **in the image plane**, in degrees,
wrapped to `[0, 180)` because a line has no direction. It comes from Hough
segments found inside the mask, length-weighted into an angle histogram.

It is **not** the stairs' 3D orientation relative to the robot — that needs
camera intrinsics plus either the nosing lines' vanishing point or the depth
stream. The image-plane angle is directly useful for alignment ("is the robot
square to the flight?") and is the input to that later 3D step.

Every frame also carries a `confidence`: the share of detected edge length
pointing the dominant way. A clean staircase reads ~1.0; competing directions
(railings, floor tiles) pull it down, and frames with no consistent direction
report `null` rather than a guess.

**Visual odometry is not implemented.** Monocular VO needs intrinsics, feature
tracking, essential-matrix decomposition, and still leaves scale ambiguous —
it's a subsystem, not a flag. See the note at the end of this section if you
want the cheap approximation.

## Source layout

| File | Role |
|---|---|
| `pick.py` | interactive ROI + frame-range picker → `rois.json` |
| `track.py` | background-subtraction tracker |
| `track_sam3.py` | SAM 3 tracker (CLI, both prompting modes) |
| `sam3_backend.py` | SAM 3 session wrapper: frame windowing, prompts, masks |
| `sam3_preflight.py` | environment check for the SAM 3 backend |
| `trails.py` | shared smoothing, gap fill, palette and tracking-JSON contract |
| `maskstore.py` | run-length mask sidecar |
| `render.py` | trail videos |
| `to_webm.py` | MP4 → WebM |
| `gui.py` | desktop front end for the static-structure path |
| `stairs_pipeline.py` | mcap/video/photos → masks → overlay + orientation |
| `mcap_source.py` | ROS 2 mcap reading, no ROS install required |
| `media.py` | normalises video / photo / photo folder / bags into one video |
| `stair_orientation.py` | dominant edge angle from a mask |

## Configuration

Key constants in `src/track.py`:

| Constant | Default | Description |
|---|---|---|
| `VIDEO_IN` | `input/crazyflo.mp4` | Input video path |
| `VIDEO_OUT` | `output/crazyflo_path.mp4` | Output path stem |
| `AGENTS` | `["cf1","cf2","cf3","payload"]` | Object names (edit in `pick.py` too) |
| `TRAIL_COLOR` | per-agent BGR | Colors for each object |
| `MAX_ASSIGN_DIST` | `260` | Max px distance for Hungarian assignment |
| `AGENT_Y_CLAMP` | `{"payload": 120}` | Hard vertical gate for low-contrast objects |
| `CLAMP_WHEN_LOST` | `{"payload"}` | Agents that fall back to nearest blob when unmatched |

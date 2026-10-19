"""Generates performance_summary.md, README.md and the submission ZIP from MEASURED values."""
import platform
import shutil
from pathlib import Path


def hardware_string():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
    except Exception:
        pass
    return f"CPU only ({platform.processor() or platform.machine()})"


def write_performance_summary(cfg, info, metrics, validations, path):
    m, i = metrics, info
    rows = "\n".join(f"| {k} | {v['frames_readable']} | {v['width']}x{v['height']} | {v['fps']:.2f} | "
                     f"{v['size_mb']:.1f} | {'PASS' if v['passed'] else 'FAIL'} |" for k, v in validations.items())
    ts = m.get("track_stats") or {}
    text = f"""# Performance Summary

**Model:** `{m['model']}` (pretrained COCO weights, no training)  
**Tracker:** `{m['tracker']}` via Ultralytics `model.track(persist=True)`  
**Confidence threshold:** {m['conf']} | **Classes:** person (0), sports ball (32) | **Inference size:** {m['imgsz']}  
**Hardware:** {hardware_string()} (torch device `{m['device']}`)

## Source video
- Resolution: {i['width']}x{i['height']}
- Duration: {i['duration_s']:.1f} s
- Frame rate: {i['fps']:.2f} FPS
- Frames: {i['frame_count']}

## Measured speed (Tasks run: {m['tasks_run']})
| Measurement | Value |
|---|---|
| Frames processed | {m['frames_processed']} / {m['frames_expected']} (no frame skipping: {m['all_frames_processed']}) |
| Total loop time (read + detect + track + analytics + drawing + frame writing) | {m['loop_time_s']:.2f} s |
| **End-to-end pipeline FPS** | **{m['pipeline_fps']:.2f}** |
| Detector + tracker call only (`model.track`) | {m['detector_tracker_time_s']:.2f} s -> {m['detector_tracker_fps']:.2f} FPS |
| Final H.264 re-encode time (not in FPS above) | {m['encode_time_s']:.2f} s |

*Pipeline FPS is the whole processing loop, not model-only inference FPS. It is also different from the source
video's playback FPS ({i['fps']:.2f}), which only controls how fast the saved video plays.*

## Output validation (every frame read back)
| Output | Readable frames | Resolution | FPS | MB | Result |
|---|---|---|---|---|---|
{rows}

## Tracking / analytics observations (automatic, no ground truth)
- Unique person IDs created: {ts.get('unique_person_ids', 'n/a')}
- Short-lived IDs (< 25 frames, likely fragments): {ts.get([k for k in ts if k.startswith('short_lived')][0], 'n/a') if ts else 'n/a'}
- Suspected ID switches (heuristic only): {ts.get('suspected_id_switches_heuristic', 'n/a')}
- Mean / max players per frame: {ts.get('mean_players_per_frame', 0):.1f} / {ts.get('max_players_in_frame', 'n/a')}
- Frames with a reliable ball: {m['reliable_ball_frames']} of {m['frames_processed']}
- CoA mode counts: {m['coa_mode_counts']}
- Frames without any CoA: {m['frames_without_coa']}

## Limitations observed
See the "Known limitations" section of README.md (filled in after visually reviewing the output videos).
"""
    Path(path).write_text(text)
    return text


def write_readme(cfg, info, metrics, path, observed_notes=""):
    m = metrics
    vc = m.get("virtual_camera", {})
    text = f"""# Sports Video Analytics & Automated Framing (Skandha Media Services Assessment)

**Objective:** detect and track players (and the ball when visible) in a wide-angle sports clip, compute
spatial analytics (footpoints, motion trails, Center of Action) and simulate a smooth virtual camera that
turns the 16:9 feed into a 9:16 vertical reel. Only pretrained models are used - nothing is trained.

## Requirements & installation
Python 3.9+. In Google Colab, PyTorch and OpenCV are preinstalled; the notebook runs `pip install ultralytics lapx`.
Locally: `pip install -r requirements.txt` (ffmpeg is optional but recommended, it re-encodes outputs to H.264).

## How to run
**Colab (recommended):** open `Skandha_Sports_AI_Assignment.ipynb` -> Runtime -> Change runtime type -> T4 GPU ->
run cells top to bottom. Upload your clip in the upload cell, or mount Google Drive and set `DRIVE_VIDEO_PATH`.
**Command line:** `python main.py --video path/to/clip.mp4 --tasks 1 2 3`

## Pipeline (in order)
1. Read the video frame by frame with OpenCV (no frames skipped).
2. `model.track(persist=True)` -> YOLO detection + {m['tracker']} (class filter: person=0, sports ball=32, conf >= {m['conf']}).
3. Draw boxes, class labels, persistent Track IDs -> `task1_tracking.mp4`.
4. Footpoints, per-ID 25-frame trails, ball reliability test, Center of Action, HUD -> `task2_analytics.mp4`.
5. EMA-smoothed, boundary-clamped virtual camera -> `final_virtual_camera.mp4` and `task3_crop_overlay.mp4`.
6. Every output is re-opened and all frames are read back for validation; `performance_summary.md` is generated.

## Model, tracker, thresholds
Model `{m['model']}`, tracker `{m['tracker']}`, confidence {m['conf']}, inference size {m['imgsz']}. Classes: 0 = person, 32 = sports ball.
Each run creates a new model/tracker object so tracker IDs and trail histories start clean.
*Badminton/tennis note:* a shuttlecock is not the COCO "sports ball" class and a tennis ball is tiny, so ball
detection may fail; the pipeline then falls back to player positions.

## Footpoints and motion trails
Footpoint = bottom-centre of the person box: `x_foot=(x1+x2)/2`, `y_foot=y2`. Feet touch the ground, so this point
represents field position better than the box centre, which shifts when a player jumps or raises arms.
Each Track ID owns a `collections.deque(maxlen={cfg.trail_len})` of `(frame, x, y)`, so trails of different
players are never connected. Trails are drawn as polylines with thickness increasing towards the newest point; the line is
broken if an ID vanished for more than {cfg.trail_max_gap} frames. IDs unseen for {cfg.trail_prune_after} frames are deleted.

## Center of Action (CoA)
Player reference `P = median(footpoints of current-frame persons)` (median resists outliers such as a far-away substitute).
- Ball reliable and players present: `CoA = (w_ball*Ball + w_players*P) / (w_ball + w_players)` with `w_ball={cfg.w_ball}`, `w_players={cfg.w_players}` (ball gets {cfg.w_ball/(cfg.w_ball+cfg.w_players):.0%} of the weight).
- Ball reliable, no players: ball position only.
- No reliable ball: `CoA = P`.
- Nothing valid: the last CoA is held for up to {cfg.coa_hold_frames} frames, then no marker is drawn (never invalid coordinates).

**Ball reliability** (all must hold): confidence >= {cfg.ball_conf_reliable}; box area < {cfg.ball_max_area_frac:.0%} of the frame;
within {cfg.ball_max_dist_frac:.0%} of the frame diagonal from the player median; no impossible jump (> {cfg.ball_max_jump_frac:.0%} of frame width per frame, scaled by the gap) from the last accepted ball.

## HUD
Rolling processing FPS (last {cfg.fps_window} frames of the real loop), active player count (person boxes in the *current* frame; the ball is never counted),
CoA mode, and the frame counter. The CoA is drawn as a magenta target/crosshair.
Processing FPS is how fast the code runs; the source FPS only sets playback speed of the saved video - they are independent.

## Virtual camera (Task 3)
Mode `{cfg.camera_mode}`: crop {vc.get('crop_size')} -> output {vc.get('output_size')}. The crop keeps the full frame height and is centred horizontally on the CoA.
**EMA:** `x_smooth[t] = alpha*x_raw[t] + (1-alpha)*x_smooth[t-1]`, alpha = {cfg.ema_alpha} (recommended 0.05-0.15; lower = smoother but laggier).
Initialised with the first valid CoA. With no valid CoA the camera holds its last position (frame centre before the first valid CoA).
**Clamping:** the target centre is clipped to `[crop_w/2, W - crop_w/2]` before smoothing and the final rectangle is clipped to the frame, so the crop never leaves the video and the EMA never "winds up" at an edge.
**Resizing:** the crop keeps an exact 9:16 ratio, so resizing only scales it (no distortion). Upscaling to a larger output (e.g. 720x1280) adds pixels but no real detail.

## Output files
| File | Content |
|---|---|
| `outputs/task1_tracking.mp4` | boxes, class labels, confidence, Track IDs |
| `outputs/task2_analytics.mp4` | boxes, IDs, footpoints, 25-frame trails, CoA marker, HUD |
| `outputs/task3_crop_overlay.mp4` | Task 2 view + stabilized crop rectangle |
| `outputs/final_virtual_camera.mp4` | the cropped, stabilized virtual-camera feed |
| `outputs/frame_log.csv`, `metrics.json` | per-frame players / ball / CoA / crop log and run metrics |

Videos over 25 MB: upload to Google Drive/YouTube and paste the link here: `<ADD LINK>`

## Results
See `performance_summary.md` (measured on: {hardware_string()}).

## Known limitations & future improvements
{observed_notes or "<Fill in after watching the videos - e.g. ID switches during occlusions, fast pans, missed ball, referees/crowd detected as players.>"}

General limitations of this design:
- Tracker IDs are not perfectly stable: crossing/occluded players can swap or get new IDs. The report lists a *heuristic* count of suspected switches, not ground-truth accuracy.
- Referees, ball boys and spectators are detected as `person` (COCO has no "player" class); a pitch mask or team-colour filter would fix this.
- The small, fast ball is often missed or blurred; a ball-specific detector would help.
- Footpoints are in image pixels; a homography to field coordinates would remove perspective effects.
- BoT-SORT with camera-motion compensation / ReID would be a better choice for fast camera pans.
- The EMA lags behind fast action by design; the clip's first frames may jump if no CoA exists until later.
"""
    Path(path).write_text(text)
    return text


def make_zip(project_dir, zip_base, exclude_raw=True):
    project_dir = Path(project_dir)
    return shutil.make_archive(str(zip_base), "zip", root_dir=project_dir.parent, base_dir=project_dir.name)

# Sports Video Analytics & Automated Framing (Skandha Media Services Assessment)

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
2. `model.track(persist=True)` -> YOLO detection + bytetrack.yaml (class filter: person=0, sports ball=32, conf >= 0.35).
3. Draw boxes, class labels, persistent Track IDs -> `task1_tracking.mp4`.
4. Footpoints, per-ID 25-frame trails, ball reliability test, Center of Action, HUD -> `task2_analytics.mp4`.
5. EMA-smoothed, boundary-clamped virtual camera -> `final_virtual_camera.mp4` and `task3_crop_overlay.mp4`.
6. Every output is re-opened and all frames are read back for validation; `performance_summary.md` is generated.

## Model, tracker, thresholds
Model `yolov8s.pt`, tracker `bytetrack.yaml`, confidence 0.35, inference size 960. Classes: 0 = person, 32 = sports ball.
Each run creates a new model/tracker object so tracker IDs and trail histories start clean.
*Badminton/tennis note:* a shuttlecock is not the COCO "sports ball" class and a tennis ball is tiny, so ball
detection may fail; the pipeline then falls back to player positions.

## Footpoints and motion trails
Footpoint = bottom-centre of the person box: `x_foot=(x1+x2)/2`, `y_foot=y2`. Feet touch the ground, so this point
represents field position better than the box centre, which shifts when a player jumps or raises arms.
Each Track ID owns a `collections.deque(maxlen=25)` of `(frame, x, y)`, so trails of different
players are never connected. Trails are drawn as polylines with thickness increasing towards the newest point; the line is
broken if an ID vanished for more than 5 frames. IDs unseen for 150 frames are deleted.

## Center of Action (CoA)
Player reference `P = median(footpoints of current-frame persons)` (median resists outliers such as a far-away substitute).
- Ball reliable and players present: `CoA = (w_ball*Ball + w_players*P) / (w_ball + w_players)` with `w_ball=3.0`, `w_players=1.0` (ball gets 75% of the weight).
- Ball reliable, no players: ball position only.
- No reliable ball: `CoA = P`.
- Nothing valid: the last CoA is held for up to 15 frames, then no marker is drawn (never invalid coordinates).

**Ball reliability** (all must hold): confidence >= 0.5; box area < 1% of the frame;
within 50% of the frame diagonal from the player median; no impossible jump (> 10% of frame width per frame, scaled by the gap) from the last accepted ball.

## HUD
Rolling processing FPS (last 30 frames of the real loop), active player count (person boxes in the *current* frame; the ball is never counted),
CoA mode, and the frame counter. The CoA is drawn as a magenta target/crosshair.
Processing FPS is how fast the code runs; the source FPS only sets playback speed of the saved video - they are independent.

## Virtual camera (Task 3)
Mode `vertical_9_16`: crop [242, 432] -> output [242, 432]. The crop keeps the full frame height and is centred horizontally on the CoA.
**EMA:** `x_smooth[t] = alpha*x_raw[t] + (1-alpha)*x_smooth[t-1]`, alpha = 0.1 (recommended 0.05-0.15; lower = smoother but laggier).
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
See `performance_summary.md` (measured on: CPU only (Intel64 Family 6 Model 154 Stepping 4, GenuineIntel)).

## Known limitations & future improvements
<Fill in after watching the videos - e.g. ID switches during occlusions, fast pans, missed ball, referees/crowd detected as players.>

General limitations of this design:
- Tracker IDs are not perfectly stable: crossing/occluded players can swap or get new IDs. The report lists a *heuristic* count of suspected switches, not ground-truth accuracy.
- Referees, ball boys and spectators are detected as `person` (COCO has no "player" class); a pitch mask or team-colour filter would fix this.
- The small, fast ball is often missed or blurred; a ball-specific detector would help.
- Footpoints are in image pixels; a homography to field coordinates would remove perspective effects.
- BoT-SORT with camera-motion compensation / ReID would be a better choice for fast camera pans.
- The EMA lags behind fast action by design; the clip's first frames may jump if no CoA exists until later.

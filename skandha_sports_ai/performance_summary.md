# Performance Summary

**Model:** `yolov8s.pt` (pretrained COCO weights, no training)  
**Tracker:** `bytetrack.yaml` via Ultralytics `model.track(persist=True)`  
**Confidence threshold:** 0.35 | **Classes:** person (0), sports ball (32) | **Inference size:** 960  
**Hardware:** CPU only (Intel64 Family 6 Model 154 Stepping 4, GenuineIntel) (torch device `cpu`)

## Source video
- Resolution: 768x432
- Duration: 28.2 s
- Frame rate: 25.00 FPS
- Frames: 705

## Measured speed (Tasks run: [1, 2, 3])
| Measurement | Value |
|---|---|
| Frames processed | 705 / 705 (no frame skipping: True) |
| Total loop time (read + detect + track + analytics + drawing + frame writing) | 837.46 s |
| **End-to-end pipeline FPS** | **0.84** |
| Detector + tracker call only (`model.track`) | 793.94 s -> 0.89 FPS |
| Final H.264 re-encode time (not in FPS above) | 0.30 s |

*Pipeline FPS is the whole processing loop, not model-only inference FPS. It is also different from the source
video's playback FPS (25.00), which only controls how fast the saved video plays.*

## Output validation (every frame read back)
| Output | Readable frames | Resolution | FPS | MB | Result |
|---|---|---|---|---|---|
| task1 | 705 | 768x432 | 25.00 | 16.6 | PASS |
| task2 | 705 | 768x432 | 25.00 | 17.7 | PASS |
| crop_overlay | 705 | 768x432 | 25.00 | 20.7 | PASS |
| virtual | 705 | 242x432 | 25.00 | 4.0 | PASS |

## Tracking / analytics observations (automatic, no ground truth)
- Unique person IDs created: 117
- Short-lived IDs (< 25 frames, likely fragments): 46
- Suspected ID switches (heuristic only): 30
- Mean / max players per frame: 12.0 / 17
- Frames with a reliable ball: 49 of 705
- CoA mode counts: {'players median': 656, 'ball+players': 49}
- Frames without any CoA: 0

## Limitations observed
See the "Known limitations" section of README.md (filled in after visually reviewing the output videos).

"""Central configuration. Every tunable value lives here so the README can describe it exactly."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    # ---- paths ---------------------------------------------------------------
    input_video: str = "/content/input/input_video.mp4"
    project_dir: str = "/content/skandha_sports_ai"

    # ---- Task 1: detection + tracking ---------------------------------------
    model_name: str = "yolov8s.pt"        # pretrained COCO weights (yolov8n/s, yolo11n/s allowed)
    tracker: str = "bytetrack.yaml"       # or "botsort.yaml"
    conf: float = 0.35                    # assignment's starting threshold
    person_cls: int = 0                   # COCO 'person'
    ball_cls: int = 32                    # COCO 'sports ball'
    imgsz: int = 960                      # inference size; larger helps small far-away players
    device: str = "auto"                  # "auto" -> cuda:0 if available else cpu

    # ---- Task 2: analytics ---------------------------------------------------
    trail_len: int = 25                   # rolling history per Track ID (deque maxlen)
    trail_max_gap: int = 5                # break a trail if an ID vanished for > N frames
    trail_prune_after: int = 150          # forget IDs unseen for > N frames (bounds memory)

    # ball reliability (all must hold for the ball to be "reliable")
    ball_conf_reliable: float = 0.50      # stricter than the global 0.35
    ball_max_area_frac: float = 0.01      # ball box area must be < 1% of the frame
    ball_max_dist_frac: float = 0.50      # ball must be within 50% of frame diagonal of player median
    ball_max_jump_frac: float = 0.10      # max jump per frame (fraction of frame width), scaled by gap
    ball_jump_gap_limit: int = 10         # after this many frames without a ball, skip the jump test

    # Center of Action weights:  CoA = (w_ball*ball + w_players*median_players)/(w_ball+w_players)
    w_ball: float = 3.0
    w_players: float = 1.0
    coa_hold_frames: int = 15             # keep last CoA this many frames if nothing is valid

    # ---- Task 3: virtual camera ---------------------------------------------
    camera_mode: str = "vertical_9_16"    # or "zoom_16_9"
    vertical_aspect: Tuple[int, int] = (9, 16)
    zoom_factor: float = 0.5              # used by zoom_16_9 (crop = 50% of width and height)
    ema_alpha: float = 0.10               # assignment recommends 0.05 - 0.15
    virtual_out_size: Optional[Tuple[int, int]] = None  # None = native crop size, e.g. (720, 1280)

    # ---- HUD / misc ----------------------------------------------------------
    fps_window: int = 30                  # frames used for the rolling processing-FPS in the HUD
    jpg_quality: int = 90

    @property
    def output_dir(self) -> Path:
        return Path(self.project_dir) / "outputs"

    @property
    def classes(self):
        return [self.person_cls, self.ball_cls]

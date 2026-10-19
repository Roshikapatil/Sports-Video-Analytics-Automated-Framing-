"""Task 3: smooth virtual camera (fixed-size crop, EMA-smoothed centre, boundary clamping)."""
import cv2
import numpy as np


class VirtualCamera:
    def __init__(self, frame_w, frame_h, cfg):
        self.W, self.H, self.alpha = frame_w, frame_h, cfg.ema_alpha
        self.mode = cfg.camera_mode
        if self.mode == "vertical_9_16":
            ch = frame_h
            cw = int(round(ch * cfg.vertical_aspect[0] / cfg.vertical_aspect[1]))
        elif self.mode == "zoom_16_9":
            cw, ch = int(round(frame_w * cfg.zoom_factor)), int(round(frame_h * cfg.zoom_factor))
        else:
            raise ValueError(f"Unknown camera_mode {self.mode}")
        self.cw = max(2, min(cw, frame_w) // 2 * 2)       # even sizes for H.264
        self.ch = max(2, min(ch, frame_h) // 2 * 2)
        self.out_size = tuple(cfg.virtual_out_size) if cfg.virtual_out_size else (self.cw, self.ch)
        self.sx = self.sy = None          # smoothed centre (None until first valid CoA)
        self.raw_x = None

    def update(self, coa_xy):
        """coa_xy: (x, y) or None. Returns crop rect (x1, y1, x2, y2) fully inside the frame."""
        if coa_xy is not None:
            # clamp the TARGET centre first so the EMA state can never leave the valid range
            cx = float(np.clip(coa_xy[0], self.cw / 2, self.W - self.cw / 2))
            cy = float(np.clip(coa_xy[1], self.ch / 2, self.H - self.ch / 2))
            self.raw_x = cx
            if self.sx is None:                         # initialise with the first valid position
                self.sx, self.sy = cx, cy
            else:                                       # x_smooth[t] = a*x_raw[t] + (1-a)*x_smooth[t-1]
                self.sx = self.alpha * cx + (1 - self.alpha) * self.sx
                if self.mode == "zoom_16_9":            # vertical mode already uses full height
                    self.sy = self.alpha * cy + (1 - self.alpha) * self.sy
                else:
                    self.sy = cy
        # else: no valid CoA -> hold the previous smoothed position (frame centre before the first valid one)
        sx = self.sx if self.sx is not None else self.W / 2
        sy = self.sy if self.sy is not None else self.H / 2
        x1 = int(np.clip(round(sx - self.cw / 2), 0, self.W - self.cw))
        y1 = int(np.clip(round(sy - self.ch / 2), 0, self.H - self.ch))
        return x1, y1, x1 + self.cw, y1 + self.ch

    def crop(self, frame, rect):
        x1, y1, x2, y2 = rect
        c = frame[y1:y2, x1:x2]
        if (c.shape[1], c.shape[0]) != self.out_size:
            # same aspect ratio -> pure scaling (no distortion); upscaling adds no real detail
            c = cv2.resize(c, self.out_size, interpolation=cv2.INTER_CUBIC)
        return c

"""Video input, output, validation and Colab-friendly previews."""
import base64
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ probing
def probe_video(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open: {p}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
    cap.release()
    return {
        "path": str(p), "width": w, "height": h, "fps": fps, "frame_count": n,
        "duration_s": (n / fps) if fps > 0 else 0.0, "codec": codec,
        "size_mb": p.stat().st_size / 1e6,
    }


def check_specs(info):
    """Compare a probed video against the assignment's input spec. Never modifies the video."""
    checks = []
    d = info["duration_s"]
    checks.append(("Duration 15-30 s", 15 <= d <= 30, f"{d:.1f} s"))
    res_ok = (info["width"], info["height"]) in [(1280, 720), (1920, 1080)]
    checks.append(("Resolution 720p or 1080p", res_ok, f"{info['width']}x{info['height']}"))
    checks.append(("FPS 25-60", 25 <= info["fps"] <= 60, f"{info['fps']:.2f}"))
    return checks


# ------------------------------------------------------------------ writing
class VideoSink:
    """Writes frames with OpenCV (mp4v), then re-encodes to H.264 (browser/phone friendly) via ffmpeg."""

    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.path.with_name(self.path.stem + "_raw.mp4")
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(fps)
        self.writer = cv2.VideoWriter(str(self.raw_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        if not self.writer.isOpened():
            raise IOError(f"Could not open VideoWriter for {self.raw_path}")
        self.frames = 0
        self.encoder = "mp4v"

    def write(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"Frame size {frame.shape[1]}x{frame.shape[0]} != sink size {self.size}")
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.raw_path), "-c:v", "libx264",
                   "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(self.path)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and self.path.exists() and self.path.stat().st_size > 0:
                self.raw_path.unlink(missing_ok=True)
                self.encoder = "libx264 (H.264)"
                return self.path
            print("ffmpeg re-encode failed, keeping mp4v file:", r.stderr[:300])
        shutil.move(str(self.raw_path), str(self.path))
        return self.path


# ------------------------------------------------------------------ validation
def validate_video(path, expected_size=None, expected_fps=None, expected_frames=None):
    """Open the file, read EVERY frame, and check size/fps/frame-count/non-blank content."""
    res = {"path": str(path), "exists": Path(path).exists(), "opens": False, "frames_readable": 0,
           "width": 0, "height": 0, "fps": 0.0, "size_mb": 0.0, "non_blank": False, "checks": {}, "passed": False}
    if not res["exists"]:
        return res
    res["size_mb"] = Path(path).stat().st_size / 1e6
    cap = cv2.VideoCapture(str(path))
    res["opens"] = cap.isOpened()
    if not res["opens"]:
        return res
    res["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    res["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
    means = []
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if n % 20 == 0:
            means.append(float(f.mean()))
        n += 1
    cap.release()
    res["frames_readable"] = n
    res["non_blank"] = bool(means) and max(means) > 2.0
    c = res["checks"]
    c["opens"] = res["opens"]
    c["has_frames"] = n > 0
    c["non_blank"] = res["non_blank"]
    if expected_size:
        c["resolution"] = (res["width"], res["height"]) == tuple(expected_size)
    if expected_fps:
        c["fps"] = abs(res["fps"] - expected_fps) < 0.5
    if expected_frames:
        c["frame_count"] = n == expected_frames
    res["passed"] = all(c.values())
    return res


# ------------------------------------------------------------------ Colab previews
def show_frame(bgr, title=None, max_width=1000):
    """Colab-safe image preview (no cv2.imshow)."""
    from IPython.display import Image, display
    h, w = bgr.shape[:2]
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    if title:
        print(title)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    display(Image(data=buf.tobytes()))


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def show_video(path, width=640):
    """Inline video player (only for files < ~25 MB; larger ones make the notebook heavy)."""
    from IPython.display import HTML
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return HTML(f'<video width="{width}" controls><source src="data:video/mp4;base64,{data}" type="video/mp4"></video>')

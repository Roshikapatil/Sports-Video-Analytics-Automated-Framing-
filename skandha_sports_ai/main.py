"""CLI entry point:  python main.py --video clip.mp4 --tasks 1 2 3"""
import argparse
from pathlib import Path

from sports_ai.config import Config
from sports_ai.pipeline import run_pipeline
from sports_ai.video_io import probe_video, validate_video
from sports_ai.reporting import write_performance_summary, write_readme

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--project-dir", default=str(Path(__file__).parent))
    a = ap.parse_args()

    cfg = Config(input_video=a.video, project_dir=a.project_dir, model_name=a.model,
                 tracker=a.tracker, ema_alpha=a.alpha)
    info = probe_video(cfg.input_video)
    m = run_pipeline(cfg, tasks=a.tasks)
    vals = {k: validate_video(p, expected_frames=m["frames_processed"]) for k, p in m["outputs"].items()}
    write_performance_summary(cfg, info, m, vals, Path(cfg.project_dir) / "performance_summary.md")
    write_readme(cfg, info, m, Path(cfg.project_dir) / "README.md")
    print({k: v["passed"] for k, v in vals.items()})

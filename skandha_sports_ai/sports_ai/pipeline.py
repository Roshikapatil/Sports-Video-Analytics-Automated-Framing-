"""End-to-end single-pass pipeline: every frame is read once, processed once, never skipped.
tasks=(1,)       -> task1_tracking.mp4
tasks=(1,2)      -> + task2_analytics.mp4
tasks=(1,2,3)    -> + final_virtual_camera.mp4 and task3_crop_overlay.mp4
A fresh Detector (=> fresh tracker) and fresh analytics objects are created on every call, so runs never leak state."""
import csv
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2

from . import visualization as viz
from .analytics import BallSelector, CoATracker, TrailManager, TrackStats, foot_point
from .camera import VirtualCamera
from .detection import Detector, resolve_device
from .video_io import VideoSink, probe_video


def run_pipeline(cfg, tasks=(1, 2, 3), max_frames=None, verbose=True):
    tasks = set(tasks)
    if 3 in tasks:
        tasks |= {1, 2}
    if 2 in tasks:
        tasks.add(1)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(cfg.input_video).resolve()
    if out_dir.resolve() in src.parents:
        raise ValueError("Input video must not be inside the outputs folder (never process an annotated output).")

    info = probe_video(cfg.input_video)
    W, H, fps, n_expected = info["width"], info["height"], info["fps"], info["frame_count"]
    total = min(n_expected, max_frames) if max_frames else n_expected

    detector = Detector(cfg)                      # fresh model + tracker state
    trails = TrailManager(cfg.trail_len, cfg.trail_max_gap, cfg.trail_prune_after)
    ball_sel = BallSelector(cfg, W, H)
    coa_tr = CoATracker(cfg)
    stats = TrackStats(W)
    cam = VirtualCamera(W, H, cfg) if 3 in tasks else None

    paths = {"task1": out_dir / "task1_tracking.mp4"}
    sinks = {"task1": VideoSink(paths["task1"], fps, (W, H))}
    if 2 in tasks:
        paths["task2"] = out_dir / "task2_analytics.mp4"
        sinks["task2"] = VideoSink(paths["task2"], fps, (W, H))
    if 3 in tasks:
        paths["crop_overlay"] = out_dir / "task3_crop_overlay.mp4"
        paths["virtual"] = out_dir / "final_virtual_camera.mp4"
        sinks["crop_overlay"] = VideoSink(paths["crop_overlay"], fps, (W, H))
        sinks["virtual"] = VideoSink(paths["virtual"], fps, cam.out_size)

    sample_idx = {total // 4, total // 2, (3 * total) // 4}
    (out_dir / "samples").mkdir(exist_ok=True)

    cap = cv2.VideoCapture(cfg.input_video)
    frame_times = deque(maxlen=cfg.fps_window)
    log_rows, coa_modes, ball_reasons = [], Counter(), Counter()
    t_model = 0.0
    frames_done = 0
    t_start = time.perf_counter()

    try:
        while frames_done < total:
            t_frame = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            i = frames_done

            # ---- detection + tracking ----------------------------------------------------------
            t0 = time.perf_counter()
            persons, balls = detector.track(frame)
            t_model += time.perf_counter() - t0

            # ---- Task 1 overlay ----------------------------------------------------------------
            f1 = frame.copy()
            for p in persons:
                viz.draw_person(f1, p)
            for b in balls:
                viz.draw_ball(f1, b)
            sinks["task1"].write(f1)

            coa, mode, ball_xy, rect = None, "n/a", None, None
            foot_by_id, n_players = {}, len(persons)
            if 2 in tasks:
                # ---- footpoints / trails -------------------------------------------------------
                foot_list = [foot_point(p.box) for p in persons]
                foot_by_id = {p.track_id: foot_point(p.box) for p in persons if p.track_id is not None}
                trails.update(i, foot_by_id)
                stats.update(i, foot_by_id, n_players)
                # ---- ball reliability + Center of Action ----------------------------------------
                ball_xy, reason = ball_sel.select(balls, coa_tr.player_median(foot_list), i)
                ball_reasons[reason] += 1
                coa, mode = coa_tr.update(i, ball_xy, foot_list)
                coa_modes[mode] += 1

                # ---- Task 2 overlay -------------------------------------------------------------
                f2 = frame.copy()
                for tid in trails.visible_ids(i):
                    viz.draw_trail(f2, trails.segments(tid), viz.id_color(tid))
                for p in persons:
                    viz.draw_person(f2, p, show_conf=False)
                    c = viz.id_color(p.track_id) if p.track_id is not None else (170, 170, 170)
                    viz.draw_footpoint(f2, foot_point(p.box), c)
                for b in balls:
                    viz.draw_ball(f2, b)
                if coa is not None:
                    viz.draw_coa(f2, coa, mode)
                proc_fps = (len(frame_times) / sum(frame_times)) if frame_times else None
                viz.draw_hud(f2, proc_fps, n_players, mode, i, total)
                sinks["task2"].write(f2)

            # ---- Task 3 virtual camera --------------------------------------------------------
            if 3 in tasks:
                rect = cam.update(coa)
                f3 = f2.copy()
                viz.draw_crop_rect(f3, rect)
                sinks["crop_overlay"].write(f3)
                vc = cam.crop(frame, rect)             # crop from the CLEAN frame
                sinks["virtual"].write(vc)

            if i in sample_idx:
                cv2.imwrite(str(out_dir / "samples" / f"frame_{i:05d}_task{max(tasks)}.jpg"),
                            f2 if 2 in tasks else f1, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpg_quality])
                if 3 in tasks:
                    cv2.imwrite(str(out_dir / "samples" / f"frame_{i:05d}_virtual.jpg"), vc)

            log_rows.append({
                "frame": i, "n_players": n_players, "n_balls_detected": len(balls),
                "ball_reliable": int(ball_xy is not None),
                "coa_x": None if coa is None else round(coa[0], 1), "coa_y": None if coa is None else round(coa[1], 1),
                "coa_mode": mode, "crop_x1": None if rect is None else rect[0],
            })
            frames_done += 1
            frame_times.append(time.perf_counter() - t_frame)
            if verbose and frames_done % 100 == 0:
                print(f"  processed {frames_done}/{total} frames")
        t_loop = time.perf_counter() - t_start
    finally:
        cap.release()
        t_enc0 = time.perf_counter()
        encoders = {}
        for k, s in sinks.items():
            s.close()
            encoders[k] = s.encoder
        t_encode = time.perf_counter() - t_enc0

    with open(out_dir / "frame_log.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()) if log_rows else ["frame"])
        wr.writeheader()
        wr.writerows(log_rows)

    metrics = {
        "tasks_run": sorted(tasks),
        "model": cfg.model_name, "tracker": cfg.tracker, "conf": cfg.conf, "imgsz": cfg.imgsz,
        "device": detector.device,
        "frames_expected": n_expected, "frames_processed": frames_done,
        "all_frames_processed": frames_done == total and (max_frames is None),
        "source_fps": fps, "resolution": f"{W}x{H}",
        "loop_time_s": t_loop, "pipeline_fps": frames_done / t_loop if t_loop else 0.0,
        "detector_tracker_time_s": t_model, "detector_tracker_fps": frames_done / t_model if t_model else 0.0,
        "encode_time_s": t_encode, "encoders": encoders,
        "outputs": {k: str(v) for k, v in paths.items()},
        "coa_mode_counts": dict(coa_modes), "ball_decision_counts": dict(ball_reasons),
        "reliable_ball_frames": int(sum(r["ball_reliable"] for r in log_rows)),
        "frames_with_any_ball_detection": int(sum(1 for r in log_rows if r["n_balls_detected"] > 0)),
        "frames_without_coa": int(sum(1 for r in log_rows if r["coa_x"] is None)) if 2 in tasks else None,
        "track_stats": stats.summary() if 2 in tasks else None,
        "suspected_switch_events": stats.suspected_pairs[:20] if 2 in tasks else None,
    }
    if cam is not None:
        metrics["virtual_camera"] = {"mode": cfg.camera_mode, "alpha": cfg.ema_alpha,
                                     "crop_size": [cam.cw, cam.ch], "output_size": list(cam.out_size)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(f"Done: {frames_done}/{total} frames | pipeline {metrics['pipeline_fps']:.1f} FPS "
              f"| detector+tracker {metrics['detector_tracker_fps']:.1f} FPS | encode {t_encode:.1f}s")
    return metrics

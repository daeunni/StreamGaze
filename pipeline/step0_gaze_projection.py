#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
HoloAssist 3D gaze projection to 2D image

Expected folder structure (HoloAssist_gaze):
    Export_py/
    ├── Video_pitchshift.mp4      (RGB video)
    ├── Video/
    │   ├── Intrinsics.txt         (camera intrinsics)
    │   ├── Pose_sync.txt          (camera poses)
    │   └── VideoMp4Timing.txt     (video timing)
    └── Eyes/
        └── Eyes_sync.txt          (gaze data)

python step0_gaze_projection.py \
  --base_dir path/to/HoloAssist_gaze \
  --no-video
'''

import argparse
import os
import sys
import cv2
import numpy as np
import csv
from fractions import Fraction
from bisect import bisect_left
from typing import Tuple, List, Optional

# Axis transformation from HoloLens to OpenCV camera coordinate system (same as hand_eye_project.py)
AXIS_TRANSFORM = np.linalg.inv(
    np.array([
        [0,  0,  1, 0],
        [-1, 0,  0, 0],
        [0, -1,  0, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)
)

def read_intrinsics_txt(path: str) -> Tuple[np.ndarray, int, int]:
    """
    Intrinsics.txt: fx, 0, cx, 0, fy, cy, 0, 0, 1, width, height (tab-separated)
    """
    with open(path, "r") as f:
        vals = list(map(float, f.read().strip().split("\t")))
    K = np.array(vals[:9], dtype=np.float64).reshape(3, 3)
    width, height = int(vals[-2]), int(vals[-1])
    return K, width, height

def read_pose_sync(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pose_sync.txt: [time(sec)?, ticks, 16 pose elems(row-major 4x4)] per line, tab-separated
    Returns:
      ticks: (N,) int64
      poses: (N,4,4) float64  (cam_pose matrices for world->cam transformation)
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: break
            arr = list(map(float, line.split("\t")))
            rows.append(arr)
    M = np.array(rows, dtype=np.float64)
    ticks = M[:, 1].astype(np.int64)
    poses = M[:, 2:].reshape(-1, 4, 4)
    return ticks, poses

def read_video_mp4_timing(path: str) -> Tuple[int, int]:
    """
    VideoMp4Timing.txt: first line start_ticks, second line end_ticks (100ns unit)
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    start_ticks = int(lines[0])
    end_ticks = int(lines[1])
    return start_ticks, end_ticks

def read_gaze_sync(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Eyes_sync.txt (tab-separated):
      time(sec), ticks, origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, valid
    Returns:
      ticks_g   : (G,) int64
      origins_g : (G,3) float64
      dirs_g    : (G,3) float64 (normalized)
      valid_g   : (G,) bool
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: break
            arr = list(map(float, line.split("\t")))
            rows.append(arr)
    M = np.array(rows, dtype=np.float64)
    ticks_g = M[:, 1].astype(np.int64)
    origins = M[:, 2:5].astype(np.float64)
    dirs = M[:, 5:8].astype(np.float64)
    # normalize directions (for stability)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dirs = dirs / norms
    valid = M[:, 8].astype(np.int64) == 1
    return ticks_g, origins, dirs, valid

def build_frame_ticks(start_ticks: int, fps: float, n_frames: int) -> np.ndarray:
    """
    Reference tick for frame i = start_ticks + i * (10^7 / fps)  (100ns unit)
    Use fractions to minimize accumulated errors.
    """
    frac = Fraction(fps).limit_denominator()
    # frame i tick delta = i * (den / num) * 1e7
    # Avoid accumulation error using integer arithmetic
    den, num = frac.denominator, frac.numerator
    deltas = (np.arange(n_frames, dtype=np.int64) * den * (10**7)) // num
    return start_ticks + deltas

def nearest_index(sorted_ticks: np.ndarray, target: int) -> int:
    """
    Find index of closest tick to target in sorted tick array
    """
    i = bisect_left(sorted_ticks, target)
    if i <= 0:
        return 0
    if i >= len(sorted_ticks):
        return len(sorted_ticks) - 1
    # choose nearer one
    if abs(sorted_ticks[i] - target) < abs(sorted_ticks[i - 1] - target):
        return i
    return i - 1

def project_gaze_point_to_camera(origin_world: np.ndarray,
                                 dir_world: np.ndarray,
                                 cam_pose_world: np.ndarray,
                                 eye_dist: float) -> Optional[np.ndarray]:
    """
    gaze ray: P(t) = origin + t * dir, t = eye_dist (meter)
    -> world point -> camera coordinates (world->cam = inv(cam_pose))
    -> apply axis transformation (AXIS_TRANSFORM)
    Returns:
      pt_cam(3,) or None (if Z<=0 etc.)
    """
    point_world = origin_world + dir_world * eye_dist  # (3,)
    # Homogeneous coordinates
    Pw = np.concatenate([point_world, [1.0]])  # (4,)
    # world->cam: inv(cam_pose)
    cam_from_world = np.linalg.inv(cam_pose_world)
    Pc = cam_from_world @ Pw  # (4,)
    # Coordinate system alignment (HandEye): AXIS_TRANSFORM
    Pc = AXIS_TRANSFORM @ Pc
    if Pc[2] <= 0:  # Cannot project points behind camera
        return None
    return Pc[:3]

def pin_to_image(pt_cam: np.ndarray, K: np.ndarray) -> Tuple[float, float]:
    """
    OpenCV pinhole projection: u = fx*X/Z + cx, v = fy*Y/Z + cy
    """
    X, Y, Z = pt_cam
    u = (X / Z) * K[0, 0] + K[0, 2]
    v = (Y / Z) * K[1, 1] + K[1, 2]
    return float(u), float(v)

def process_batch(args):
    """Batch process all sessions in base_dir"""
    from pathlib import Path
    
    base_dir = Path(args.base_dir)
    print(f"\n[INFO] Scanning for sessions in: {base_dir}")
    
    # Find all valid sessions
    sessions = []
    for session_dir in sorted(base_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        
        export_py = session_dir / "Export_py"
        if not export_py.exists():
            continue
        
        # Check if all required files exist
        video_file = export_py / args.video_name
        eyes_file = export_py / "Eyes" / "Eyes_sync.txt"
        intrinsics_file = export_py / "Video" / "Intrinsics.txt"
        
        if video_file.exists() and eyes_file.exists() and intrinsics_file.exists():
            sessions.append({
                'name': session_dir.name,
                'path': str(export_py)
            })
    
    print(f"[INFO] Found {len(sessions)} valid sessions")
    print(f"\n[INFO] Starting batch processing...")
    print(f"=" * 80)
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, session in enumerate(sessions, 1):
        session_name = session['name']
        export_py_path = session['path']
        
        # Output paths
        video_out = os.path.join(export_py_path, f"{session_name}_gaze.mp4")
        csv_out = os.path.join(export_py_path, f"{session_name}_gaze_2d.csv")
        
        # Check if already processed
        if os.path.exists(csv_out) and not args.overwrite:
            skipped_count += 1
            print(f"[{i}/{len(sessions)}] {session_name}: ⏭ Skipped (already exists)")
            continue
        
        print(f"\n[{i}/{len(sessions)}] Processing: {session_name}")
        
        # Create temporary args for this session
        session_args = argparse.Namespace(**vars(args))
        session_args.folder_path = export_py_path
        session_args.out = video_out
        session_args.out_csv = csv_out
        
        try:
            process_single_session(session_args)
            success_count += 1
            print(f"            ✓ Success")
        except Exception as e:
            failed_count += 1
            print(f"            ✗ Failed: {e}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"[SUMMARY] Batch processing complete")
    print(f"  Total sessions:   {len(sessions)}")
    print(f"  ✓ Successful:     {success_count}")
    print(f"  ⏭ Skipped:        {skipped_count}")
    print(f"  ✗ Failed:         {failed_count}")
    if len(sessions) > 0:
        print(f"  Success rate:     {success_count/len(sessions)*100:.1f}%")

def main():
    ap = argparse.ArgumentParser(
        description="Overlay 2D gaze onto HoloAssist RGB video (single or batch processing).",
        epilog="Single: python step0_gaze_projection.py --folder_path /path/to/Export_py\nBatch: python step0_gaze_projection.py --base_dir /path/to/HoloAssist_gaze --no-video"
    )
    # Mode selection: single session or batch
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--folder_path",
                    help="Single session: Sequence root path up to .../Rxxx-.../Export_py")
    mode_group.add_argument("--base_dir",
                    help="Batch mode: Base directory containing all session folders (e.g., HoloAssist_gaze)")
    ap.add_argument("--video_name", default="Video_pitchshift.mp4",
                    help="Video filename (default: Video_pitchshift.mp4)")
    ap.add_argument("--eye_dist", type=float, default=0.5,
                    help="Distance (m) along the gaze ray to place the point (default: 0.5)")
    ap.add_argument("--out", default="output_gaze.mp4",
                    help="Output video path")
    ap.add_argument("--out_csv", default="gaze_2d_coordinates.csv",
                    help="Output CSV file for 2D gaze coordinates (default: gaze_2d_coordinates.csv)")
    ap.add_argument("--dot_radius", type=int, default=6)
    ap.add_argument("--dot_thickness", type=int, default=-1, help="-1 to fill circle")
    ap.add_argument("--dot_bgr", type=int, nargs=3, default=[0, 0, 255],
                    help="Gaze dot color in BGR (default red)")
    ap.add_argument("--skip_invalid", action="store_true",
                    help="Skip frames if nearest gaze is invalid")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip video generation (only save CSV coordinates, much faster)")
    ap.add_argument("--overwrite", action="store_true",
                    help="[Batch mode] Overwrite existing results")
    args = ap.parse_args()
    
    # Check if batch mode or single mode
    if args.base_dir:
        # Batch processing mode
        from pathlib import Path
        process_batch(args)
    else:
        # Single session mode
        process_single_session(args)

def process_single_session(args):
    """Process a single session"""
    base = args.folder_path
    # HoloAssist_gaze structure: Video_pitchshift.mp4 is directly under Export_py/
    video_path = os.path.join(base, args.video_name)
    video_timing_path = os.path.join(base, "Video", "VideoMp4Timing.txt")
    pose_sync_path = os.path.join(base, "Video", "Pose_sync.txt")
    intrinsics_path = os.path.join(base, "Video", "Intrinsics.txt")
    eyes_sync_path = os.path.join(base, "Eyes", "Eyes_sync.txt")

    # Check file existence
    print(f"\n[INFO] Checking required files...")
    for p in [video_path, video_timing_path, pose_sync_path, intrinsics_path, eyes_sync_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)
    print(f"       ✓ All files found")

    # Load data
    print(f"\n[INFO] Loading camera and gaze data...")
    K, width, height = read_intrinsics_txt(intrinsics_path)
    print(f"       ✓ Camera intrinsics: {width}x{height}")
    
    pose_ticks, poses = read_pose_sync(pose_sync_path)
    print(f"       ✓ Camera poses: {len(poses)} samples")
    
    start_ticks, end_ticks = read_video_mp4_timing(video_timing_path)
    print(f"       ✓ Video timing: {start_ticks} to {end_ticks}")
    
    gaze_ticks, gaze_origins, gaze_dirs, gaze_valid = read_gaze_sync(eyes_sync_path)
    valid_gaze_count = np.sum(gaze_valid)
    print(f"       ✓ Gaze data: {len(gaze_ticks)} samples ({valid_gaze_count} valid)")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output settings (keep original resolution)
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.out, fourcc, fps, (vw, vh), True)
    else:
        out = None
        print(f"[INFO] Skipping video generation (--no-video enabled)")

    # Frame-based tick array
    frame_ticks = build_frame_ticks(start_ticks, fps, n_frames)

    # Ensure pose/gaze ticks are sorted
    order_pose = np.argsort(pose_ticks)
    pose_ticks = pose_ticks[order_pose]
    poses = poses[order_pose]

    order_gaze = np.argsort(gaze_ticks)
    gaze_ticks = gaze_ticks[order_gaze]
    gaze_origins = gaze_origins[order_gaze]
    gaze_dirs = gaze_dirs[order_gaze]
    gaze_valid = gaze_valid[order_gaze]

    # List to store 2D coordinates
    gaze_2d_records = []

    # Progress output settings
    print(f"\n[INFO] Processing {n_frames} frames...")
    print(f"       Video: {vw}x{vh} @ {fps:.2f} fps")
    print(f"       Progress updates every {max(1, n_frames//20)} frames")
    
    # Main loop
    frame_idx = 0
    while frame_idx < n_frames:
        # Read frame only if video output is needed
        if not args.no_video:
            ok, frame = cap.read()
            if not ok: break
        else:
            frame = None

        # Progress output (every 5%)
        if frame_idx % max(1, n_frames // 20) == 0 or frame_idx == n_frames - 1:
            progress = (frame_idx + 1) / n_frames * 100
            print(f"       [{progress:5.1f}%] Frame {frame_idx + 1}/{n_frames}")

        ftick = int(frame_ticks[frame_idx])

        # pose: use closest pose to frame tick
        pi = nearest_index(pose_ticks, ftick)
        cam_pose = poses[pi]

        # gaze: use closest gaze to frame tick
        gi = nearest_index(gaze_ticks, ftick)
        is_valid = gaze_valid[gi]
        
        if args.skip_invalid and not is_valid:
            # skip if invalid
            gaze_2d_records.append({
                'frame_idx': frame_idx,
                'timestamp_sec': frame_idx / fps,
                'u': None,
                'v': None,
                'valid': False,
                'in_frame': False
            })
            if out is not None:
                out.write(frame)
            frame_idx += 1
            continue

        # Try even if invalid (can use skip_invalid if desired)
        origin_w = gaze_origins[gi]
        dir_w = gaze_dirs[gi]

        # world -> camera -> axis transform
        pt_cam = project_gaze_point_to_camera(origin_w, dir_w, cam_pose, args.eye_dist)

        u_coord, v_coord = None, None
        in_frame = False
        
        if pt_cam is not None:
            u, v = pin_to_image(pt_cam, K)
            u_coord, v_coord = u, v
            # Draw only if inside frame
            if 0 <= u < vw and 0 <= v < vh:
                in_frame = True
                if frame is not None:  # Draw only when generating video
                    cv2.circle(
                        frame,
                        (int(round(u)), int(round(v))),
                        args.dot_radius,
                        args.dot_bgr,
                        args.dot_thickness
                    )
        
        # Record 2D coordinates
        gaze_2d_records.append({
            'frame_idx': frame_idx,
            'timestamp_sec': frame_idx / fps,
            'u': u_coord,
            'v': v_coord,
            'valid': is_valid,
            'in_frame': in_frame
        })

        if out is not None:
            out.write(frame)
        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()
        print(f"\n[DONE] Video saved: {args.out}")
    else:
        print(f"\n[INFO] Video generation skipped")
    
    # Save 2D coordinates to CSV file
    print(f"[INFO] Saving 2D coordinates to CSV...")
    with open(args.out_csv, 'w', newline='') as csvfile:
        fieldnames = ['frame_idx', 'timestamp_sec', 'u', 'v', 'valid', 'in_frame']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gaze_2d_records)
    
    print(f"[DONE] Saved 2D coordinates: {args.out_csv}")
    print(f"       Total frames: {len(gaze_2d_records)}")
    valid_count = sum(1 for r in gaze_2d_records if r['valid'] and r['u'] is not None)
    print(f"       Valid gaze points: {valid_count}/{len(gaze_2d_records)} ({valid_count/len(gaze_2d_records)*100:.1f}%)")

if __name__ == "__main__":
    main()

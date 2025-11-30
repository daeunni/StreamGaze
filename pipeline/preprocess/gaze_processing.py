"""
Gaze processing functions for EGTEA dataset
"""

import pandas as pd
import numpy as np
import cv2
import os
import imageio
from pathlib import Path


def _str2frame(frame_str, fps=None):
    """Convert frame string to frame number and time in seconds"""
    if fps==None:
        fps = 24

    splited_time = frame_str.split(':')
    assert len(splited_time) == 4

    time_sec = 3600 * int(splited_time[0]) \
               + 60 * int(splited_time[1]) +  int(splited_time[2])

    frame_num = time_sec * fps + int(splited_time[3])

    return frame_num, time_sec


def parse_gtea_gaze(filename, gaze_resolution=None):
    '''
    Read gaze file in CSV format
    Input: 
        name of a gaze csv file
    return 
        an array where the each row follows: 
        (frame_num): px (0-1), py (0-1), gaze_type
    '''
    if gaze_resolution is None:
        # gaze resolution (default 1280*960)
        gaze_resolution = np.array([960, 1280], dtype=np.float32)

    # load all lines
    lines = [line.rstrip('\n') for line in open(filename)]
    # deal with different version of begaze
    ver = 1
    if '## Number of Samples:' in lines[9]:
        line = lines[9]
        ver = 1
    else:
        line = lines[10]
        ver = 2

    # get the number of samples
    values = line.split()
    num_samples = int(values[4])

    # skip the header
    lines = lines[34:]

    # pre-allocate the array 
    # (Note the number of samples in header is not always accurate)
    num_frames = 0
    gaze_data = np.zeros((num_samples*2, 4), dtype=np.float32)

    # parse each line
    for line in lines:
        values = line.split()
        # read gaze_x, gaze_y, gaze_type and frame_number from the file
        if len(values)==7 and ver==1:
            px, py = float(values[3]), float(values[4])
            frame = int(values[5])
            gaze_type = values[6]

        elif len(values)==26 and ver==2:
            px, py = float(values[5]), float(values[6])
            frame, time_sec = _str2frame(values[-2])
            gaze_type = values[-1]

        else:
            raise ValueError('Format not supported')

        # avg the gaze points if needed
        if gaze_data[frame, 2] > 0:
            gaze_data[frame,0] = (gaze_data[frame,0] + px)/2.0
            gaze_data[frame,1] = (gaze_data[frame,1] + py)/2.0
        else:
            gaze_data[frame,0] = px
            gaze_data[frame,1] = py

        # gaze type
        # 0 untracked (no gaze point available); 
        # 1 fixation (pause of gaze); 
        # 2 saccade (jump of gaze); 
        # 3 unkown (unknown gaze type return by BeGaze); 
        # 4 truncated (gaze out of range of the video)
        if gaze_type == 'Fixation':
            gaze_data[frame, 2] = 1
        elif gaze_type == 'Saccade':
            gaze_data[frame, 2] = 2 
        else:
            gaze_data[frame, 2] = 3

        num_frames = max(num_frames, frame)

    gaze_data = gaze_data[:num_frames+1, :]

    # post processing:
    # (1) filter out out of bound gaze points
    # (2) normalize gaze into the range of 0-1
    for frame_idx in range(0, num_frames+1):

        px = gaze_data[frame_idx, 0] 
        py = gaze_data[frame_idx, 1]
        gaze_type = gaze_data[frame_idx, 2]

        # truncate the gaze points
        if (px < 0 or px > (gaze_resolution[1]-1)) \
           or (py < 0 or py > (gaze_resolution[0]-1)):
            gaze_data[frame_idx, 2] = 4

        px = min(max(0, px), gaze_resolution[1]-1)
        py = min(max(0, py), gaze_resolution[0]-1)

        # normalize the gaze
        gaze_data[frame_idx, 0] = px / gaze_resolution[1]
        gaze_data[frame_idx, 1] = py / gaze_resolution[0]
        gaze_data[frame_idx, 2] = gaze_type            

    return gaze_data


def parse_ego4d_gaze(csv_path, fps=30):
    """
    Parse Ego4D gaze data from CSV format
    
    Input:
        csv_path: Path to Ego4D gaze CSV file
        fps: Video frame rate (default: 30 for Ego4D)
        
    Returns:
        numpy array where each row is: (frame_num): px (0-1), py (0-1), gaze_type
        Format matches parse_gtea_gaze output for compatibility
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Get required columns
    timestamps = df['canonical_timestamp_s'].values
    norm_x = df['norm_pos_x'].values
    norm_y = df['norm_pos_y'].values
    confidence = df['confidence'].values
    
    # Calculate total video duration and number of frames
    max_time = timestamps[-1]
    num_frames = int(max_time * fps) + 1
    
    # Initialize gaze data array (frame_idx, px, py, gaze_type)
    gaze_data = np.zeros((num_frames, 4), dtype=np.float32)
    
    # Mark frames with gaze data
    frame_has_data = np.zeros(num_frames, dtype=bool)
    
    # Process each gaze sample
    for i in range(len(timestamps)):
        # Convert timestamp to frame index
        frame_idx = int(timestamps[i] * fps)
        
        if frame_idx >= num_frames:
            continue
            
        # Get normalized gaze position
        px = norm_x[i]
        py = norm_y[i]
        conf = confidence[i]
        
        # Skip invalid gaze points
        if np.isnan(px) or np.isnan(py):
            continue
            
        # Average multiple samples for the same frame
        if frame_has_data[frame_idx]:
            gaze_data[frame_idx, 0] = (gaze_data[frame_idx, 0] + px) / 2.0
            gaze_data[frame_idx, 1] = (gaze_data[frame_idx, 1] + py) / 2.0
        else:
            gaze_data[frame_idx, 0] = px
            gaze_data[frame_idx, 1] = py
            frame_has_data[frame_idx] = True
        
        # Gaze type based on confidence
        # For Ego4D, we'll treat all valid gaze points as fixations (type 1)
        # since the dataset doesn't explicitly label fixation vs saccade
        if conf >= 0.9:
            gaze_data[frame_idx, 2] = 1  # High confidence -> fixation
        elif conf >= 0.5:
            gaze_data[frame_idx, 2] = 2  # Medium confidence -> saccade
        else:
            gaze_data[frame_idx, 2] = 3  # Low confidence -> unknown
    
    # Fill in frames without gaze data
    for frame_idx in range(num_frames):
        if not frame_has_data[frame_idx]:
            gaze_data[frame_idx, 2] = 0  # Untracked
            
        # Clamp values to valid range [0, 1]
        px = gaze_data[frame_idx, 0]
        py = gaze_data[frame_idx, 1]
        
        if px < 0 or px > 1 or py < 0 or py > 1:
            gaze_data[frame_idx, 2] = 4  # Truncated (out of bounds)
            
        gaze_data[frame_idx, 0] = np.clip(px, 0, 1)
        gaze_data[frame_idx, 1] = np.clip(py, 0, 1)
    
    return gaze_data


def parse_holoassist_gaze(csv_path, fps=24.46, video_resolution=(896, 504)):
    """
    Parse HoloAssist gaze data from CSV format (generated by step0_gaze_projection.py)
    
    Input:
        csv_path: Path to HoloAssist gaze CSV file (frame_idx, timestamp_sec, u, v, valid, in_frame)
        fps: Video frame rate (default: 24.46 for HoloAssist)
        video_resolution: (width, height) tuple (default: 896x504 for HoloAssist)
        
    Returns:
        numpy array where each row is: (frame_num): px (0-1), py (0-1), gaze_type
        Format matches parse_gtea_gaze output for compatibility
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Get video dimensions
    width, height = video_resolution
    
    # Get max frame index to determine array size
    max_frame = int(df['frame_idx'].max())
    num_frames = max_frame + 1
    
    # Initialize gaze data array
    gaze_data = np.zeros((num_frames, 4), dtype=np.float32)
    
    # Process each row
    for _, row in df.iterrows():
        frame_idx = int(row['frame_idx'])
        u = row['u']  # pixel coordinate
        v = row['v']  # pixel coordinate
        valid = row['valid']
        in_frame = row['in_frame']
        
        # Normalize coordinates to 0-1 range
        if pd.notna(u) and pd.notna(v):
            px = u / width
            py = v / height
        else:
            px = 0.5  # Default to center if invalid
            py = 0.5
        
        gaze_data[frame_idx, 0] = px
        gaze_data[frame_idx, 1] = py
        
        # Determine gaze type
        # 0: untracked, 1: fixation, 2: saccade, 3: unknown, 4: truncated (out of bounds)
        if not valid or pd.isna(u) or pd.isna(v):
            gaze_data[frame_idx, 2] = 0  # untracked
        elif not in_frame:
            gaze_data[frame_idx, 2] = 4  # truncated (out of frame)
        else:
            # For HoloAssist, we'll mark all valid in-frame points as fixation
            # (actual fixation detection will happen later in extract_fixation_segments)
            gaze_data[frame_idx, 2] = 1  # fixation
    
    return gaze_data


def parse_egoexo_gaze(npy_path, fps=30):
    """
    Parse EgoExoLearn gaze data from .npy format
    
    Input:
        npy_path: Path to EgoExoLearn gaze .npy file
        fps: Video frame rate (default: 30 for EgoExoLearn)
        
    Returns:
        numpy array where each row is: (frame_num): px (0-1), py (0-1), gaze_type
        Format matches parse_gtea_gaze output for compatibility
    """
    # Load .npy file
    gaze_npy = np.load(npy_path)
    # Expected format: (N, 3) where columns are [x_norm, y_norm, validity]
    
    num_frames = len(gaze_npy)
    
    # Initialize gaze data array (frame_idx, px, py, gaze_type)
    gaze_data = np.zeros((num_frames, 4), dtype=np.float32)
    
    # Process each frame
    for frame_idx in range(num_frames):
        px = gaze_npy[frame_idx, 0]  # Normalized X (0-1)
        py = gaze_npy[frame_idx, 1]  # Normalized Y (0-1)
        is_valid = gaze_npy[frame_idx, 2]  # Validity flag (0 or 1)
        
        gaze_data[frame_idx, 0] = px
        gaze_data[frame_idx, 1] = py
        
        # Gaze type based on validity
        # 0: untracked, 1: fixation (we treat all valid points as fixations)
        if is_valid == 1:
            # Check if gaze is within valid range
            if 0 <= px <= 1 and 0 <= py <= 1:
                gaze_data[frame_idx, 2] = 1  # Valid -> fixation
            else:
                gaze_data[frame_idx, 2] = 4  # Out of bounds -> truncated
        else:
            gaze_data[frame_idx, 2] = 0  # Invalid -> untracked
            
        # Clamp values to valid range [0, 1]
        gaze_data[frame_idx, 0] = np.clip(px, 0, 1)
        gaze_data[frame_idx, 1] = np.clip(py, 0, 1)
    
    return gaze_data


def extract_fixation_segments(df, radius_thresh=0.05, duration_thresh=0.5, gap_thresh=0.2):
    """
    Extracts fixation segments from frame-wise gaze data,
    allowing short interruptions (gaps) within fixations.

    Args:
        df (DataFrame): Gaze data with 'px', 'py', 'time_seconds' columns
        radius_thresh (float): Max distance from start to consider still part of fixation
        duration_thresh (float): Minimum duration for valid fixation (in seconds)
        gap_thresh (float): Allowable brief interruption time (e.g., eye flick) in seconds

    Returns:
        fixations (list): List of fixation segments
    """
    if len(df) == 0:
        print("Empty dataframe, no fixations to extract")
        return []
        
    timestamps = df['time_seconds'].values
    xs = df['px'].values
    ys = df['py'].values

    fixations = []
    start_idx = 0
    i = 1

    while i < len(xs):
        dist = np.sqrt((xs[i] - xs[start_idx])**2 + (ys[i] - ys[start_idx])**2)

        if dist > radius_thresh:
            gap_duration = timestamps[i] - timestamps[i - 1]

            if gap_duration <= gap_thresh:
                # Short flick — ignore and continue
                i += 1
                continue

            # Finalize fixation if it was long enough
            duration = timestamps[i - 1] - timestamps[start_idx]
            if duration >= duration_thresh:
                fixations.append({
                    "start_time": float(timestamps[start_idx]),
                    "end_time": float(timestamps[i - 1]),
                    "center_x": float(np.mean(xs[start_idx:i])),
                    "center_y": float(np.mean(ys[start_idx:i])),
                    "duration": float(duration)
                })

            # Start new fixation
            start_idx = i

        i += 1

    # Consider the last fixation as well
    duration = timestamps[-1] - timestamps[start_idx]
    if duration >= duration_thresh:
        fixations.append({
            "start_time": float(timestamps[start_idx]),
            "end_time": float(timestamps[-1]),
            "center_x": float(np.mean(xs[start_idx:])),
            "center_y": float(np.mean(ys[start_idx:])),
            "duration": float(duration)
        })

    print(f"Total fixations extracted: {len(fixations)}")
    return fixations


def detect_saccade_segments_ivt_dispersion(
    gaze_xs, gaze_ys, timestamps,
    velocity_thresh=0.05,
    min_duration=0.05,
    dispersion_thresh=0.03  
    ):
    """
    Detect saccade segments using I-VT (velocity threshold) + spatial dispersion check.

    Args:
        gaze_xs, gaze_ys: list of gaze positions (normalized px/py, e.g., 0~1)
        timestamps: list of time in seconds
        velocity_thresh: threshold for movement speed between points (normalized distance / second)
        min_duration: minimum duration to be considered a valid saccade (seconds)
        dispersion_thresh: spatial dispersion (range in px/py) required to count as a true saccade

    Returns:
        saccade_segments: list of dictionaries with start_time, end_time, center_x/y, duration
    """
    gaze_xs = np.array(gaze_xs)
    gaze_ys = np.array(gaze_ys)
    timestamps = np.array(timestamps)

    saccade_mask = np.zeros(len(gaze_xs), dtype=bool)

    # Step 1: velocity 기반 마스크 생성
    for i in range(1, len(gaze_xs)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            continue
        dx = gaze_xs[i] - gaze_xs[i - 1]
        dy = gaze_ys[i] - gaze_ys[i - 1]
        v = np.sqrt(dx ** 2 + dy ** 2) / dt
        if v > velocity_thresh:
            saccade_mask[i] = True

    # Step 2: 연속 구간 -> saccade 후보
    saccade_segments = []
    in_segment = False
    start_idx = 0

    for i in range(len(saccade_mask)):
        if saccade_mask[i]:
            if not in_segment:
                in_segment = True
                start_idx = i
        else:
            if in_segment:
                end_idx = i - 1
                duration = timestamps[end_idx] - timestamps[start_idx]
                if duration >= min_duration:
                    x_range = gaze_xs[start_idx:end_idx+1].max() - gaze_xs[start_idx:end_idx+1].min()
                    y_range = gaze_ys[start_idx:end_idx+1].max() - gaze_ys[start_idx:end_idx+1].min()
                    if max(x_range, y_range) > dispersion_thresh:
                        saccade_segments.append({
                            "start_time": float(timestamps[start_idx]),
                            "end_time": float(timestamps[end_idx]),
                            "center_x": float(np.mean(gaze_xs[start_idx:end_idx+1])),
                            "center_y": float(np.mean(gaze_ys[start_idx:end_idx+1])),
                            "duration": float(duration)
                        })
                in_segment = False

    # 마지막 구간 처리
    if in_segment:
        end_idx = len(saccade_mask) - 1
        duration = timestamps[end_idx] - timestamps[start_idx]
        if duration >= min_duration:
            x_range = gaze_xs[start_idx:end_idx+1].max() - gaze_xs[start_idx:end_idx+1].min()
            y_range = gaze_ys[start_idx:end_idx+1].max() - gaze_ys[start_idx:end_idx+1].min()
            if max(x_range, y_range) > dispersion_thresh:
                saccade_segments.append({
                    "start_time": float(timestamps[start_idx]),
                    "end_time": float(timestamps[end_idx]),
                    "center_x": float(np.mean(gaze_xs[start_idx:end_idx+1])),
                    "center_y": float(np.mean(gaze_ys[start_idx:end_idx+1])),
                    "duration": float(duration)
                })

    print(f"Total refined saccade segments detected: {len(saccade_segments)}")
    return saccade_segments


def extract_confusion_segments(df, duration_thresh=0.5):
    """
    Extracts confusion segments using I-VT + Dispersion method
    for more accurate saccade detection.

    Returns:
        confusion_segments (list): List of sustained saccade segments
    """
    if len(df) == 0:
        print("Empty dataframe, no confusion segments to extract")
        return []
    
    # Extract gaze data
    gaze_xs = df['px'].values
    gaze_ys = df['py'].values
    timestamps = df['time_seconds'].values
    
    # Use I-VT + Dispersion method with stricter parameters for confusion detection
    confusion_segments = detect_saccade_segments_ivt_dispersion(
        gaze_xs, gaze_ys, timestamps,
        velocity_thresh=0.01,       # 속도 기준 (낮은 값으로 더 민감하게)
        min_duration=2.5,             # 최소 2.5초 - confusion은 긴 saccade
        dispersion_thresh=0.1       # 공간적 분산 임계값
    )
    
    print(f"Confusion (refined saccade) segments extracted: {len(confusion_segments)}")
    return confusion_segments

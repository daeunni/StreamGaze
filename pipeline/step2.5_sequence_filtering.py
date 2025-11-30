"""
Step 2.5: Sequence Filtering (Simplified)
Merges fixations from Step 2 and filters consecutive similar episodes

Usage:
    python step2.5_sequence_filtering.py --dataset egtea
    python step2.5_sequence_filtering.py --dataset ego4d
    python step2.5_sequence_filtering.py --dataset egoexo
    python step2.5_sequence_filtering.py --dataset holoassist
    python step2.5_sequence_filtering.py --dataset ego4d --reverse  # Process in reverse order

Input:
    - {video}_fixation_with_internvl_v2_scene.csv (from Step 2)

Features:
    - Spatial/temporal fixation merging into episodes
    - Consecutive similar episode removal (visual similarity based)
    - GIF generation for visual inspection
    - Automatically skips already processed videos

Note: Scene filtering and duration filtering have been removed.
      Use step1.5 for those filters if needed.
"""

import os
import ast
import cv2
import imageio
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import argparse

# Additional imports for text processing
import unicodedata
import re

# Get pipeline directory dynamically
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Metadata Merging Functions
# =============================================================================

def merge_all_metadata(dataset, base_data_path):
    """
    Merge all video metadata files into a single total_metadata.csv
    
    Args:
        dataset: Dataset name (egtea, ego4d, egoexo, holoassist)
        base_data_path: Base path containing video subdirectories
    """
    import glob
    
    print(f"Collecting metadata files from: {base_data_path}")
    
    # Find all fixation_merged_filtered_v2.csv files
    pattern = os.path.join(base_data_path, "*", "*_fixation_merged_filtered_v2.csv")
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print(f"⚠️  No metadata files found matching pattern: {pattern}")
        print(f"   Looking for alternative file pattern...")
        # Try alternative pattern
        pattern = os.path.join(base_data_path, "*", "*_fixation_with_internvl_v2_scene.csv")
        csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print(f"❌ No metadata files found to merge!")
        return
    
    print(f"Found {len(csv_files)} metadata files")
    
    # Read and merge all CSVs
    all_dfs = []
    for csv_file in tqdm(csv_files, desc="Reading CSV files"):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                # Extract video name from file path
                # e.g., /path/to/OP01-R01-PastaSalad/OP01-R01-PastaSalad_fixation_merged_filtered_v2.csv
                video_name = os.path.basename(os.path.dirname(csv_file))
                df['video_name'] = video_name
                df['source_file'] = csv_file
                all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {csv_file}: {e}")
    
    if not all_dfs:
        print(f"❌ No valid dataframes to merge!")
        return
    
    # Concatenate all dataframes
    print(f"Merging {len(all_dfs)} dataframes...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to final_data/{dataset}/total_metadata.csv
    output_dir = os.path.join(PIPELINE_DIR, 'final_data', dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'total_metadata.csv')
    
    merged_df.to_csv(output_file, index=False)
    
    print(f"✅ Merged metadata saved: {output_file}")
    print(f"   Total rows: {len(merged_df):,}")
    print(f"   Total videos: {merged_df['video_name'].nunique() if 'video_name' in merged_df.columns else 'N/A'}")
    print(f"   Columns: {list(merged_df.columns)}")

# =============================================================================
# Utility Functions
# =============================================================================

def load_csv_safely(file_path):
    """Safely load CSV with error handling"""
    try:
        return pd.read_csv(
            file_path,
            sep=',',
            quotechar='"',
            skipinitialspace=True,
            on_bad_lines='skip',
            engine='python'
        )
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def light_norm(s: str) -> str:
    """Text preprocessing (for prompt stabilization, not too aggressive)"""
    s = unicodedata.normalize("NFKC", (s or "")).strip()
    s = re.sub(r"\s+", " ", s)
    return s


# =============================================================================
# Data Processing and Fixation Merging
# =============================================================================

def is_black_frame(frame, threshold=10):
    """Check if single frame is black screen"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    return mean_val < threshold

def is_black_scene(video_path, start_time, end_time, n_samples=5, threshold=10):
    """
    Determine as black scene if most frames in segment are black
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_points = np.linspace(start_time, end_time, n_samples)
    black_count = 0
    
    for t in time_points:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if is_black_frame(frame, threshold):
            black_count += 1
    
    cap.release()
    return black_count >= (n_samples // 2 + 1)  # Black scene if more than half are black


def check_visual_consistency_hist(video_path, start_time, end_time, 
                                  n_samples=8, hist_threshold=0.9):
    """
    OpenCV histogram-based scene consistency check
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return True, 1.0, 1.0, []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_points = np.linspace(start_time, end_time, n_samples)
    frames = []
    
    for t in time_points:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            frames.append(hist)
    
    cap.release()
    
    if len(frames) < 2:
        return True, 1.0, 1.0, []
    
    scores = []
    for i in range(len(frames) - 1):
        score = cv2.compareHist(frames[i], frames[i+1], cv2.HISTCMP_CORREL)
        scores.append(score)
    
    min_score = float(np.min(scores))
    mean_score = float(np.mean(scores))
    is_consistent = min_score >= hist_threshold
    
    return is_consistent, min_score, mean_score, scores


def filter_scene_consistency_by_hist(merged_df, video_path, hist_threshold=0.9, 
                             min_duration=1.5, black_thresh=10):
    """
    Filter merged episodes based on histogram + black screen detection
    """
    filtered_indices = []
    stats = []
    
    print(f"Filtering {len(merged_df)} episodes with histogram threshold={hist_threshold}")
    
    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        duration = row['end_time_seconds'] - row['start_time_seconds']
        
        # Automatically pass episodes that are too short
        if duration < min_duration:
            filtered_indices.append(idx)
            stats.append({
                'idx': idx,
                'duration': duration,
                'is_consistent': True,
                'min_score': 1.0,
                'mean_score': 1.0,
                'reason': 'too_short'
            })
            continue
        
        # Black screen filtering
        if is_black_scene(video_path, row['start_time_seconds'], row['end_time_seconds'],
                          n_samples=5, threshold=black_thresh):
            stats.append({
                'idx': idx,
                'duration': duration,
                'is_consistent': False,
                'min_score': 0.0,
                'mean_score': 0.0,
                'reason': 'black_scene'
            })
            print(f"  ⚫ Episode {idx}: duration={duration:.2f}s -> black scene removed")
            continue
        
        # Histogram-based consistency check
        is_consistent, min_score, mean_score, scores = check_visual_consistency_hist(
            video_path,
            row['start_time_seconds'],
            row['end_time_seconds'],
            n_samples=max(5, int(duration * 2)),
            hist_threshold=hist_threshold
        )
        
        if is_consistent:
            filtered_indices.append(idx)
            stats.append({
                'idx': idx,
                'duration': duration,
                'is_consistent': True,
                'min_score': min_score,
                'mean_score': mean_score,
                'reason': 'passed'
            })
        else:
            stats.append({
                'idx': idx,
                'duration': duration,
                'is_consistent': False,
                'min_score': min_score,
                'mean_score': mean_score,
                'reason': 'scene_change'
            })
            print(f"  ❌ Episode {idx}: duration={duration:.2f}s, "
                  f"min_score={min_score:.3f}, mean_score={mean_score:.3f}")
    
    filtered_df = merged_df.loc[filtered_indices].reset_index(drop=True)
    stats_df = pd.DataFrame(stats)
    
    print(f"\n✅ Kept: {len(filtered_df)}/{len(merged_df)} episodes")
    print(f"❌ Filtered out: {len(merged_df) - len(filtered_df)} episodes")
    
    return filtered_df, stats_df


def normalize_column_names(df, dataset='egtea'):
    """Normalize column names between datasets"""
    df = df.copy()
    
    if dataset in ['ego4d', 'egoexo', 'holoassist']:
        # These datasets use 'start_time', 'end_time' or already have 'start_time_seconds'
        # Rename to standard names for internal processing if needed
        if 'start_time' in df.columns and 'start_time_seconds' not in df.columns:
            df['start_time_seconds'] = df['start_time']
        if 'end_time' in df.columns and 'end_time_seconds' not in df.columns:
            df['end_time_seconds'] = df['end_time']
    
    # Add fixation_id if not present
    if 'fixation_id' not in df.columns:
        df['fixation_id'] = range(len(df))
    
    return df


def process_internvl_data_simple(internvl_df, video_name, dataset='egtea'):
    """Process internvl data for a single video (simplified version without LLM normalization)"""
    print(f"Processing internvl data for {video_name}")
    
    # Normalize column names first
    internvl_df = normalize_column_names(internvl_df, dataset)
    
    # Convert exact_gaze_object column from string(str) to dict
    internvl_df = internvl_df.copy()
    internvl_df["exact_gaze_object"] = internvl_df["exact_gaze_object"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Extract object names for canonical naming (simplified)
    def extract_object_name(d):
        if isinstance(d, str):
            try:
                d = ast.literal_eval(d)
            except Exception:
                d = {}
        # Handle non-dict types (float, int, None, etc.)
        if not isinstance(d, dict):
            return ""
        name = (d or {}).get("object_identity") or (d or {}).get("detailed_caption") or ""
        return light_norm(str(name))
    
    internvl_df["canon_object"] = internvl_df["exact_gaze_object"].apply(extract_object_name)
    internvl_df["canon_conf"] = 100  # Default confidence
    
    return internvl_df


def merge_spatial_temporal_threshold(df, spatial_thresh=50, time_thresh=2.0):
    """Function to merge episodes based on spatial/temporal conditions only"""
    df = df.sort_values(by="start_time_seconds").reset_index(drop=True)


    episodes = []
    current_episode = [df.iloc[0]]

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]

        # spatial distance
        spatial_close = euclidean_distance(
            (prev["center_x"], prev["center_y"]),
            (curr["center_x"], curr["center_y"])
        ) < spatial_thresh

        # temporal distance
        temporal_close = (curr["start_time_seconds"] - prev["end_time_seconds"]) < time_thresh

        # merge rule: merge if spatial AND temporal conditions are met (regardless of object)
        if spatial_close and temporal_close:
            current_episode.append(curr)
        else:
            episodes.append(current_episode)
            current_episode = [curr]

    episodes.append(current_episode)

    merged = []
    for ep in episodes:
        ep_df = pd.DataFrame(ep)

        # Collect object_identity from all exact_gaze_object
        all_exact_objects = []
        for obj in ep_df["exact_gaze_object"]:
            if isinstance(obj, dict):
                all_exact_objects.append(obj)
            elif isinstance(obj, str):
                try:
                    obj_dict = ast.literal_eval(obj)
                    if isinstance(obj_dict, dict):
                        all_exact_objects.append(obj_dict)
                except:
                    continue

        # Find mode (most frequent) from exact_gaze_object
        if all_exact_objects:
            obj_identities = [obj.get("object_identity", "") for obj in all_exact_objects if obj.get("object_identity")]
            if obj_identities:
                counter = Counter(obj_identities)
                rep_obj_id = counter.most_common(1)[0][0]
                
                # Find representative dict (use first appearing rep_obj_id)
                rep_obj_dict = None
                for obj in all_exact_objects:
                    if obj.get("object_identity") == rep_obj_id:
                        rep_obj_dict = obj
                        break
            else:
                rep_obj_dict = None
        else:
            rep_obj_dict = None

        # Merge other objects + remove duplicates
        def merge_unique_object_lists(series):
            seen = set()
            merged_list = []
            for objs in series:
                if isinstance(objs, list):
                    for o in objs:
                        if isinstance(o, dict) and "object_identity" in o:
                            oid = o["object_identity"]
                            if oid not in seen:
                                seen.add(oid)
                                merged_list.append(o)
                elif isinstance(objs, str):
                    try:
                        objs_list = ast.literal_eval(objs)
                        if isinstance(objs_list, list):
                            for o in objs_list:
                                if isinstance(o, dict) and "object_identity" in o:
                                    oid = o["object_identity"]
                                    if oid not in seen:
                                        seen.add(oid)
                                        merged_list.append(o)
                    except:
                        continue
            return merged_list

        in_area = merge_unique_object_lists(ep_df["other_objects_in_cropped_area"])
        out_area = merge_unique_object_lists(ep_df["other_objects_outside_fov"])

        # Collect action_captions if available
        action_captions = []
        if "action_caption" in ep_df.columns:
            action_captions = list(ep_df["action_caption"])
        
        merged.append({
            "episode_start_time": ep_df["start_time_seconds"].min(),
            "episode_end_time": ep_df["end_time_seconds"].max(),
            "duration": ep_df["end_time_seconds"].max() - ep_df["start_time_seconds"].min(),
            "fixation_ids": list(ep_df["fixation_id"]),
            "representative_object": rep_obj_dict,
            "other_objects_in_cropped_area": in_area,
            "other_objects_outside_fov": out_area,
            "captions": list(ep_df["internvl_caption"]),
            "action_captions": action_captions,
        })

    return pd.DataFrame(merged), episodes


# =============================================================================
# Video Processing and GIF Generation
# =============================================================================

def extract_fixation_clips_to_gif(merged_df, video_name, video_path, output_dir, dataset='egtea', is_lab=False, gif_fps=8, scale=0.6, max_duration=15):
    """
    Extract fixation clips from video and save directly as GIFs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the gaze visualization video if available, otherwise use original video
    if dataset == 'egtea':
        gaze_viz_path = f'{os.path.join(PIPELINE_DIR, "final_data")}/egtea/metadata/{video_name}/{video_name}_gaze_visualization.mp4'
    elif dataset == 'ego4d':
        gaze_viz_path = f'{os.path.join(PIPELINE_DIR, "final_data")}/ego4d/metadata/{video_name}/{video_name}_gaze_visualization.mp4'
    elif dataset == 'egoexo':
        if is_lab:
            gaze_viz_path = f'{os.path.join(PIPELINE_DIR, "final_data")}/egoexo/metadata/lab/{video_name}/{video_name}_gaze_visualization.mp4'
        else:
            gaze_viz_path = f'{os.path.join(PIPELINE_DIR, "final_data")}/egoexo/metadata/{video_name}/{video_name}_gaze_visualization.mp4'
    elif dataset == 'kitchen':
        gaze_viz_path = f'{os.path.join(PIPELINE_DIR, "final_data")}/egoexo/metadata/kitchen_160/{video_name}/{video_name}_gaze_visualization.mp4'
    elif dataset == 'holoassist':
        gaze_viz_path = f'{os.path.join(PIPELINE_DIR, "final_data")}/holoassist/metadata/{video_name}/{video_name}_gaze_visualization.mp4'
    else:
        gaze_viz_path = None
    
    if gaze_viz_path and os.path.exists(gaze_viz_path):
        video_file = gaze_viz_path
        print(f"Using gaze visualization video: {gaze_viz_path}")
    elif os.path.exists(video_path):
        video_file = video_path
        print(f"Gaze visualization not found, using original video: {video_path}")
    else:
        print(f"⚠️  Error: Neither gaze visualization nor original video found for {video_name}")
        print(f"  - Gaze viz path: {gaze_viz_path}")
        print(f"  - Original path: {video_path}")
        print(f"Skipping GIF generation for {video_name}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # New dimensions after scaling
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    print(f"Video properties: {width}x{height}, {video_fps} FPS, {total_frames} frames")
    print(f"GIF output: {new_width}x{new_height}, {gif_fps} FPS")
    
    # Process each fixation with progress bar
    fixation_progress = tqdm(merged_df.iterrows(), 
                           total=len(merged_df), 
                           desc=f"Extracting GIFs for {video_name}", 
                           unit="fixation")
    
    for idx, row in fixation_progress:
        start_time = row['episode_start_time']
        end_time = row['episode_end_time']
        rep_obj = row['representative_object']
        duration = end_time - start_time
        
        # Get object name for display
        if isinstance(rep_obj, dict):
            obj_name = rep_obj.get('object_identity', 'unknown')
        else:
            obj_name = 'unknown'
        
        # Update progress bar description
        fixation_progress.set_description(f"Extracting {obj_name} ({start_time:.1f}s-{end_time:.1f}s)")
        
        # Calculate frame numbers
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        
        # Validate frame numbers
        if start_frame >= total_frames:
            fixation_progress.write(f"Warning: Start frame {start_frame} >= total frames {total_frames}, skipping")
            continue
            
        if end_frame > total_frames:
            fixation_progress.write(f"Warning: End frame {end_frame} > total frames {total_frames}, adjusting to {total_frames}")
            end_frame = total_frames
            
        if start_frame >= end_frame:
            fixation_progress.write(f"Warning: Invalid frame range {start_frame}-{end_frame}, skipping")
            continue
        
        # Create output filename
        safe_object_name = "".join(c for c in str(obj_name) if c.isalnum() or c in (' ', '-', '_')).rstrip()
        output_filename = f"episode_{idx:03d}_{safe_object_name}_{start_time:.1f}s-{end_time:.1f}s.gif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Calculate frame interval for target FPS
        frame_interval = max(1, int(video_fps / gif_fps))
        
        # Calculate max frames based on duration
        max_frames = int(max_duration * video_fps)
        frames_to_process = min(end_frame - start_frame, max_frames)
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames for GIF
        frames = []
        frame_count = 0
        actual_frames_extracted = 0
        
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                fixation_progress.write(f"Warning: Could not read frame {frame_num}")
                break
            
            # Take every frame_interval-th frame
            if frame_count % frame_interval == 0:
                # Resize frame
                resized_frame = cv2.resize(frame, (new_width, new_height))
                
                # Add enhanced frame info overlay
                frame_with_info = resized_frame.copy()
                
                # # Time info
                cv2.putText(frame_with_info, f"Time: {start_time:.1f}s - {end_time:.1f}s", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Convert BGR to RGB for imageio
                rgb_frame = cv2.cvtColor(frame_with_info, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                actual_frames_extracted += 1
            
            frame_count += 1
            
            # Stop if we've reached max duration
            if actual_frames_extracted >= max_duration * gif_fps:
                break
        
        # Save as GIF
        if frames:
            try:
                imageio.mimsave(output_path, frames, fps=gif_fps, loop=0)
                fixation_progress.write(f"✓ GIF saved: {output_filename} ({len(frames)} frames)")
            except Exception as e:
                fixation_progress.write(f"✗ Error saving GIF {output_filename}: {e}")
        else:
            fixation_progress.write(f"✗ No frames extracted for {obj_name}")
    
    cap.release()
    fixation_progress.write(f"\nAll episode GIFs saved to: {output_dir}")

def should_merge(ep1, ep2, video_path, hist_thresh=0.9):
    """Check if two episodes should be merged based only on visual similarity"""
    obj1 = ep1.get('representative_object')
    obj2 = ep2.get('representative_object')
    
    # Don't merge if None or not dict
    if not isinstance(obj1, dict) or not isinstance(obj2, dict):
        return False
    
    # 2. Visual similarity check (from start of ep1 to end of ep2)
    is_consistent, min_score, mean_score, _ = check_visual_consistency_hist(
        video_path,
        ep1['episode_start_time'], ep2['episode_end_time'],
        n_samples=5, hist_threshold=hist_thresh
    )
    return is_consistent


def filter_consecutive_similar_episodes(merged_df, video_path, hist_thresh=0.9):
    """
    Remove consecutive similar episodes keeping only one (no label check)
    """
    # Sort by time
    df = merged_df.sort_values('episode_start_time').reset_index(drop=True)
    
    keep_indices = []
    skip_until_idx = -1
    
    print(f"Filtering consecutive similar episodes from {len(df)} episodes")
    print(f"Histogram threshold: {hist_thresh}\n")
    
    i = 0
    while i < len(df):
        if i <= skip_until_idx:
            i += 1
            continue
            
        # Start with current episode as representative
        current_group = [i]
        representative = df.iloc[i]
        
        # Find consecutive similar episodes
        j = i + 1
        while j < len(df):
            next_ep = df.iloc[j]
            
            # Compare with current representative episode
            if should_merge(representative, next_ep, video_path, hist_thresh):
                current_group.append(j)
                # Safely extract object identity
                rep_obj = representative.get('representative_object', {})
                rep_obj_id = rep_obj.get('object_identity', 'N/A') if isinstance(rep_obj, dict) else 'N/A'
                next_obj = next_ep.get('representative_object', {})
                next_obj_id = next_obj.get('object_identity', 'N/A') if isinstance(next_obj, dict) else 'N/A'
                print(f"  📍 Episode {i} ({rep_obj_id}) "
                      f"~~ Episode {j} ({next_obj_id}) "
                      f"-> similar, will keep only first")
                j += 1
            else:
                break
        
        # Keep only first episode (uncomment below to keep longest)
        keep_indices.append(i)
        
        # Or to keep the longest episode:
        # longest_idx = max(current_group, key=lambda idx: df.iloc[idx]['duration'])
        # keep_indices.append(longest_idx)
        
        if len(current_group) > 1:
            print(f"  ✅ Kept episode {i}, removed {len(current_group)-1} similar episode(s)\n")
        
        skip_until_idx = j - 1
        i = j
    
    filtered_df = df.iloc[keep_indices].reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"✅ Final: {len(filtered_df)}/{len(df)} episodes kept")
    print(f"❌ Removed: {len(df) - len(filtered_df)} consecutive similar episodes")
    print(f"{'='*60}")
    
    return filtered_df


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_single_video(video_name, base_data_path, video_base_path, dataset='egtea'):
    """Process a single video and return results"""
    print(f"\n{'='*60}")
    print(f"Processing video: {video_name}")
    print(f"{'='*60}")
    
    # Check if required files exist
    # HoloAssist and EgoExoLearn have different video path structures
    if dataset == 'holoassist':
        video_path = f'{video_base_path}{video_name}/Export_py/Video_pitchshift.mp4'
    elif dataset == 'egoexo':
        video_path = f'{video_base_path}{video_name}.mp4'
    else:
        video_path = f'{video_base_path}{video_name}.mp4'
    data_dir = f'{base_data_path}{video_name}/'
    
    required_files = [
        f'{video_name}_fixation_with_internvl_v2_scene.csv'  # Only need this file
    ]
    
    # Check if all required files exist
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files for {video_name}: {missing_files}")
        return None
    
    # Load datasets
    internvl = load_csv_safely(f'{data_dir}{video_name}_fixation_with_internvl_v2_scene.csv')
    
    if internvl is None:
        print(f"Failed to load internvl dataset for {video_name}")
        return None
    
    return {
        'video_name': video_name,
        'video_path': video_path,
        'internvl': internvl,
        'dataset': dataset
    }


def process_video_complete(video_data, output_base_path, no_viz=False, is_lab=False):
    """Complete processing pipeline for a single video"""
    video_name = video_data['video_name']
    video_path = video_data['video_path']
    internvl_df = video_data['internvl']
    dataset = video_data.get('dataset', 'egtea')
    
    print(f"\nProcessing {video_name}...")
    
    # Check if output files already exist
    output_data_dir = f'{output_base_path}{video_name}'
    merged_output_path = f'{output_data_dir}/{video_name}_fixation_merged_filtered_v2.csv'
    gif_output_dir = f'{output_data_dir}/gaze_video_chunk/fixation_filtered_v2'
    
    # Check if both CSV and GIF files exist
    csv_exists = os.path.exists(merged_output_path)
    gif_dir_exists = os.path.exists(gif_output_dir) and os.path.isdir(gif_output_dir)
    gif_files_exist = gif_dir_exists and len([f for f in os.listdir(gif_output_dir) if f.endswith('.gif')]) > 0
    
    # Skip if both CSV and GIF files already exist
    if csv_exists and gif_files_exist and not no_viz:
        gif_count = len([f for f in os.listdir(gif_output_dir) if f.endswith('.gif')])
        print(f"✓ Output files already exist:")
        print(f"  - CSV: {merged_output_path}")
        print(f"  - GIFs: {gif_output_dir} ({gif_count} files)")
        print(f"Skipping processing for {video_name}")
        return {
            'video_name': video_name,
            'num_episodes': 'skipped',
            'output_dir': gif_output_dir,
            'data_dir': output_data_dir
        }
    elif csv_exists and not gif_files_exist:
        print(f"⚠ CSV exists but GIFs missing, will generate GIFs for {video_name}")
    elif not csv_exists and gif_files_exist:
        print(f"⚠ GIFs exist but CSV missing, will regenerate CSV for {video_name}")
    else:
        print(f"Processing {video_name} (no existing files found)")
    
    # Step 1: Process internvl data (simplified)
    processed_internvl = process_internvl_data_simple(internvl_df, video_name, dataset)
    print(f"Loaded {len(processed_internvl)} fixations from internvl data")
    
    # Step 2: Merge fixations (spatial + temporal)
    merged_df, episodes = merge_spatial_temporal_threshold(
        processed_internvl,
        spatial_thresh=50,
        time_thresh=2.0
    )
    
    print(f"Found {len(merged_df)} merged episodes for {video_name}")
    
    # Step 3: Filter consecutive similar episodes
    merged_df = filter_consecutive_similar_episodes(
        merged_df, 
        video_path,
        hist_thresh=0.70
    )

    # Step 5: Generate GIFs with enhanced visualization
    if not no_viz:
        # Check if we need to generate GIFs (not already existing)
        if not gif_files_exist:
            print(f"Extracting episode clips to: {gif_output_dir}")
            
            extract_fixation_clips_to_gif(
                merged_df=merged_df,
                video_name=video_name,
                video_path=video_path,
                output_dir=gif_output_dir,
                dataset=dataset,     # Dataset type (egtea or ego4d)
                is_lab=is_lab,       # Whether this is egoexo-lab dataset
                gif_fps=8,           # GIF FPS
                scale=0.6,           # 60% of original size
                max_duration=15      # Max 15 seconds per GIF
            )
        else:
            print(f"✓ GIFs already exist, skipping generation for {video_name}")
    else:
        print(f"Skipping GIF generation (--no-viz)")
    
    # Step 5: Save processed data
    os.makedirs(output_data_dir, exist_ok=True)
    
    # Save merged episodes with _fixation_merged suffix
    merged_df.to_csv(merged_output_path, index=False)
    print(f"Saved merged episodes to: {merged_output_path}")
    
    return {
        'video_name': video_name,
        'num_episodes': len(merged_df),
        'output_dir': gif_output_dir,
        'data_dir': output_data_dir
    }


# =============================================================================
# Configuration and Constants
# =============================================================================

def process_egtea(reverse=False, no_viz=False):
    """Process EGTEA dataset"""
    print("🚀 Starting EGTEA Sequence Filtering")
    print("=" * 60)
    
    # Base paths for EGTEA
    BASE_DATA_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'egtea', 'metadata') + '/'
    VIDEO_BASE_PATH = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egtea', 'videos') + '/'
    OUTPUT_BASE_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'egtea', 'metadata') + '/'
    
    # Get all video tasks
    tasks = sorted([task for task in os.listdir(BASE_DATA_PATH) 
             if os.path.isdir(os.path.join(BASE_DATA_PATH, task))])
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")
    
    print(f"Found {len(tasks)} video tasks to process")
    print(f"Starting batch processing of {len(tasks)} videos...")
    
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    # Use tqdm for progress tracking
    progress_bar = tqdm(tasks, desc="Processing videos", unit="video")
    
    for video_name in progress_bar:
        progress_bar.set_description(f"Processing {video_name}")
        
        # Check if already processed
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_merged_filtered_v2.csv'
        if os.path.exists(output_check):
            skipped += 1
            progress_bar.write(f"⏭️  SKIPPING: {video_name} (already processed)")
            progress_bar.set_postfix({
                'Success': successful,
                'Skipped': skipped,
                'Failed': failed
            })
            continue
        
        try:
            # Load video data
            video_data = process_single_video(video_name, BASE_DATA_PATH, VIDEO_BASE_PATH, dataset='egtea')
            if video_data is None:
                progress_bar.write(f"Skipping {video_name} due to missing data")
                failed += 1
                continue
            
            # Process video completely
            result = process_video_complete(video_data, OUTPUT_BASE_PATH, no_viz=no_viz)
            results.append(result)
            
            if result['num_episodes'] == 'skipped':
                skipped += 1
                progress_bar.write(f"⏭ Skipped {video_name} (already processed)")
            else:
                successful += 1
                progress_bar.write(f"✓ Successfully processed {video_name}")
            
        except Exception as e:
            progress_bar.write(f"✗ Error processing {video_name}: {e}")
            failed += 1
            continue
        
        # Update progress bar with current stats
        progress_bar.set_postfix({
            'Success': successful,
            'Skipped': skipped,
            'Failed': failed,
            'Success Rate': f"{successful/(successful+skipped+failed)*100:.1f}%" if (successful+skipped+failed) > 0 else "0%"
        }) 
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EGTEA BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    if len(tasks) > 0:
        print(f"Success rate: {successful/len(tasks)*100:.1f}%")
        print(f"Skip rate: {skipped/len(tasks)*100:.1f}%")
    
    return results


def process_egoexo(reverse=False, no_viz=False):
    """Process EgoExoLearn dataset"""
    print("🚀 Starting EgoExoLearn Sequence Filtering")
    print("=" * 60)
    
    # Base paths for EgoExoLearn
    BASE_DATA_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'metadata') + '/'
    VIDEO_BASE_PATH = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egoexolearn', 'full') + '/'
    OUTPUT_BASE_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'metadata') + '/'
    
    # Get all video tasks
    tasks = sorted([task for task in os.listdir(BASE_DATA_PATH) 
             if os.path.isdir(os.path.join(BASE_DATA_PATH, task))])
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")
    
    print(f"Found {len(tasks)} video tasks to process")
    print(f"Starting batch processing of {len(tasks)} videos...")
    
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    # Use tqdm for progress tracking
    progress_bar = tqdm(tasks, desc="Processing videos", unit="video")
    
    for video_name in progress_bar:
        progress_bar.set_description(f"Processing {video_name}")
        
        # Check if already processed
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_merged_filtered_v2.csv'
        if os.path.exists(output_check):
            skipped += 1
            progress_bar.write(f"⏭️  SKIPPING: {video_name} (already processed)")
            progress_bar.set_postfix({
                'Success': successful,
                'Skipped': skipped,
                'Failed': failed
            })
            continue
        
        try:
            # Load video data
            video_data = process_single_video(video_name, BASE_DATA_PATH, VIDEO_BASE_PATH, dataset='egoexo')
            if video_data is None:
                progress_bar.write(f"Skipping {video_name} due to missing data")
                failed += 1
                continue
            
            # Process video completely
            result = process_video_complete(video_data, OUTPUT_BASE_PATH, no_viz=no_viz)
            results.append(result)
            
            if result['num_episodes'] == 'skipped':
                skipped += 1
                progress_bar.write(f"⏭ Skipped {video_name} (already processed)")
            else:
                successful += 1
                progress_bar.write(f"✓ Successfully processed {video_name}")
            
        except Exception as e:
            progress_bar.write(f"✗ Error processing {video_name}: {e}")
            failed += 1
            continue
        
        # Update progress bar with current stats
        progress_bar.set_postfix({
            'Success': successful,
            'Skipped': skipped,
            'Failed': failed,
            'Success Rate': f"{successful/(successful+skipped+failed)*100:.1f}%" if (successful+skipped+failed) > 0 else "0%"
        }) 
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EGOEXOLEARN BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    if len(tasks) > 0:
        print(f"Success rate: {successful/len(tasks)*100:.1f}%")
        print(f"Skip rate: {skipped/len(tasks)*100:.1f}%")
    
    return results


def process_egoexo_lab(reverse=False, no_viz=False):
    """Process EgoExoLearn Lab dataset"""
    print("🚀 Starting EgoExoLearn Lab Sequence Filtering")
    print("=" * 60)
    
    # Base paths for EgoExoLearn Lab
    BASE_DATA_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'metadata', 'lab') + '/'
    VIDEO_BASE_PATH = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egoexolearn', 'full') + '/'
    OUTPUT_BASE_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'metadata', 'lab') + '/'
    
    # Get all video tasks
    tasks = sorted([task for task in os.listdir(BASE_DATA_PATH) 
             if os.path.isdir(os.path.join(BASE_DATA_PATH, task))])
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")
    
    print(f"Found {len(tasks)} lab video tasks to process")
    print(f"Starting batch processing of {len(tasks)} videos...")
    
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    # Use tqdm for progress tracking
    progress_bar = tqdm(tasks, desc="Processing videos", unit="video")
    
    for video_name in progress_bar:
        progress_bar.set_description(f"Processing {video_name}")
        
        # Check if already processed
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_merged_filtered_v2.csv'
        if os.path.exists(output_check):
            skipped += 1
            progress_bar.write(f"⏭️  SKIPPING: {video_name} (already processed)")
            progress_bar.set_postfix({
                'Success': successful,
                'Skipped': skipped,
                'Failed': failed
            })
            continue
        
        try:
            # Load video data
            video_data = process_single_video(video_name, BASE_DATA_PATH, VIDEO_BASE_PATH, dataset='egoexo')
            if video_data is None:
                progress_bar.write(f"Skipping {video_name} due to missing data")
                failed += 1
                continue
            
            # Process video completely
            result = process_video_complete(video_data, OUTPUT_BASE_PATH, no_viz=no_viz, is_lab=True)
            results.append(result)
            
            if result['num_episodes'] == 'skipped':
                skipped += 1
                progress_bar.write(f"⏭ Skipped {video_name} (already processed)")
            else:
                successful += 1
                progress_bar.write(f"✓ Successfully processed {video_name}")
            
        except Exception as e:
            progress_bar.write(f"✗ Error processing {video_name}: {e}")
            failed += 1
            continue
        
        # Update progress bar with current stats
        progress_bar.set_postfix({
            'Success': successful,
            'Skipped': skipped,
            'Failed': failed,
            'Success Rate': f"{successful/(successful+skipped+failed)*100:.1f}%" if (successful+skipped+failed) > 0 else "0%"
        }) 
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EGOEXO LAB BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    if len(tasks) > 0:
        print(f"Success rate: {successful/len(tasks)*100:.1f}%")
        print(f"Skip rate: {skipped/len(tasks)*100:.1f}%")
    
    return results


def process_egoexo_kitchen(reverse=False, no_viz=False):
    """Process EgoExoLearn Kitchen dataset"""
    print("🚀 Starting EgoExoLearn Kitchen Sequence Filtering")
    print("=" * 60)
    
    # Base paths for EgoExoLearn Kitchen
    BASE_DATA_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'metadata', 'kitchen_160') + '/'
    VIDEO_BASE_PATH = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egoexolearn', 'full') + '/'
    OUTPUT_BASE_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'metadata', 'kitchen_160') + '/'
    
    # Get all video tasks
    tasks = sorted([task for task in os.listdir(BASE_DATA_PATH) 
             if os.path.isdir(os.path.join(BASE_DATA_PATH, task))])
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")
    
    print(f"Found {len(tasks)} kitchen video tasks to process")
    print(f"Starting batch processing of {len(tasks)} videos...")
    
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    # Use tqdm for progress tracking
    progress_bar = tqdm(tasks, desc="Processing videos", unit="video")
    
    for video_name in progress_bar:
        progress_bar.set_description(f"Processing {video_name}")
        
        # Check if already processed
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_merged_filtered_v2.csv'
        if os.path.exists(output_check):
            skipped += 1
            progress_bar.write(f"⏭️  SKIPPING: {video_name} (already processed)")
            progress_bar.set_postfix({
                'Success': successful,
                'Skipped': skipped,
                'Failed': failed
            })
            continue
        
        try:
            # Load video data
            video_data = process_single_video(video_name, BASE_DATA_PATH, VIDEO_BASE_PATH, dataset='kitchen')
            if video_data is None:
                progress_bar.write(f"Skipping {video_name} due to missing data")
                failed += 1
                continue
            
            # Process video completely
            result = process_video_complete(video_data, OUTPUT_BASE_PATH, no_viz=no_viz, is_lab=False)
            results.append(result)
            
            if result['num_episodes'] == 'skipped':
                skipped += 1
                progress_bar.write(f"⏭ Skipped {video_name} (already processed)")
            else:
                successful += 1
                progress_bar.write(f"✓ Successfully processed {video_name}")
            
        except Exception as e:
            progress_bar.write(f"✗ Error processing {video_name}: {e}")
            failed += 1
            continue
        
        # Update progress bar with current stats
        progress_bar.set_postfix({
            'Success': successful,
            'Skipped': skipped,
            'Failed': failed,
            'Success Rate': f"{successful/(successful+skipped+failed)*100:.1f}%" if (successful+skipped+failed) > 0 else "0%"
        }) 
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EGOEXO KITCHEN BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    if len(tasks) > 0:
        print(f"Success rate: {successful/len(tasks)*100:.1f}%")
        print(f"Skip rate: {skipped/len(tasks)*100:.1f}%")
    
    return results


def process_holoassist(reverse=False, no_viz=False):
    """Process HoloAssist dataset"""
    print("🚀 Starting HoloAssist Sequence Filtering")
    print("=" * 60)
    
    # Base paths for HoloAssist
    BASE_DATA_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'holoassist', 'metadata') + '/'
    VIDEO_BASE_PATH = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'holoassist', 'full') + '/'
    OUTPUT_BASE_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'holoassist', 'metadata') + '/'
    
    # Get all video tasks
    tasks = sorted([task for task in os.listdir(BASE_DATA_PATH) 
             if os.path.isdir(os.path.join(BASE_DATA_PATH, task))])
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")
    
    print(f"Found {len(tasks)} video tasks to process")
    print(f"Starting batch processing of {len(tasks)} videos...")
    
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    # Use tqdm for progress tracking
    progress_bar = tqdm(tasks, desc="Processing videos", unit="video")
    
    for video_name in progress_bar:
        progress_bar.set_description(f"Processing {video_name}")
        
        # Check if already processed
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_merged_filtered_v2.csv'
        if os.path.exists(output_check):
            skipped += 1
            progress_bar.write(f"⏭️  SKIPPING: {video_name} (already processed)")
            progress_bar.set_postfix({
                'Success': successful,
                'Skipped': skipped,
                'Failed': failed
            })
            continue
        
        try:
            # Load video data
            video_data = process_single_video(video_name, BASE_DATA_PATH, VIDEO_BASE_PATH, dataset='holoassist')
            if video_data is None:
                progress_bar.write(f"Skipping {video_name} due to missing data")
                failed += 1
                continue
            
            # Process video completely
            result = process_video_complete(video_data, OUTPUT_BASE_PATH, no_viz=no_viz)
            results.append(result)
            
            if result['num_episodes'] == 'skipped':
                skipped += 1
                progress_bar.write(f"⏭ Skipped {video_name} (already processed)")
            else:
                successful += 1
                progress_bar.write(f"✓ Successfully processed {video_name}")
            
        except Exception as e:
            progress_bar.write(f"✗ Error processing {video_name}: {e}")
            failed += 1
            continue
        
        # Update progress bar with current stats
        progress_bar.set_postfix({
            'Success': successful,
            'Skipped': skipped,
            'Failed': failed,
            'Success Rate': f"{successful/(successful+skipped+failed)*100:.1f}%" if (successful+skipped+failed) > 0 else "0%"
        }) 
    
    # Summary
    print(f"\n{'='*60}")
    print(f"HOLOASSIST BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    if len(tasks) > 0:
        print(f"Success rate: {successful/len(tasks)*100:.1f}%")
        print(f"Skip rate: {skipped/len(tasks)*100:.1f}%")
    
    return results


def process_ego4d(reverse=False, no_viz=False):
    """Process Ego4D dataset"""
    print("🚀 Starting Ego4D Sequence Filtering")
    print("=" * 60)
    
    # Base paths for Ego4D
    BASE_DATA_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'ego4d', 'metadata') + '/'
    VIDEO_BASE_PATH = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'ego4d', 'v2', 'gaze_videos', 'v2', 'full_scale') + '/'
    OUTPUT_BASE_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'ego4d', 'metadata') + '/'
    
    # Get all video tasks
    tasks = sorted([task for task in os.listdir(BASE_DATA_PATH) 
             if os.path.isdir(os.path.join(BASE_DATA_PATH, task))])
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")
    
    print(f"Found {len(tasks)} video tasks to process")
    print(f"Starting batch processing of {len(tasks)} videos...")
    
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    # Use tqdm for progress tracking
    progress_bar = tqdm(tasks, desc="Processing videos", unit="video")
    
    for video_name in progress_bar:
        progress_bar.set_description(f"Processing {video_name}")
        
        # Check if already processed
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_merged_filtered_v2.csv'
        if os.path.exists(output_check):
            skipped += 1
            progress_bar.write(f"⏭️  SKIPPING: {video_name} (already processed)")
            progress_bar.set_postfix({
                'Success': successful,
                'Skipped': skipped,
                'Failed': failed
            })
            continue
        
        try:
            # Load video data
            video_data = process_single_video(video_name, BASE_DATA_PATH, VIDEO_BASE_PATH, dataset='ego4d')
            if video_data is None:
                progress_bar.write(f"Skipping {video_name} due to missing data")
                failed += 1
                continue
            
            # Process video completely
            result = process_video_complete(video_data, OUTPUT_BASE_PATH, no_viz=no_viz)
            results.append(result)
            
            if result['num_episodes'] == 'skipped':
                skipped += 1
                progress_bar.write(f"⏭ Skipped {video_name} (already processed)")
            else:
                successful += 1
                progress_bar.write(f"✓ Successfully processed {video_name}")
            
        except Exception as e:
            progress_bar.write(f"✗ Error processing {video_name}: {e}")
            failed += 1
            continue
        
        # Update progress bar with current stats
        progress_bar.set_postfix({
            'Success': successful,
            'Skipped': skipped,
            'Failed': failed,
            'Success Rate': f"{successful/(successful+skipped+failed)*100:.1f}%" if (successful+skipped+failed) > 0 else "0%"
        }) 
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EGO4D BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    if len(tasks) > 0:
        print(f"Success rate: {successful/len(tasks)*100:.1f}%")
        print(f"Skip rate: {skipped/len(tasks)*100:.1f}%")
    
    return results


def main():
    """Main execution function with dataset selection"""
    parser = argparse.ArgumentParser(description='Sequence filtering: merge and filter consecutive similar episodes')
    parser.add_argument('--dataset', type=str, choices=['egtea', 'ego4d', 'egoexo', 'egoexo-lab', 'kitchen', 'holoassist'], required=True,
                        help='Dataset to process: egtea, ego4d, egoexo, egoexo-lab, kitchen, or holoassist')
    parser.add_argument('--reverse', action='store_true',
                        help='Process videos in reverse order (useful for parallel processing)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip GIF generation for faster processing')
    
    args = parser.parse_args()
    
    if args.dataset == 'egtea':
        results = process_egtea(reverse=args.reverse, no_viz=args.no_viz)
    elif args.dataset == 'ego4d':
        results = process_ego4d(reverse=args.reverse, no_viz=args.no_viz)
    elif args.dataset == 'egoexo':
        results = process_egoexo(reverse=args.reverse, no_viz=args.no_viz)
    elif args.dataset == 'egoexo-lab':
        results = process_egoexo_lab(reverse=args.reverse, no_viz=args.no_viz)
    elif args.dataset == 'kitchen':
        results = process_egoexo_kitchen(reverse=args.reverse, no_viz=args.no_viz)
    elif args.dataset == 'holoassist':
        results = process_holoassist(reverse=args.reverse, no_viz=args.no_viz)
    else:
        print(f"Unknown dataset: {args.dataset}")
        print("Please choose 'egtea', 'ego4d', 'egoexo', 'egoexo-lab', 'kitchen', or 'holoassist'")
        return None
    
    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Process all videos in the metadata directory
    results = main()
    
    if results is not None:
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        
        if results:
            successful = len([r for r in results if r is not None and r['num_episodes'] != 'skipped'])
            skipped = len([r for r in results if r is not None and r['num_episodes'] == 'skipped'])
            failed = len(results) - successful - skipped
            print(f"✅ Successfully processed: {successful} videos")
            print(f"⏭ Skipped (already exists): {skipped} videos")
            print(f"❌ Failed: {failed} videos")
            
            if successful > 0:
                print(f"\n📁 Sample output files:")
                for result in results[:3]:  # Show first 3 results
                    if result and result['num_episodes'] != 'skipped':
                        print(f"  - {result['data_dir']}/{result['video_name']}_fixation_merged_filtered_v2.csv")
                
                # Merge all video results into a single total_metadata.csv
                print(f"\n{'='*60}")
                print("📊 Merging all video results into total_metadata.csv...")
                print(f"{'='*60}")
                merge_all_metadata(dataset, BASE_DATA_PATH)
        else:
            print("❌ No videos were processed")
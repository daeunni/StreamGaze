"""
Step 1.5: Fixation Filtering and Merging (Before Object Extraction)
Filters and merges fixations from Step 1 to create high-quality episodes
This runs BEFORE Step 2 (InternVL object extraction)

Usage:
    python step1.5_filtering_fixation.py --dataset egtea
    python step1.5_filtering_fixation.py --dataset ego4d
    python step1.5_filtering_fixation.py --dataset egoexo
    python step1.5_filtering_fixation.py --dataset holoassist
    python step1.5_filtering_fixation.py --dataset ego4d --reverse  # Process in reverse order

Features:
    - Scene consistency filtering (histogram-based)
    - Black screen removal
    - Duration filtering (>= 2.5 seconds)
    - Works with fixation_dataset.csv (NO object info needed)
    - Automatically skips already processed videos
"""

import os
import ast
import cv2
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
    
    if dataset in ['ego4d', 'egoexo']:
        # Ego4D and EgoExoLearn use 'start_time', 'end_time'
        # Rename to standard names for internal processing
        if 'start_time' in df.columns and 'start_time_seconds' not in df.columns:
            df['start_time_seconds'] = df['start_time']
        if 'end_time' in df.columns and 'end_time_seconds' not in df.columns:
            df['end_time_seconds'] = df['end_time']
    
    # Add fixation_id if not present
    if 'fixation_id' not in df.columns:
        df['fixation_id'] = range(len(df))
    
    return df


def process_fixation_data_simple(fixation_df, video_name, dataset='egtea'):
    """Process fixation data for a single video (works WITHOUT object info)"""
    print(f"Processing fixation data for {video_name}")
    
    # Normalize column names first
    fixation_df = normalize_column_names(fixation_df, dataset)
    
    # Add placeholder columns if they don't exist (for compatibility)
    fixation_df = fixation_df.copy()
    
    if 'exact_gaze_object' not in fixation_df.columns:
        fixation_df["exact_gaze_object"] = [{}] * len(fixation_df)
    else:
        # Parse if string
        fixation_df["exact_gaze_object"] = fixation_df["exact_gaze_object"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, dict) else {})
        )
    
    if 'other_objects_in_cropped_area' not in fixation_df.columns:
        fixation_df["other_objects_in_cropped_area"] = [[] for _ in range(len(fixation_df))]
    
    if 'other_objects_outside_fov' not in fixation_df.columns:
        fixation_df["other_objects_outside_fov"] = [[] for _ in range(len(fixation_df))]
    
    if 'internvl_caption' not in fixation_df.columns:
        fixation_df["internvl_caption"] = [""] * len(fixation_df)
    
    return fixation_df


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

        merged.append({
            "episode_start_time": ep_df["start_time_seconds"].min(),
            "episode_end_time": ep_df["end_time_seconds"].max(),
            "duration": ep_df["end_time_seconds"].max() - ep_df["start_time_seconds"].min(),
            "fixation_ids": list(ep_df["fixation_id"]),
            "representative_object": rep_obj_dict,
            "other_objects_in_cropped_area": in_area,
            "other_objects_outside_fov": out_area,
            "captions": list(ep_df["internvl_caption"]),
        })

    return pd.DataFrame(merged), episodes


# =============================================================================
# Episode Filtering Functions
# =============================================================================

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
                print(f"  📍 Episode {i} ({representative.get('representative_object', {}).get('object_identity', 'N/A')}) "
                      f"~~ Episode {j} ({next_ep.get('representative_object', {}).get('object_identity', 'N/A')}) "
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
    """Process a single video and return results (Step 1.5: uses fixation_dataset.csv)"""
    print(f"\n{'='*60}")
    print(f"Processing video: {video_name}")
    print(f"{'='*60}")
    
    # Check if required files exist
    # HoloAssist has different video path structure
    if dataset == 'holoassist':
        video_path = f'{video_base_path}{video_name}/Export_py/Video_pitchshift.mp4'
    else:
        video_path = f'{video_base_path}{video_name}.mp4'
    data_dir = f'{base_data_path}{video_name}/'
    
    required_files = [
        f'{video_name}_fixation_dataset.csv'  # Step 1 output (NO object info)
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
    fixation_df = load_csv_safely(f'{data_dir}{video_name}_fixation_dataset.csv')
    
    if fixation_df is None:
        print(f"Failed to load fixation dataset for {video_name}")
        return None
    
    return {
        'video_name': video_name,
        'video_path': video_path,
        'fixation_df': fixation_df,
        'dataset': dataset
    }


def process_video_complete(video_data, output_base_path):
    """Complete processing pipeline for a single video"""
    video_name = video_data['video_name']
    video_path = video_data['video_path']
    fixation_df = video_data['fixation_df']
    dataset = video_data.get('dataset', 'egtea')
    
    print(f"\nProcessing {video_name}...")
    
    # Check if output files already exist
    output_data_dir = f'{output_base_path}{video_name}'
    merged_output_path = f'{output_data_dir}/{video_name}_fixation_filtered.csv'  # Step 1.5 output
    
    # Check if output file exists
    if os.path.exists(merged_output_path):
        print(f"⏭️ Output file already exists for {video_name}")
        return {
            'video_name': video_name,
            'num_episodes': 'skipped',
            'data_dir': output_data_dir
        }
    
    # Step 1: Normalize column names (maintain Step1 columns)
    fixation_df = normalize_column_names(fixation_df, dataset)
    
    print(f"Original fixation count: {len(fixation_df)}")
    print(f"Columns: {fixation_df.columns.tolist()}")

    # Step 2: Filter by scene consistency using histogram
    # HoloAssist: use more relaxed threshold (0.5) for scene changes
    hist_thresh = 0.5 if dataset == 'holoassist' else 0.9
    
    filtered_df, filter_stats = filter_scene_consistency_by_hist(
        fixation_df,
        video_path,
        hist_threshold=hist_thresh,
        min_duration=1.5,
    )
    
    print(f"After scene consistency filtering: {len(filtered_df)}/{len(fixation_df)} fixations remain")
    
    # Step 3: Filter by minimum duration (>= 2.5 seconds)
    min_duration = 2.5
    duration_before = len(filtered_df)
    if 'duration_seconds' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['duration_seconds'] >= min_duration].reset_index(drop=True)
    else:
        # Calculate duration if not present
        filtered_df['duration_seconds'] = filtered_df['end_time_seconds'] - filtered_df['start_time_seconds']
        filtered_df = filtered_df[filtered_df['duration_seconds'] >= min_duration].reset_index(drop=True)
    duration_after = len(filtered_df)
    print(f"Duration filtering (>= {min_duration}s): {duration_before} → {duration_after} fixations ({duration_after/duration_before*100:.1f}% kept)")
    
    # Step 4: Save filtered fixations (KEEP Step1 columns including action_caption)
    os.makedirs(output_data_dir, exist_ok=True)
    
    # Ensure we keep the original columns from Step1
    # segment_id, start_time_seconds, end_time_seconds, duration_seconds, center_x, center_y, action_caption
    filtered_df.to_csv(merged_output_path, index=False)
    print(f"Saved filtered fixations to: {merged_output_path}")
    print(f"Columns saved: {filtered_df.columns.tolist()}")
    
    # Save filter statistics
    filter_stats_path = f'{output_data_dir}/{video_name}_filter_stats.csv'
    filter_stats.to_csv(filter_stats_path, index=False)
    print(f"Saved filter statistics to: {filter_stats_path}")
    
    return {
        'video_name': video_name,
        'num_fixations': len(filtered_df),
        'data_dir': output_data_dir
    }


# =============================================================================
# Configuration and Constants
# =============================================================================

def process_egtea(reverse=False):
    """Process EGTEA dataset"""
    print("🚀 Starting EGTEA Fixation Filtering and Merging")
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
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_filtered.csv'
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
            result = process_video_complete(video_data, OUTPUT_BASE_PATH)
            results.append(result)
            
            if result.get('num_fixations') == 'skipped' or result.get('num_episodes') == 'skipped':
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


def process_ego4d(reverse=False):
    """Process Ego4D dataset"""
    print("🚀 Starting Ego4D Fixation Filtering")
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
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_filtered.csv'
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
            result = process_video_complete(video_data, OUTPUT_BASE_PATH)
            results.append(result)
            
            if result.get('num_fixations') == 'skipped' or result.get('num_episodes') == 'skipped':
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


def process_egoexo(reverse=False):
    """Process EgoExoLearn dataset"""
    print("🚀 Starting EgoExoLearn Fixation Filtering")
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
        
        # Check if already processed (CSV exists)
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_filtered.csv'
        
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
            result = process_video_complete(video_data, OUTPUT_BASE_PATH)
            results.append(result)
            
            if result.get('num_fixations') == 'skipped' or result.get('num_episodes') == 'skipped':
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


def process_egoexo_lab(reverse=False):
    """Process EgoExoLearn Lab dataset"""
    print("🚀 Starting EgoExoLearn Lab Fixation Filtering")
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
        
        # Check if already processed (CSV exists)
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_filtered.csv'
        
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
            result = process_video_complete(video_data, OUTPUT_BASE_PATH)
            results.append(result)
            
            if result.get('num_fixations') == 'skipped' or result.get('num_episodes') == 'skipped':
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


def process_egoexo_kitchen(reverse=False):
    """Process EgoExoLearn Kitchen dataset"""
    print("🚀 Starting EgoExoLearn Kitchen Fixation Filtering")
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
        
        # Check if already processed (CSV exists)
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_filtered.csv'
        
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
            result = process_video_complete(video_data, OUTPUT_BASE_PATH)
            results.append(result)
            
            if result.get('num_fixations') == 'skipped' or result.get('num_episodes') == 'skipped':
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


def process_holoassist(reverse=False):
    """Process HoloAssist dataset"""
    print("🚀 Starting HoloAssist Fixation Filtering")
    print("=" * 60)
    
    # Base paths for HoloAssist
    BASE_DATA_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'holoassist', 'metadata') + '/'
    VIDEO_BASE_PATH = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'holoassist', 'full') + '/'
    OUTPUT_BASE_PATH = os.path.join(PIPELINE_DIR, 'final_data', 'holoassist', 'metadata') + '/'
    
    # Load annotation data to filter sessions
    import json
    annotation_file = os.path.join(VIDEO_BASE_PATH, "data-annnotation-trainval-v1_1.json")
    annotated_video_names = set()
    if os.path.exists(annotation_file):
        print("Loading annotation data to filter sessions...")
        with open(annotation_file, 'r') as f:
            annotation_data = json.load(f)
        annotated_video_names = set([v.get('video_name') for v in annotation_data if 'video_name' in v])
        print(f"Found {len(annotated_video_names)} videos with annotations")
    
    # Get all video tasks (only those with annotations)
    all_tasks = sorted([task for task in os.listdir(BASE_DATA_PATH) 
             if os.path.isdir(os.path.join(BASE_DATA_PATH, task))])
    
    # Filter to only include annotated sessions
    tasks = [task for task in all_tasks if task in annotated_video_names]
    skipped_no_annotation = len(all_tasks) - len(tasks)
    
    if skipped_no_annotation > 0:
        print(f"⏭️  Filtered out {skipped_no_annotation} sessions without annotations")
    print(f"Processing {len(tasks)} sessions with annotations")
    
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
        
        # Check if already processed (CSV exists)
        output_check = f'{OUTPUT_BASE_PATH}{video_name}/{video_name}_fixation_filtered.csv'
        
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
            result = process_video_complete(video_data, OUTPUT_BASE_PATH)
            results.append(result)
            
            if result.get('num_fixations') == 'skipped' or result.get('num_episodes') == 'skipped':
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


def main():
    """Main execution function with dataset selection"""
    parser = argparse.ArgumentParser(description='Filter and merge fixations for different datasets')
    parser.add_argument('--dataset', type=str, choices=['egtea', 'ego4d', 'egoexo', 'egoexo-lab', 'kitchen', 'holoassist'], required=True,
                        help='Dataset to process: egtea, ego4d, egoexo, egoexo-lab, kitchen, or holoassist')
    parser.add_argument('--reverse', action='store_true',
                        help='Process videos in reverse order (useful for parallel processing)')
    
    args = parser.parse_args()
    
    if args.dataset == 'egtea':
        results = process_egtea(reverse=args.reverse)
    elif args.dataset == 'ego4d':
        results = process_ego4d(reverse=args.reverse)
    elif args.dataset == 'egoexo':
        results = process_egoexo(reverse=args.reverse)
    elif args.dataset == 'egoexo-lab':
        results = process_egoexo_lab(reverse=args.reverse)
    elif args.dataset == 'kitchen':
        results = process_egoexo_kitchen(reverse=args.reverse)
    elif args.dataset == 'holoassist':
        results = process_holoassist(reverse=args.reverse)
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
            successful = len([r for r in results if r is not None and 
                            r.get('num_fixations', r.get('num_episodes')) != 'skipped'])
            skipped = len([r for r in results if r is not None and 
                         r.get('num_fixations', r.get('num_episodes')) == 'skipped'])
            failed = len(results) - successful - skipped
            print(f"✅ Successfully processed: {successful} videos")
            print(f"⏭ Skipped (already exists): {skipped} videos")
            print(f"❌ Failed: {failed} videos")
            
            if successful > 0:
                print(f"\n📁 Sample output files:")
                for result in results[:3]:  # Show first 3 results
                    if result and result.get('num_fixations', result.get('num_episodes')) != 'skipped':
                        print(f"  - {result['data_dir']}/{result['video_name']}_fixation_filtered.csv")
        else:
            print("❌ No videos were processed")
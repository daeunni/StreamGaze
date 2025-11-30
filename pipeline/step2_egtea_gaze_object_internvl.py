
from io import BytesIO
from typing import Tuple, List, Dict
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# from moviepy.editor import VideoFileClip
from datetime import datetime
import pandas as pd
from PIL import Image
from tqdm import tqdm 
from collections import Counter
import argparse

# Import functions from preprocess module (original Gemini functions)
from preprocess import (
    save_frequency_analysis,
    count_total_fixations
)

# Import NEW InternVL functions
from preprocess.internvl_processor import (
    extract_objects_and_scene_from_video_clip_internvl_v2_sequential,  # NEW: Sequential version
    get_processor_v2
)

# Get pipeline directory dynamically
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

# Usage:
# CUDA_VISIBLE_DEVICES=0,1,2,3 python step2_egtea_gaze_object_internvl.py --dataset egtea
# CUDA_VISIBLE_DEVICES=0,1,2,3 python step2_egtea_gaze_object_internvl.py --dataset ego4d
# CUDA_VISIBLE_DEVICES=4,5,6,7 python step2_egtea_gaze_object_internvl.py --dataset egoexo
# CUDA_VISIBLE_DEVICES=0,1,2,3 python step2_egtea_gaze_object_internvl.py --dataset holoassist

print("✅ Enhanced function with scene caption loaded!")
print("✅ Multi-threaded function loaded!")
print("🚀 InternVL processor ready!")

# Initialize InternVL processor v2 (single instance for memory safety)
print("🔥 Initializing InternVL-38B model v2...")
import torch
import gc

# Clear any existing models from memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Initialize single processor instance (no multi-GPU loading)
internvl_processor_v2 = get_processor_v2()
print("✅ InternVL-38B model v2 loaded and ready!")
print(f"📍 GPU memory after model loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def process_egtea(reverse=False, start_pct=0):
    """Process EGTEA dataset with InternVL"""
    print("🚀 Starting EGTEA Object Extraction with InternVL")
    print("=" * 60)
    
    base_dir = os.path.join(PIPELINE_DIR, 'final_data', 'egtea', 'metadata')
    video_base_dir = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egtea', 'videos')
    tasks = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    # Apply start_pct slicing
    if start_pct > 0:
        start_idx = int(len(tasks) * start_pct / 100)
        tasks = tasks[start_idx:]
        print(f"🎯 Starting from {start_pct}% position (index {start_idx}/{len(tasks) + start_idx})")
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")


    # Count total fixations for ETA calculation
    print("🔍 Counting total fixations across all videos...")
    total_fixations, video_fixations = count_total_fixations(base_dir)
    print(f"📊 Total videos: {len(tasks)}")
    print(f"📊 Total fixations to process: {total_fixations}")
    print(f"⏱️ Estimated total time: {(total_fixations * 5) / 3600:.1f} hours")
    print("=" * 60)
    
    # Initialize counters for progress tracking
    processed_fixations = 0
    overall_start_time = time.time()
    
    # Process each video in the dataset
    for video_idx, video_name in enumerate(tqdm(tasks, desc="Processing videos", unit="video")):
        print(f"\n{'='*60}")
        print(f"Processing video: {video_name}")
        print(f"{'='*60}")
        
        # Check if already processed (EGTEA)
        output_csv_check = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
        if os.path.exists(output_csv_check):
            print(f"⏭️  SKIPPING: Already processed (found {os.path.basename(output_csv_check)})")
            continue
        
        try:
            # Try multiple approaches to read the fixation CSV file
            print(f"Loading fixation dataset...")
            csv_path = os.path.join(base_dir, video_name, f'{video_name}_fixation_filtered.csv')
            
            # Check if file exists first
            if not os.path.exists(csv_path):
                print(f"❌ Fixation filtered CSV not found: {csv_path}")
                print(f"⏭️  SKIPPING: {video_name} (run step1.5 first)")
                continue
            
            try:
                # First try: standard approach
                fixation_dataset = pd.read_csv(csv_path)
                print(f"✅ Fixation dataset loaded successfully with standard approach")
            except pd.errors.ParserError as e:
                print(f"⚠️ Standard CSV parsing failed: {e}")
                try:
                    # Second try: more flexible parsing
                    fixation_dataset = pd.read_csv(csv_path, 
                                                 sep=',',
                                                 quotechar='"',
                                                 skipinitialspace=True,
                                                 on_bad_lines='skip',
                                                 engine='python')
                    print(f"✅ Successfully parsed fixation dataset with flexible options")
                except Exception as e2:
                    print(f"⚠️ Flexible parsing also failed: {e2}")
                    # Third try: use error_bad_lines=False for older pandas versions
                    try:
                        fixation_dataset = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)
                        print(f"✅ Successfully parsed fixation dataset with error_bad_lines=False")
                    except Exception as e3:
                        print(f"❌ All fixation dataset parsing attempts failed: {e3}")
                        print(f"⏭️  SKIPPING: {video_name}")
                        continue

            
            # Get video path from metadata (EGTEA uses start_time_seconds, end_time_seconds)
            video_path = os.path.join(video_base_dir, f'{video_name}.mp4')

            # Process all rows with InternVL v2 (including two-stage analysis)
            print("🚀 Starting InternVL-38B v2 two-stage object extraction for ALL fixations...")
            print(f"Total fixations to process: {len(fixation_dataset)}")
            print("=" * 60)

            # Create a copy of the dataset with new columns (including internvl_caption)
            fixation_dataset_with_scene = fixation_dataset.copy()
            fixation_dataset_with_scene['exact_gaze_object'] = None
            fixation_dataset_with_scene['other_objects_in_cropped_area'] = None
            fixation_dataset_with_scene['other_objects_outside_fov'] = None
            fixation_dataset_with_scene['internvl_caption'] = None
            fixation_dataset_with_scene['processing_status'] = None
            fixation_dataset_with_scene['processing_error'] = None

            # Initialize object pool for this video
            object_pool = set()
            print(f"📦 Initialized empty object pool for {video_name}")

            video_start_time = time.time()
            current_video_fixations = len(fixation_dataset)
            
            # Prepare requests for multi-threaded processing
            requests_data = []
            for idx, row in fixation_dataset.iterrows():
                requests_data.append({
                    'video_path': video_path,
                    'gaze_x': row['center_x'],
                    'gaze_y': row['center_y'],
                    'start_time': row['start_time_seconds'],
                    'end_time': row['end_time_seconds'],
                    'request_id': f'fixation_{idx}'
                })
            
            # Process all fixations using InternVL v2 multi-threaded function
            print(f"🚀 Processing {len(requests_data)} fixations using InternVL v2 multi-threading...")
            
            # Check GPU availability and memory
            import torch
            if torch.cuda.is_available():
                print(f"🔥 GPU available: {torch.cuda.device_count()} devices")
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            else:
                print("❌ No GPU available, using CPU")
            
            # Use sequential processing (NO THREADING) to avoid OOM
            print("🚀 Using sequential processing (no threading) for 38B model safety...")
            
            # Calculate FOV radius based on camera HFOV and perifovea angle
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            HFOV_deg = 90.0  # Standard HFOV for EGTEA
            r_deg = 13.0     # Perifovea radius in degrees
            px_per_deg = frame_width / HFOV_deg
            fov_radius = int(r_deg * px_per_deg)
            
            print(f"   Video resolution: {frame_width}px width, HFOV: {HFOV_deg}°")
            print(f"   Using perifovea radius: {fov_radius} px (~{r_deg}°)")
            
            # Disable GIF saving (for performance improvement)
            # video_output_dir = os.path.join(base_dir, video_name, 'internvl_object_clips')
            
            results = extract_objects_and_scene_from_video_clip_internvl_v2_sequential(
                requests_data=requests_data,
                fov_radius=fov_radius,
                save_images=False,
                show_images=False,
                object_pool=list(object_pool) if object_pool else None,
                temperature=0.3,
                output_dir=None  # Disable GIF saving
            )

            print(results)
            
            # Update progress tracking
            processed_fixations += len(requests_data)
            
            # Save results to dataset and update object pool
            for idx, result in enumerate(results):
                if "error" in result:
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'error'
                    fixation_dataset_with_scene.at[idx, 'processing_error'] = result['error']
                else:
                    # Successfully processed - using new format
                    fixation_dataset_with_scene.at[idx, 'exact_gaze_object'] = result.get('exact_gaze_object', {})
                    fixation_dataset_with_scene.at[idx, 'other_objects_in_cropped_area'] = result.get('other_objects_in_cropped_area', [])
                    fixation_dataset_with_scene.at[idx, 'other_objects_outside_fov'] = result.get('other_objects_outside_fov', [])
                    fixation_dataset_with_scene.at[idx, 'internvl_caption'] = result.get('scene_caption', '')
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'success'
                    
                    # Update object pool with new objects found
                    if 'exact_gaze_object' in result and 'object_identity' in result['exact_gaze_object']:
                        object_pool.add(result['exact_gaze_object']['object_identity'].strip().lower())
                    
                    if 'other_objects_in_cropped_area' in result:
                        for obj in result['other_objects_in_cropped_area']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())
                    
                    if 'other_objects_outside_fov' in result:
                        for obj in result['other_objects_outside_fov']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())

            print("\n" + "=" * 60)
            print("🎉 ENHANCED PROCESSING COMPLETED!")
            print("=" * 60)

            # Save object pool to txt file
            object_pool_txt_path = os.path.join(base_dir, video_name, f'{video_name}_object_pool.txt')
            with open(object_pool_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"OBJECT POOL - {video_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total unique objects found: {len(object_pool)}\n\n")
                f.write("Object names (alphabetically sorted):\n")
                f.write("-" * 30 + "\n")
                for obj in sorted(object_pool):
                    f.write(f"{obj}\n")
            print(f"✅ Object pool saved to: {object_pool_txt_path}")
            print(f"📦 Final object pool size: {len(object_pool)} objects")

            # Count other_objects occurrences across all fixations
            print("📈 OTHER OBJECTS OCCURRENCE ANALYSIS")
            print("=" * 60)

            # Extract all other_objects from successful fixations (new format)
            other_object_counter = Counter()
            successful_rows = fixation_dataset_with_scene[fixation_dataset_with_scene['processing_status'] == 'success']

            for idx, row in successful_rows.iterrows():
                # Count objects in cropped area
                cropped_objects = row['other_objects_in_cropped_area']
                if cropped_objects and isinstance(cropped_objects, list):
                    for obj in cropped_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1
                
                # Count objects outside FOV
                outside_objects = row['other_objects_outside_fov']
                if outside_objects and isinstance(outside_objects, list):
                    for obj in outside_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1

            # Save frequency analysis results
            output_path = os.path.join(base_dir, video_name, f'{video_name}_object_frequency_analysis.txt')
            save_frequency_analysis(other_object_counter, output_path, video_name)
            print(f"\n✅ Frequency analysis results saved to: {output_path}")

            # Save the processed dataset with scene information
            output_csv = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
            fixation_dataset_with_scene.to_csv(output_csv, index=False)
            print(f"✅ Processed dataset saved to: {output_csv}")

        except Exception as e:
            print(f"❌ Error processing video {video_name}: {str(e)}")
            print(f"⏭️  SKIPPING: {video_name} and continuing to next video...")
            continue

    print("\n" + "=" * 60)
    print("🎉 ALL EGTEA VIDEOS PROCESSED!")
    print("=" * 60)


def process_ego4d(reverse=False, start_pct=0):
    """Process Ego4D dataset with InternVL"""
    print("🚀 Starting Ego4D Object Extraction with InternVL")
    print("=" * 60)
    
    base_dir = os.path.join(PIPELINE_DIR, 'final_data', 'ego4d', 'metadata')
    video_base_dir = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'ego4d', 'v2', 'gaze_videos', 'v2', 'full_scale')
    tasks = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    # Apply start_pct slicing
    if start_pct > 0:
        start_idx = int(len(tasks) * start_pct / 100)
        tasks = tasks[start_idx:]
        print(f"🎯 Starting from {start_pct}% position (index {start_idx}/{len(tasks) + start_idx})")
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")

    # Count total fixations for ETA calculation
    print("🔍 Counting total fixations across all videos...")
    total_fixations, video_fixations = count_total_fixations(base_dir)
    print(f"📊 Total videos: {len(tasks)}")
    print(f"📊 Total fixations to process: {total_fixations}")
    print(f"⏱️ Estimated total time: {(total_fixations * 5) / 3600:.1f} hours")
    print("=" * 60)
    
    # Initialize counters for progress tracking
    processed_fixations = 0
    overall_start_time = time.time()
    
    # Process each video in the dataset
    for video_idx, video_name in enumerate(tqdm(tasks, desc="Processing videos", unit="video")):
        print(f"\n{'='*60}")
        print(f"Processing video: {video_name}")
        print(f"{'='*60}")
        
        # Check if already processed (Ego4D)
        output_csv_check = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
        if os.path.exists(output_csv_check):
            print(f"⏭️  SKIPPING: Already processed (found {os.path.basename(output_csv_check)})")
            continue
        
        try:
            # Try multiple approaches to read the fixation CSV file
            print(f"Loading fixation dataset...")
            csv_path = os.path.join(base_dir, video_name, f'{video_name}_fixation_filtered.csv')
            
            # Check if file exists first
            if not os.path.exists(csv_path):
                print(f"❌ Fixation filtered CSV not found: {csv_path}")
                print(f"⏭️  SKIPPING: {video_name} (run step1.5 first)")
                continue
            
            try:
                # First try: standard approach
                fixation_dataset = pd.read_csv(csv_path)
                print(f"✅ Fixation dataset loaded successfully with standard approach")
            except pd.errors.ParserError as e:
                print(f"⚠️ Standard CSV parsing failed: {e}")
                try:
                    # Second try: more flexible parsing
                    fixation_dataset = pd.read_csv(csv_path, 
                                                 sep=',',
                                                 quotechar='"',
                                                 skipinitialspace=True,
                                                 on_bad_lines='skip',
                                                 engine='python')
                    print(f"✅ Successfully parsed fixation dataset with flexible options")
                except Exception as e2:
                    print(f"⚠️ Flexible parsing also failed: {e2}")
                    # Third try: use error_bad_lines=False for older pandas versions
                    try:
                        fixation_dataset = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)
                        print(f"✅ Successfully parsed fixation dataset with error_bad_lines=False")
                    except Exception as e3:
                        print(f"❌ All fixation dataset parsing attempts failed: {e3}")
                        print(f"⏭️  SKIPPING: {video_name}")
                        continue

            # Get video path (Ego4D uses start_time, end_time - different from EGTEA)
            video_path = os.path.join(video_base_dir, f'{video_name}.mp4')

            # Process all rows with InternVL v2 (including two-stage analysis)
            print("🚀 Starting InternVL-38B v2 two-stage object extraction for ALL fixations...")
            print(f"Total fixations to process: {len(fixation_dataset)}")
            print("=" * 60)

            # Create a copy of the dataset with new columns (including internvl_caption)
            fixation_dataset_with_scene = fixation_dataset.copy()
            fixation_dataset_with_scene['exact_gaze_object'] = None
            fixation_dataset_with_scene['other_objects_in_cropped_area'] = None
            fixation_dataset_with_scene['other_objects_outside_fov'] = None
            fixation_dataset_with_scene['internvl_caption'] = None
            fixation_dataset_with_scene['processing_status'] = None
            fixation_dataset_with_scene['processing_error'] = None

            # Initialize object pool for this video
            object_pool = set()
            print(f"📦 Initialized empty object pool for {video_name}")

            video_start_time = time.time()
            current_video_fixations = len(fixation_dataset)
            
            # Prepare requests for multi-threaded processing
            # NOTE: Ego4D uses 'start_time' and 'end_time' instead of 'start_time_seconds' and 'end_time_seconds'
            requests_data = []
            for idx, row in fixation_dataset.iterrows():
                requests_data.append({
                    'video_path': video_path,
                    'gaze_x': row['center_x'],
                    'gaze_y': row['center_y'],
                    'start_time': row['start_time'],  # Ego4D column name
                    'end_time': row['end_time'],      # Ego4D column name
                    'request_id': f'fixation_{idx}'
                })
            
            # Process all fixations using InternVL v2 multi-threaded function
            print(f"🚀 Processing {len(requests_data)} fixations using InternVL v2 multi-threading...")
            
            # Check GPU availability and memory
            import torch
            if torch.cuda.is_available():
                print(f"🔥 GPU available: {torch.cuda.device_count()} devices")
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            else:
                print("❌ No GPU available, using CPU")
            
            # Use sequential processing (NO THREADING) to avoid OOM
            print("🚀 Using sequential processing (no threading) for 38B model safety...")
            
            # Calculate FOV radius based on camera HFOV and perifovea angle
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            HFOV_deg = 90.0  # Standard HFOV for Ego4D
            r_deg = 13.0     # Perifovea radius in degrees
            px_per_deg = frame_width / HFOV_deg
            fov_radius = int(r_deg * px_per_deg)
            
            print(f"   Video resolution: {frame_width}px width, HFOV: {HFOV_deg}°")
            print(f"   Using perifovea radius: {fov_radius} px (~{r_deg}°)")
            
            results = extract_objects_and_scene_from_video_clip_internvl_v2_sequential(
                requests_data=requests_data,
                fov_radius=fov_radius,
                save_images=False,
                show_images=False,
                object_pool=list(object_pool) if object_pool else None,
                temperature=0.3,
                output_dir=None  # Disable GIF saving
            )

            print(results)
            
            # Update progress tracking
            processed_fixations += len(requests_data)
            
            # Save results to dataset and update object pool
            for idx, result in enumerate(results):
                if "error" in result:
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'error'
                    fixation_dataset_with_scene.at[idx, 'processing_error'] = result['error']
                else:
                    # Successfully processed - using new format
                    fixation_dataset_with_scene.at[idx, 'exact_gaze_object'] = result.get('exact_gaze_object', {})
                    fixation_dataset_with_scene.at[idx, 'other_objects_in_cropped_area'] = result.get('other_objects_in_cropped_area', [])
                    fixation_dataset_with_scene.at[idx, 'other_objects_outside_fov'] = result.get('other_objects_outside_fov', [])
                    fixation_dataset_with_scene.at[idx, 'internvl_caption'] = result.get('scene_caption', '')
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'success'
                    
                    # Update object pool with new objects found
                    if 'exact_gaze_object' in result and 'object_identity' in result['exact_gaze_object']:
                        object_pool.add(result['exact_gaze_object']['object_identity'].strip().lower())
                    
                    if 'other_objects_in_cropped_area' in result:
                        for obj in result['other_objects_in_cropped_area']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())
                    
                    if 'other_objects_outside_fov' in result:
                        for obj in result['other_objects_outside_fov']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())

            print("\n" + "=" * 60)
            print("🎉 ENHANCED PROCESSING COMPLETED!")
            print("=" * 60)

            # Save object pool to txt file
            object_pool_txt_path = os.path.join(base_dir, video_name, f'{video_name}_object_pool.txt')
            with open(object_pool_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"OBJECT POOL - {video_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total unique objects found: {len(object_pool)}\n\n")
                f.write("Object names (alphabetically sorted):\n")
                f.write("-" * 30 + "\n")
                for obj in sorted(object_pool):
                    f.write(f"{obj}\n")
            print(f"✅ Object pool saved to: {object_pool_txt_path}")
            print(f"📦 Final object pool size: {len(object_pool)} objects")

            # Count other_objects occurrences across all fixations
            print("📈 OTHER OBJECTS OCCURRENCE ANALYSIS")
            print("=" * 60)

            # Extract all other_objects from successful fixations (new format)
            other_object_counter = Counter()
            successful_rows = fixation_dataset_with_scene[fixation_dataset_with_scene['processing_status'] == 'success']

            for idx, row in successful_rows.iterrows():
                # Count objects in cropped area
                cropped_objects = row['other_objects_in_cropped_area']
                if cropped_objects and isinstance(cropped_objects, list):
                    for obj in cropped_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1
                
                # Count objects outside FOV
                outside_objects = row['other_objects_outside_fov']
                if outside_objects and isinstance(outside_objects, list):
                    for obj in outside_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1

            # Save frequency analysis results
            output_path = os.path.join(base_dir, video_name, f'{video_name}_object_frequency_analysis.txt')
            save_frequency_analysis(other_object_counter, output_path, video_name)
            print(f"\n✅ Frequency analysis results saved to: {output_path}")

            # Save the processed dataset with scene information
            output_csv = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
            fixation_dataset_with_scene.to_csv(output_csv, index=False)
            print(f"✅ Processed dataset saved to: {output_csv}")

        except Exception as e:
            print(f"❌ Error processing video {video_name}: {str(e)}")
            print(f"⏭️  SKIPPING: {video_name} and continuing to next video...")
            continue

    print("\n" + "=" * 60)
    print("🎉 ALL EGO4D VIDEOS PROCESSED!")
    print("=" * 60)


def process_egoexo(reverse=False, start_pct=0):
    """Process EgoExoLearn dataset with InternVL (with action_caption context)"""
    print("🚀 Starting EgoExoLearn Object Extraction with InternVL")
    print("=" * 60)
    
    base_dir = os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'metadata')
    video_base_dir = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egoexolearn', 'full')
    tasks = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    # Apply start_pct slicing
    if start_pct > 0:
        start_idx = int(len(tasks) * start_pct / 100)
        tasks = tasks[start_idx:]
        print(f"🎯 Starting from {start_pct}% position (index {start_idx}/{len(tasks) + start_idx})")
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")

    # Count total fixations for ETA calculation
    print("🔍 Counting total fixations across all videos...")
    total_fixations, video_fixations = count_total_fixations(base_dir)
    print(f"📊 Total videos: {len(tasks)}")
    print(f"📊 Total fixations to process: {total_fixations}")
    print(f"⏱️ Estimated total time: {(total_fixations * 5) / 3600:.1f} hours")
    print("=" * 60)
    
    # Initialize counters for progress tracking
    processed_fixations = 0
    overall_start_time = time.time()
    
    # Process each video in the dataset
    for video_idx, video_name in enumerate(tqdm(tasks, desc="Processing videos", unit="video")):
        print(f"\n{'='*60}")
        print(f"Processing video: {video_name}")
        print(f"{'='*60}")
        
        # Check if already processed (EgoExoLearn)
        output_csv_check = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
        if os.path.exists(output_csv_check):
            print(f"⏭️  SKIPPING: Already processed (found {os.path.basename(output_csv_check)})")
            continue
        
        try:
            # Try multiple approaches to read the fixation CSV file
            print(f"Loading fixation dataset...")
            csv_path = os.path.join(base_dir, video_name, f'{video_name}_fixation_filtered.csv')
            
            # Check if file exists first
            if not os.path.exists(csv_path):
                print(f"❌ Fixation filtered CSV not found: {csv_path}")
                print(f"⏭️  SKIPPING: {video_name} (run step1.5 first)")
                continue
            
            try:
                # First try: standard approach
                fixation_dataset = pd.read_csv(csv_path)
                print(f"✅ Fixation dataset loaded successfully with standard approach")
            except pd.errors.ParserError as e:
                print(f"⚠️ Standard CSV parsing failed: {e}")
                try:
                    # Second try: more flexible parsing
                    fixation_dataset = pd.read_csv(csv_path, 
                                                 sep=',',
                                                 quotechar='"',
                                                 skipinitialspace=True,
                                                 on_bad_lines='skip',
                                                 engine='python')
                    print(f"✅ Successfully parsed fixation dataset with flexible options")
                except Exception as e2:
                    print(f"⚠️ Flexible parsing also failed: {e2}")
                    # Third try: use error_bad_lines=False for older pandas versions
                    try:
                        fixation_dataset = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)
                        print(f"✅ Successfully parsed fixation dataset with error_bad_lines=False")
                    except Exception as e3:
                        print(f"❌ All fixation dataset parsing attempts failed: {e3}")
                        print(f"⏭️  SKIPPING: {video_name}")
                        continue

            # Get video path (EgoExoLearn uses start_time_seconds, end_time_seconds like EGTEA)
            video_path = os.path.join(video_base_dir, f'{video_name}.mp4')

            # Process all rows with InternVL v2 (including two-stage analysis)
            print("🚀 Starting InternVL-38B v2 two-stage object extraction for ALL fixations...")
            print(f"Total fixations to process: {len(fixation_dataset)}")
            print("=" * 60)

            # Create a copy of the dataset with new columns (including internvl_caption)
            fixation_dataset_with_scene = fixation_dataset.copy()
            fixation_dataset_with_scene['exact_gaze_object'] = None
            fixation_dataset_with_scene['other_objects_in_cropped_area'] = None
            fixation_dataset_with_scene['other_objects_outside_fov'] = None
            fixation_dataset_with_scene['internvl_caption'] = None
            fixation_dataset_with_scene['processing_status'] = None
            fixation_dataset_with_scene['processing_error'] = None

            # Initialize object pool for this video
            object_pool = set()
            print(f"📦 Initialized empty object pool for {video_name}")

            video_start_time = time.time()
            current_video_fixations = len(fixation_dataset)
            
            # Prepare requests for multi-threaded processing
            # NOTE: EgoExoLearn uses 'start_time_seconds' and 'end_time_seconds' (like EGTEA)
            # AND includes action_caption for context!
            requests_data = []
            for idx, row in fixation_dataset.iterrows():
                request = {
                    'video_path': video_path,
                    'gaze_x': row['center_x'],
                    'gaze_y': row['center_y'],
                    'start_time': row['start_time_seconds'],
                    'end_time': row['end_time_seconds'],
                    'request_id': f'fixation_{idx}'
                }
                
                # Add action_caption if available
                if 'action_caption' in row and pd.notna(row['action_caption']):
                    request['action_caption'] = row['action_caption']
                    
                requests_data.append(request)
            
            # Process all fixations using InternVL v2 multi-threaded function
            print(f"🚀 Processing {len(requests_data)} fixations using InternVL v2 with action context...")
            
            # Check GPU availability and memory
            import torch
            if torch.cuda.is_available():
                print(f"🔥 GPU available: {torch.cuda.device_count()} devices")
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            else:
                print("❌ No GPU available, using CPU")
            
            # Use sequential processing (NO THREADING) to avoid OOM
            print("🚀 Using sequential processing (no threading) for 38B model safety...")
            
            # Calculate FOV radius based on camera HFOV and perifovea angle
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            HFOV_deg = 90.0  # Standard HFOV for EgoExo
            r_deg = 13.0     # Perifovea radius in degrees
            px_per_deg = frame_width / HFOV_deg
            fov_radius = int(r_deg * px_per_deg)
            
            print(f"   Video resolution: {frame_width}px width, HFOV: {HFOV_deg}°")
            print(f"   Using perifovea radius: {fov_radius} px (~{r_deg}°)")
            
            results = extract_objects_and_scene_from_video_clip_internvl_v2_sequential(
                requests_data=requests_data,
                fov_radius=fov_radius,
                save_images=False,
                # save_images = True, 
                show_images=False,
                object_pool=list(object_pool) if object_pool else None,
                temperature=0.3,
                output_dir=None  # Disable GIF saving
            )

            print(results)
            
            # Update progress tracking
            processed_fixations += len(requests_data)
            
            # Save results to dataset and update object pool
            for idx, result in enumerate(results):
                if "error" in result:
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'error'
                    fixation_dataset_with_scene.at[idx, 'processing_error'] = result['error']
                else:
                    # Successfully processed - using new format
                    fixation_dataset_with_scene.at[idx, 'exact_gaze_object'] = result.get('exact_gaze_object', {})
                    fixation_dataset_with_scene.at[idx, 'other_objects_in_cropped_area'] = result.get('other_objects_in_cropped_area', [])
                    fixation_dataset_with_scene.at[idx, 'other_objects_outside_fov'] = result.get('other_objects_outside_fov', [])
                    fixation_dataset_with_scene.at[idx, 'internvl_caption'] = result.get('scene_caption', '')
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'success'
                    
                    # Update object pool with new objects found
                    if 'exact_gaze_object' in result and 'object_identity' in result['exact_gaze_object']:
                        object_pool.add(result['exact_gaze_object']['object_identity'].strip().lower())
                    
                    if 'other_objects_in_cropped_area' in result:
                        for obj in result['other_objects_in_cropped_area']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())
                    
                    if 'other_objects_outside_fov' in result:
                        for obj in result['other_objects_outside_fov']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())

            print("\n" + "=" * 60)
            print("🎉 ENHANCED PROCESSING COMPLETED!")
            print("=" * 60)

            # Save object pool to txt file
            object_pool_txt_path = os.path.join(base_dir, video_name, f'{video_name}_object_pool.txt')
            with open(object_pool_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"OBJECT POOL - {video_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total unique objects found: {len(object_pool)}\n\n")
                f.write("Object names (alphabetically sorted):\n")
                f.write("-" * 30 + "\n")
                for obj in sorted(object_pool):
                    f.write(f"{obj}\n")
            print(f"✅ Object pool saved to: {object_pool_txt_path}")
            print(f"📦 Final object pool size: {len(object_pool)} objects")

            # Count other_objects occurrences across all fixations
            print("📈 OTHER OBJECTS OCCURRENCE ANALYSIS")
            print("=" * 60)

            # Extract all other_objects from successful fixations (new format)
            other_object_counter = Counter()
            successful_rows = fixation_dataset_with_scene[fixation_dataset_with_scene['processing_status'] == 'success']

            for idx, row in successful_rows.iterrows():
                # Count objects in cropped area
                cropped_objects = row['other_objects_in_cropped_area']
                if cropped_objects and isinstance(cropped_objects, list):
                    for obj in cropped_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1
                
                # Count objects outside FOV
                outside_objects = row['other_objects_outside_fov']
                if outside_objects and isinstance(outside_objects, list):
                    for obj in outside_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1

            # Save frequency analysis results
            output_path = os.path.join(base_dir, video_name, f'{video_name}_object_frequency_analysis.txt')
            save_frequency_analysis(other_object_counter, output_path, video_name)
            print(f"\n✅ Frequency analysis results saved to: {output_path}")

            # Save the processed dataset with scene information
            output_csv = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
            fixation_dataset_with_scene.to_csv(output_csv, index=False)
            print(f"✅ Processed dataset saved to: {output_csv}")

        except Exception as e:
            print(f"❌ Error processing video {video_name}: {str(e)}")
            print(f"⏭️  SKIPPING: {video_name} and continuing to next video...")
            continue

    print("\n" + "=" * 60)
    print("🎉 ALL EGOEXOLEARN VIDEOS PROCESSED!")
    print("=" * 60)


def process_holoassist(reverse=False, start_pct=0):
    """Process HoloAssist dataset with InternVL (with action_caption context)"""
    print("🚀 Starting HoloAssist Object Extraction with InternVL")
    print("=" * 60)
    
    base_dir = os.path.join(PIPELINE_DIR, 'final_data', 'holoassist', 'metadata')
    video_base_dir = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'holoassist', 'full')
    
    # Load annotation data to filter sessions
    import json as json_module
    annotation_file = os.path.join(video_base_dir, "data-annnotation-trainval-v1_1.json")
    annotated_video_names = set()
    if os.path.exists(annotation_file):
        print("Loading annotation data to filter sessions...")
        with open(annotation_file, 'r') as f:
            annotation_data_filter = json_module.load(f)
        annotated_video_names = set([v.get('video_name') for v in annotation_data_filter if 'video_name' in v])
        print(f"Found {len(annotated_video_names)} videos with annotations")
    
    # Get all tasks (only those with annotations)
    all_tasks = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    tasks = [task for task in all_tasks if task in annotated_video_names]
    skipped_no_annotation = len(all_tasks) - len(tasks)
    
    if skipped_no_annotation > 0:
        print(f"⏭️  Filtered out {skipped_no_annotation} sessions without annotations")
    print(f"Processing {len(tasks)} sessions with annotations")
    
    # Apply start_pct slicing
    if start_pct > 0:
        start_idx = int(len(tasks) * start_pct / 100)
        tasks = tasks[start_idx:]
        print(f"🎯 Starting from {start_pct}% position (index {start_idx}/{len(tasks) + start_idx})")
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")

    # Count total fixations for ETA calculation
    print("🔍 Counting total fixations across all videos...")
    total_fixations, video_fixations = count_total_fixations(base_dir)
    print(f"📊 Total videos: {len(tasks)}")
    print(f"📊 Total fixations to process: {total_fixations}")
    print(f"⏱️ Estimated total time: {(total_fixations * 5) / 3600:.1f} hours")
    print("=" * 60)
    
    # Initialize counters for progress tracking
    processed_fixations = 0
    overall_start_time = time.time()
    
    # Process each video in the dataset
    for video_idx, video_name in enumerate(tqdm(tasks, desc="Processing videos", unit="video")):
        print(f"\n{'='*60}")
        print(f"Processing video: {video_name}")
        print(f"{'='*60}")
        
        # Check if already processed (HoloAssist)
        output_csv_check = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
        if os.path.exists(output_csv_check):
            print(f"⏭️  SKIPPING: Already processed (found {os.path.basename(output_csv_check)})")
            continue
        
        try:
            # Try multiple approaches to read the fixation CSV file
            print(f"Loading fixation dataset...")
            csv_path = os.path.join(base_dir, video_name, f'{video_name}_fixation_filtered.csv')
            
            # Check if file exists first
            if not os.path.exists(csv_path):
                print(f"❌ Fixation filtered CSV not found: {csv_path}")
                print(f"⏭️  SKIPPING: {video_name} (run step1.5 first)")
                continue
            
            try:
                # First try: standard approach
                fixation_dataset = pd.read_csv(csv_path)
                print(f"✅ Fixation dataset loaded successfully with standard approach")
            except pd.errors.ParserError as e:
                print(f"⚠️ Standard CSV parsing failed: {e}")
                try:
                    # Second try: more flexible parsing
                    fixation_dataset = pd.read_csv(csv_path, 
                                                 sep=',',
                                                 quotechar='"',
                                                 skipinitialspace=True,
                                                 on_bad_lines='skip',
                                                 engine='python')
                    print(f"✅ Successfully parsed fixation dataset with flexible options")
                except Exception as e2:
                    print(f"⚠️ Flexible parsing also failed: {e2}")
                    # Third try: use error_bad_lines=False for older pandas versions
                    try:
                        fixation_dataset = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)
                        print(f"✅ Successfully parsed fixation dataset with error_bad_lines=False")
                    except Exception as e3:
                        print(f"❌ All fixation dataset parsing attempts failed: {e3}")
                        print(f"⏭️  SKIPPING: {video_name}")
                        continue

            # Get video path (HoloAssist uses different structure)
            video_path = os.path.join(video_base_dir, video_name, 'Export_py', 'Video_pitchshift.mp4')

            # Process all rows with InternVL v2 (including two-stage analysis)
            print("🚀 Starting InternVL-38B v2 two-stage object extraction for ALL fixations...")
            print(f"Total fixations to process: {len(fixation_dataset)}")
            print("=" * 60)

            # Create a copy of the dataset with new columns (including internvl_caption)
            fixation_dataset_with_scene = fixation_dataset.copy()
            fixation_dataset_with_scene['exact_gaze_object'] = None
            fixation_dataset_with_scene['other_objects_in_cropped_area'] = None
            fixation_dataset_with_scene['other_objects_outside_fov'] = None
            fixation_dataset_with_scene['internvl_caption'] = None
            fixation_dataset_with_scene['processing_status'] = None
            fixation_dataset_with_scene['processing_error'] = None

            # Initialize object pool for this video
            object_pool = set()
            print(f"📦 Initialized empty object pool for {video_name}")

            video_start_time = time.time()
            current_video_fixations = len(fixation_dataset)
            
            # Prepare requests for multi-threaded processing
            # NOTE: HoloAssist uses 'start_time_seconds' and 'end_time_seconds' (like EGTEA)
            # AND includes action_caption for context!
            requests_data = []
            for idx, row in fixation_dataset.iterrows():
                request = {
                    'video_path': video_path,
                    'gaze_x': row['center_x'],
                    'gaze_y': row['center_y'],
                    'start_time': row['start_time_seconds'],
                    'end_time': row['end_time_seconds'],
                    'request_id': f'fixation_{idx}'
                }
                
                # Add action_caption if available
                if 'action_caption' in row and pd.notna(row['action_caption']):
                    request['action_caption'] = row['action_caption']
                    
                requests_data.append(request)
            
            # Process all fixations using InternVL v2 multi-threaded function
            print(f"🚀 Processing {len(requests_data)} fixations using InternVL v2 with action context...")
            
            # Check GPU availability and memory
            import torch
            if torch.cuda.is_available():
                print(f"🔥 GPU available: {torch.cuda.device_count()} devices")
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            else:
                print("❌ No GPU available, using CPU")
            
            # Use sequential processing (NO THREADING) to avoid OOM
            print("🚀 Using sequential processing (no threading) for 38B model safety...")
            
            # Calculate FOV radius based on camera HFOV and perifovea angle
            import cv2
            import math
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            # Read camera intrinsics from Intrinsics.txt to calculate HFOV
            intrinsics_path = os.path.join(os.path.dirname(video_path), 'Video', 'Intrinsics.txt')
            HFOV_deg = 90.0  # Default fallback
            
            if os.path.exists(intrinsics_path):
                try:
                    with open(intrinsics_path, 'r') as f:
                        intrinsic_data = f.readline().strip().split('\t')
                    fx = float(intrinsic_data[0])  # Focal length in x
                    width = int(intrinsic_data[-2])  # Image width
                    # Calculate HFOV: HFOV = 2 * arctan(width / (2 * fx))
                    HFOV_rad = 2 * math.atan(width / (2 * fx))
                    HFOV_deg = math.degrees(HFOV_rad)
                    print(f"   ✅ Loaded camera intrinsics: fx={fx:.2f}, width={width}px")
                except Exception as e:
                    print(f"   ⚠️ Failed to read intrinsics: {e}, using default HFOV")
            else:
                print(f"   ⚠️ Intrinsics not found, using default HFOV")
            
            r_deg = 13.0     # Perifovea radius in degrees
            px_per_deg = frame_width / HFOV_deg
            fov_radius = int(r_deg * px_per_deg)
            
            print(f"   Video resolution: {frame_width}px width, HFOV: {HFOV_deg:.1f}°")
            print(f"   Using perifovea radius: {fov_radius} px (~{r_deg}°)")
            
            results = extract_objects_and_scene_from_video_clip_internvl_v2_sequential(
                requests_data=requests_data,
                fov_radius=fov_radius,
                save_images=False,
                show_images=False,
                object_pool=list(object_pool) if object_pool else None,
                temperature=0.3,
                output_dir=None  # Disable GIF saving
            )

            print(results)
            
            # Update progress tracking
            processed_fixations += len(requests_data)
            
            # Save results to dataset and update object pool
            for idx, result in enumerate(results):
                if "error" in result:
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'error'
                    fixation_dataset_with_scene.at[idx, 'processing_error'] = result['error']
                else:
                    # Successfully processed - using new format
                    fixation_dataset_with_scene.at[idx, 'exact_gaze_object'] = result.get('exact_gaze_object', {})
                    fixation_dataset_with_scene.at[idx, 'other_objects_in_cropped_area'] = result.get('other_objects_in_cropped_area', [])
                    fixation_dataset_with_scene.at[idx, 'other_objects_outside_fov'] = result.get('other_objects_outside_fov', [])
                    fixation_dataset_with_scene.at[idx, 'internvl_caption'] = result.get('scene_caption', '')
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'success'
                    
                    # Update object pool with new objects found
                    if 'exact_gaze_object' in result and 'object_identity' in result['exact_gaze_object']:
                        object_pool.add(result['exact_gaze_object']['object_identity'].strip().lower())
                    
                    if 'other_objects_in_cropped_area' in result:
                        for obj in result['other_objects_in_cropped_area']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())
                    
                    if 'other_objects_outside_fov' in result:
                        for obj in result['other_objects_outside_fov']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())

            print("\n" + "=" * 60)
            print("🎉 ENHANCED PROCESSING COMPLETED!")
            print("=" * 60)

            # Save object pool to txt file
            object_pool_txt_path = os.path.join(base_dir, video_name, f'{video_name}_object_pool.txt')
            with open(object_pool_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"OBJECT POOL - {video_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total unique objects found: {len(object_pool)}\n\n")
                f.write("Object names (alphabetically sorted):\n")
                f.write("-" * 30 + "\n")
                for obj in sorted(object_pool):
                    f.write(f"{obj}\n")
            print(f"✅ Object pool saved to: {object_pool_txt_path}")
            print(f"📦 Final object pool size: {len(object_pool)} objects")

            # Count other_objects occurrences across all fixations
            print("📈 OTHER OBJECTS OCCURRENCE ANALYSIS")
            print("=" * 60)

            # Extract all other_objects from successful fixations (new format)
            other_object_counter = Counter()
            successful_rows = fixation_dataset_with_scene[fixation_dataset_with_scene['processing_status'] == 'success']

            for idx, row in successful_rows.iterrows():
                # Count objects in cropped area
                cropped_objects = row['other_objects_in_cropped_area']
                if cropped_objects and isinstance(cropped_objects, list):
                    for obj in cropped_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1
                
                # Count objects outside FOV
                outside_objects = row['other_objects_outside_fov']
                if outside_objects and isinstance(outside_objects, list):
                    for obj in outside_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1

            # Save frequency analysis results
            output_path = os.path.join(base_dir, video_name, f'{video_name}_object_frequency_analysis.txt')
            save_frequency_analysis(other_object_counter, output_path, video_name)
            print(f"\n✅ Frequency analysis results saved to: {output_path}")

            # Save the processed dataset with scene information
            output_csv = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
            fixation_dataset_with_scene.to_csv(output_csv, index=False)
            print(f"✅ Processed dataset saved to: {output_csv}")

        except Exception as e:
            print(f"❌ Error processing video {video_name}: {str(e)}")
            print(f"⏭️  SKIPPING: {video_name} and continuing to next video...")
            continue

    print("\n" + "=" * 60)
    print("🎉 ALL HOLOASSIST VIDEOS PROCESSED!")
    print("=" * 60)


def process_egoexo_lab(reverse=False, start_pct=0):
    """Process EgoExoLearn Lab dataset with InternVL (with action_caption context)"""
    print("🚀 Starting EgoExoLearn Lab Object Extraction with InternVL")
    print("=" * 60)
    
    base_dir = os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'metadata', 'lab')
    video_base_dir = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egoexolearn', 'full')
    tasks = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    # Apply start_pct slicing
    if start_pct > 0:
        start_idx = int(len(tasks) * start_pct / 100)
        tasks = tasks[start_idx:]
        print(f"🎯 Starting from {start_pct}% position (index {start_idx}/{len(tasks) + start_idx})")
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")

    # Count total fixations for ETA calculation
    print("🔍 Counting total fixations across all lab videos...")
    total_fixations, video_fixations = count_total_fixations(base_dir)
    print(f"📊 Total videos: {len(tasks)}")
    print(f"📊 Total fixations to process: {total_fixations}")
    print(f"⏱️ Estimated total time: {(total_fixations * 5) / 3600:.1f} hours")
    print("=" * 60)
    
    # Initialize counters for progress tracking
    processed_fixations = 0
    overall_start_time = time.time()
    
    # Process each video in the dataset
    for video_idx, video_name in enumerate(tqdm(tasks, desc="Processing lab videos", unit="video")):
        print(f"\n{'='*60}")
        print(f"Processing lab video: {video_name}")
        print(f"{'='*60}")
        
        # Check if already processed (EgoExoLearn Lab)
        output_csv_check = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
        if os.path.exists(output_csv_check):
            print(f"⏭️  SKIPPING: Already processed (found {os.path.basename(output_csv_check)})")
            continue
        
        try:
            # Try multiple approaches to read the fixation CSV file
            print(f"Loading fixation dataset...")
            csv_path = os.path.join(base_dir, video_name, f'{video_name}_fixation_filtered.csv')
            
            # Check if file exists first
            if not os.path.exists(csv_path):
                print(f"❌ Fixation file not found: {csv_path}")
                continue
            
            # Load fixation dataset
            fixation_dataset = pd.read_csv(
                csv_path,
                sep=',',
                quotechar='"',
                skipinitialspace=True,
                on_bad_lines='skip',
                engine='python'
            )
            
            print(f"Loaded fixation dataset with {len(fixation_dataset)} fixations")
            print(f"Columns: {list(fixation_dataset.columns)[:10]}...")
            
            # Video path
            video_path = os.path.join(video_base_dir, f'{video_name}.mp4')
            
            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                continue
            
            print(f"Video path: {video_path}")
            print(f"Frame count: {len(fixation_dataset)}")
            print(f"Total fixations to process: {len(fixation_dataset)}")
            print("=" * 60)

            # Create a copy of the dataset with new columns (including internvl_caption)
            fixation_dataset_with_scene = fixation_dataset.copy()
            fixation_dataset_with_scene['exact_gaze_object'] = None
            fixation_dataset_with_scene['other_objects_in_cropped_area'] = None
            fixation_dataset_with_scene['other_objects_outside_fov'] = None
            fixation_dataset_with_scene['internvl_caption'] = None
            fixation_dataset_with_scene['processing_status'] = None
            fixation_dataset_with_scene['processing_error'] = None

            # Initialize object pool for this video
            object_pool = set()
            print(f"📦 Initialized empty object pool for {video_name}")

            video_start_time = time.time()
            current_video_fixations = len(fixation_dataset)
            
            # Prepare requests for multi-threaded processing
            requests_data = []
            for idx, row in fixation_dataset.iterrows():
                request = {
                    'video_path': video_path,
                    'gaze_x': row['center_x'],
                    'gaze_y': row['center_y'],
                    'start_time': row['start_time_seconds'],
                    'end_time': row['end_time_seconds'],
                    'request_id': f'fixation_{idx}'
                }
                
                # Add action_caption if available
                if 'action_caption' in row and pd.notna(row['action_caption']):
                    request['action_caption'] = row['action_caption']
                    
                requests_data.append(request)
            
            # Process all fixations using InternVL v2 function
            print(f"🚀 Processing {len(requests_data)} fixations using InternVL v2 with action context...")
            
            # Check GPU availability and memory
            import torch
            if torch.cuda.is_available():
                print(f"🔥 GPU available: {torch.cuda.device_count()} devices")
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            else:
                print("❌ No GPU available, using CPU")
            
            # Use sequential processing (NO THREADING) to avoid OOM
            print("🚀 Using sequential processing (no threading) for 38B model safety...")
            
            # Calculate FOV radius based on camera HFOV and perifovea angle
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            HFOV_deg = 90.0  # Standard HFOV for EgoExo
            r_deg = 13.0     # Perifovea radius in degrees
            px_per_deg = frame_width / HFOV_deg
            fov_radius = int(r_deg * px_per_deg)
            
            print(f"   Video resolution: {frame_width}px width, HFOV: {HFOV_deg}°")
            print(f"   Using perifovea radius: {fov_radius} px (~{r_deg}°)")
            
            results = extract_objects_and_scene_from_video_clip_internvl_v2_sequential(
                requests_data=requests_data,
                fov_radius=fov_radius,
                save_images=False,
                show_images=False,
                object_pool=list(object_pool) if object_pool else None,
                temperature=0.3,
                output_dir=None  # Disable GIF saving
            )
            
            print(results)
            
            # Update progress tracking
            processed_fixations += len(requests_data)
            
            # Save results to dataset and update object pool
            for idx, result in enumerate(results):
                if "error" in result:
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'error'
                    fixation_dataset_with_scene.at[idx, 'processing_error'] = result['error']
                else:
                    # Successfully processed - using new format
                    fixation_dataset_with_scene.at[idx, 'exact_gaze_object'] = result.get('exact_gaze_object', {})
                    fixation_dataset_with_scene.at[idx, 'other_objects_in_cropped_area'] = result.get('other_objects_in_cropped_area', [])
                    fixation_dataset_with_scene.at[idx, 'other_objects_outside_fov'] = result.get('other_objects_outside_fov', [])
                    fixation_dataset_with_scene.at[idx, 'internvl_caption'] = result.get('internvl_caption', '')
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'success'
                    
                    # Update object pool: extract object identities from returned results
                    exact_obj = result.get('exact_gaze_object', {})
                    if isinstance(exact_obj, dict) and 'object_identity' in exact_obj:
                        object_pool.add(exact_obj['object_identity'])
                    
                    for obj in result.get('other_objects_in_cropped_area', []):
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_pool.add(obj['object_identity'])
                    
                    for obj in result.get('other_objects_outside_fov', []):
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_pool.add(obj['object_identity'])
            
            # Save the updated dataset
            output_dir = os.path.join(base_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)
            
            output_csv_path = os.path.join(output_dir, f'{video_name}_fixation_with_internvl_v2_scene.csv')
            fixation_dataset_with_scene.to_csv(output_csv_path, index=False)
            
            # Save object pool
            object_pool_path = os.path.join(output_dir, f'{video_name}_object_pool.txt')
            with open(object_pool_path, 'w') as f:
                for obj in sorted(object_pool):
                    f.write(f"{obj}\n")
            
            # Calculate statistics
            video_elapsed = time.time() - video_start_time
            processed_fixations += current_video_fixations
            overall_elapsed = time.time() - overall_start_time
            
            print(f"\n✅ Completed {video_name}")
            print(f"  Saved to: {output_csv_path}")
            print(f"  Object pool size: {len(object_pool)}")
            print(f"  Video processing time: {video_elapsed:.1f}s")
            print(f"  Overall progress: {processed_fixations}/{total_fixations} fixations ({processed_fixations/total_fixations*100:.1f}%)")
            print(f"  Remaining: {total_fixations - processed_fixations} fixations")
            print(f"  ETA: {((total_fixations - processed_fixations) * video_elapsed / current_video_fixations) / 3600:.1f} hours")
            
        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("🎉 EgoExoLearn Lab Processing Complete!")
    print(f"{'='*60}")


def process_egoexo_kitchen(reverse=False, start_pct=0):
    """Process EgoExoLearn Kitchen dataset with InternVL (with action_caption context)"""
    print("🚀 Starting EgoExoLearn Kitchen Object Extraction with InternVL")
    print("=" * 60)
    
    base_dir = os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'metadata', 'kitchen_160')
    video_base_dir = os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egoexolearn', 'full')
    tasks = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    # Apply start_pct slicing
    if start_pct > 0:
        start_idx = int(len(tasks) * start_pct / 100)
        tasks = tasks[start_idx:]
        print(f"🎯 Starting from {start_pct}% position (index {start_idx}/{len(tasks) + start_idx})")
    
    # Reverse order if requested
    if reverse:
        tasks = tasks[::-1]
        print("⏪ Processing videos in REVERSE order")

    # Count total fixations for ETA calculation
    print("🔍 Counting total fixations across all kitchen videos...")
    total_fixations, video_fixations = count_total_fixations(base_dir)
    print(f"📊 Total videos: {len(tasks)}")
    print(f"📊 Total fixations to process: {total_fixations}")
    print(f"⏱️ Estimated total time: {(total_fixations * 5) / 3600:.1f} hours")
    print("=" * 60)
    
    # Initialize counters for progress tracking
    processed_fixations = 0
    overall_start_time = time.time()
    
    # Process each video in the dataset
    for video_idx, video_name in enumerate(tqdm(tasks, desc="Processing kitchen videos", unit="video")):
        print(f"\n{'='*60}")
        print(f"Processing kitchen video: {video_name}")
        print(f"{'='*60}")

        if (video_name != 'bee95ccc-ac78-11ee-819f-80615f12b59e') : #and (video_name != 'bee98c06-ac78-11ee-819f-80615f12b59e'): 
            continue 

        
        # Check if already processed
        # output_csv_check = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
        # if os.path.exists(output_csv_check):
        #     print(f"⏭️  SKIPPING: Already processed (found {os.path.basename(output_csv_check)})")
        #     continue
        
        try:
            # Try multiple approaches to read the fixation CSV file
            print(f"Loading fixation dataset...")
            csv_path = os.path.join(base_dir, video_name, f'{video_name}_fixation_filtered.csv')
            
            # Check if file exists first
            if not os.path.exists(csv_path):
                print(f"❌ Fixation filtered CSV not found: {csv_path}")
                print(f"⏭️  SKIPPING: {video_name} (run step1.5 first)")
                continue
            
            try:
                # First try: standard approach
                fixation_dataset = pd.read_csv(csv_path)
                print(f"✅ Fixation dataset loaded successfully with standard approach")
            except pd.errors.ParserError as e:
                print(f"⚠️ Standard CSV parsing failed: {e}")
                try:
                    # Second try: more flexible parsing
                    fixation_dataset = pd.read_csv(csv_path, 
                                                 sep=',',
                                                 quotechar='"',
                                                 skipinitialspace=True,
                                                 on_bad_lines='skip',
                                                 engine='python')
                    print(f"✅ Successfully parsed fixation dataset with flexible options")
                except Exception as e2:
                    print(f"⚠️ Flexible parsing also failed: {e2}")
                    # Third try: use error_bad_lines=False for older pandas versions
                    try:
                        fixation_dataset = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)
                        print(f"✅ Successfully parsed fixation dataset with error_bad_lines=False")
                    except Exception as e3:
                        print(f"❌ All fixation dataset parsing attempts failed: {e3}")
                        print(f"⏭️  SKIPPING: {video_name}")
                        continue

            # Get video path
            video_path = os.path.join(video_base_dir, f'{video_name}.mp4')

            # Process all rows with InternVL v2 (including two-stage analysis)
            print("🚀 Starting InternVL-38B v2 two-stage object extraction for ALL fixations...")
            print(f"Total fixations to process: {len(fixation_dataset)}")
            print("=" * 60)

            # Create a copy of the dataset with new columns (including internvl_caption)
            fixation_dataset_with_scene = fixation_dataset.copy()
            fixation_dataset_with_scene['exact_gaze_object'] = None
            fixation_dataset_with_scene['other_objects_in_cropped_area'] = None
            fixation_dataset_with_scene['other_objects_outside_fov'] = None
            fixation_dataset_with_scene['internvl_caption'] = None
            fixation_dataset_with_scene['processing_status'] = None
            fixation_dataset_with_scene['processing_error'] = None

            # Initialize object pool for this video
            object_pool = set()
            print(f"📦 Initialized empty object pool for {video_name}")

            video_start_time = time.time()
            current_video_fixations = len(fixation_dataset)
            
            # Prepare requests for multi-threaded processing
            # Kitchen uses 'start_time_seconds' and 'end_time_seconds' (like EGTEA)
            # AND includes action_caption for context!
            requests_data = []
            for idx, row in fixation_dataset.iterrows():
                request = {
                    'video_path': video_path,
                    'gaze_x': row['center_x'],
                    'gaze_y': row['center_y'],
                    'start_time': row['start_time_seconds'],
                    'end_time': row['end_time_seconds'],
                    'request_id': f'fixation_{idx}'
                }
                
                # Add action_caption if available
                if 'action_caption' in row and pd.notna(row['action_caption']):
                    request['action_caption'] = row['action_caption']
                    
                requests_data.append(request)
            
            # Process all fixations using InternVL v2 multi-threaded function
            print(f"🚀 Processing {len(requests_data)} fixations using InternVL v2 with action context...")
            
            # Check GPU availability and memory
            import torch
            if torch.cuda.is_available():
                print(f"🔥 GPU available: {torch.cuda.device_count()} devices")
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            else:
                print("❌ No GPU available, using CPU")
            
            # Use sequential processing (NO THREADING) to avoid OOM
            print("🚀 Using sequential processing (no threading) for 38B model safety...")
            
            # Calculate FOV radius based on camera HFOV and perifovea angle
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            
            HFOV_deg = 90.0  # Standard HFOV for EgoExo
            r_deg = 13.0     # Perifovea radius in degrees
            px_per_deg = frame_width / HFOV_deg
            fov_radius = int(r_deg * px_per_deg)
            
            print(f"   Video resolution: {frame_width}px width, HFOV: {HFOV_deg}°")
            print(f"   Using perifovea radius: {fov_radius} px (~{r_deg}°)")
            
            results = extract_objects_and_scene_from_video_clip_internvl_v2_sequential(
                requests_data=requests_data,
                fov_radius=fov_radius,
                # save_images=False,
                save_images=True,
                show_images=False,
                object_pool=list(object_pool) if object_pool else None,
                temperature=0.3,
                output_dir=None,  # Disable GIF saving
                dataset_type='egoexo_kitchen'  # Focus on cooking ingredients
            )

            print(results)
            
            # Update progress tracking
            processed_fixations += len(requests_data)
            
            # Save results to dataset and update object pool
            for idx, result in enumerate(results):
                if "error" in result:
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'error'
                    fixation_dataset_with_scene.at[idx, 'processing_error'] = result['error']
                else:
                    # Successfully processed - using new format
                    fixation_dataset_with_scene.at[idx, 'exact_gaze_object'] = result.get('exact_gaze_object', {})
                    fixation_dataset_with_scene.at[idx, 'other_objects_in_cropped_area'] = result.get('other_objects_in_cropped_area', [])
                    fixation_dataset_with_scene.at[idx, 'other_objects_outside_fov'] = result.get('other_objects_outside_fov', [])
                    fixation_dataset_with_scene.at[idx, 'internvl_caption'] = result.get('scene_caption', '')
                    fixation_dataset_with_scene.at[idx, 'processing_status'] = 'success'
                    
                    # Update object pool with new objects found
                    if 'exact_gaze_object' in result and 'object_identity' in result['exact_gaze_object']:
                        object_pool.add(result['exact_gaze_object']['object_identity'].strip().lower())
                    
                    if 'other_objects_in_cropped_area' in result:
                        for obj in result['other_objects_in_cropped_area']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())
                    
                    if 'other_objects_outside_fov' in result:
                        for obj in result['other_objects_outside_fov']:
                            if 'object_identity' in obj:
                                object_pool.add(obj['object_identity'].strip().lower())

            print("\n" + "=" * 60)
            print("🎉 ENHANCED PROCESSING COMPLETED!")
            print("=" * 60)

            # Save object pool to txt file
            object_pool_txt_path = os.path.join(base_dir, video_name, f'{video_name}_object_pool.txt')
            with open(object_pool_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"OBJECT POOL - {video_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total unique objects found: {len(object_pool)}\n\n")
                f.write("Object names (alphabetically sorted):\n")
                f.write("-" * 30 + "\n")
                for obj in sorted(object_pool):
                    f.write(f"{obj}\n")
            print(f"✅ Object pool saved to: {object_pool_txt_path}")
            print(f"📦 Final object pool size: {len(object_pool)} objects")

            # Count other_objects occurrences across all fixations
            print("📈 OTHER OBJECTS OCCURRENCE ANALYSIS")
            print("=" * 60)

            # Extract all other_objects from successful fixations (new format)
            other_object_counter = Counter()
            successful_rows = fixation_dataset_with_scene[fixation_dataset_with_scene['processing_status'] == 'success']

            for idx, row in successful_rows.iterrows():
                # Count objects in cropped area
                cropped_objects = row['other_objects_in_cropped_area']
                if cropped_objects and isinstance(cropped_objects, list):
                    for obj in cropped_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1
                
                # Count objects outside FOV
                outside_objects = row['other_objects_outside_fov']
                if outside_objects and isinstance(outside_objects, list):
                    for obj in outside_objects:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            object_identity = obj['object_identity']
                            other_object_counter[object_identity] += 1

            # Save frequency analysis results
            output_path = os.path.join(base_dir, video_name, f'{video_name}_object_frequency_analysis.txt')
            save_frequency_analysis(other_object_counter, output_path, video_name)
            print(f"\n✅ Frequency analysis results saved to: {output_path}")

            # Save the processed dataset with scene information
            output_csv = os.path.join(base_dir, video_name, f'{video_name}_fixation_with_internvl_v2_scene.csv')
            fixation_dataset_with_scene.to_csv(output_csv, index=False)
            print(f"✅ Processed dataset saved to: {output_csv}")

        except Exception as e:
            print(f"❌ Error processing video {video_name}: {str(e)}")
            print(f"⏭️  SKIPPING: {video_name} and continuing to next video...")
            continue

    print("\n" + "=" * 60)
    print("🎉 ALL EGOEXO KITCHEN VIDEOS PROCESSED!")
    print("=" * 60)


def main():
    """Main processing function with dataset selection"""
    parser = argparse.ArgumentParser(description='Process gaze object extraction with InternVL for different datasets')
    parser.add_argument('--dataset', type=str, choices=['egtea', 'ego4d', 'egoexo', 'egoexo-lab', 'egoexo-kitchen', 'holoassist'], required=True,
                        help='Dataset to process: egtea, ego4d, egoexo, egoexo-lab, egoexo-kitchen, or holoassist')
    parser.add_argument('--reverse', action='store_true',
                        help='Process videos in reverse order (useful for parallel processing)')
    parser.add_argument('--start-pct', type=int, default=0,
                        help='Start processing from this percentage point (0-100)')
    
    args = parser.parse_args()
    
    if args.dataset == 'egtea':
        process_egtea(reverse=args.reverse, start_pct=args.start_pct)
    elif args.dataset == 'ego4d':
        process_ego4d(reverse=args.reverse, start_pct=args.start_pct)
    elif args.dataset == 'egoexo':
        process_egoexo(reverse=args.reverse, start_pct=args.start_pct)
    elif args.dataset == 'egoexo-lab':
        process_egoexo_lab(reverse=args.reverse, start_pct=args.start_pct)
    elif args.dataset == 'egoexo-kitchen':
        process_egoexo_kitchen(reverse=args.reverse, start_pct=args.start_pct)
    elif args.dataset == 'holoassist':
        process_holoassist(reverse=args.reverse, start_pct=args.start_pct)
    else:
        print(f"Unknown dataset: {args.dataset}")
        print("Please choose 'egtea', 'ego4d', 'egoexo', 'egoexo-lab', 'egoexo-kitchen', or 'holoassist'")


if __name__ == "__main__":
    main()

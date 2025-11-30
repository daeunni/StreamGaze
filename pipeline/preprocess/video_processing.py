"""
Video processing functions for loading and handling video files
"""

import cv2
import os
import glob
import pandas as pd
from pathlib import Path
from .gaze_processing import parse_gtea_gaze, parse_ego4d_gaze, parse_egoexo_gaze, parse_holoassist_gaze, extract_fixation_segments, extract_confusion_segments
from .action_mapping import map_actions_to_gaze, create_segment_dataset_with_actions, seconds_to_timestamp
from .visualization import plot_gaze_segments, visualize_gaze_with_trail, visualize_gaze_green_dot_red_fov, extract_and_save_gifs


'''
Extract fixation/confusion from single video 
'''

def process_single_video(video_path, gaze_path, output_dir, action_data=None, skip_viz=False, viz_only=False):
    """Process a single video
    
    Args:
        video_path: Path to video file
        gaze_path: Path to gaze data file
        output_dir: Output directory
        action_data: Action annotation data (optional)
        skip_viz: Skip visualization generation
        viz_only: Only regenerate visualization (requires existing fixation data)
    """
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    if viz_only:
        print(f"Processing (VIZ-ONLY): {video_name}")
    else:
        print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Create output directory
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # In viz-only mode, load existing fixation data and skip to visualization
    if viz_only:
        fixation_csv = os.path.join(video_output_dir, f"{video_name}_fixation_dataset.csv")
        if not os.path.exists(fixation_csv):
            print(f"   ❌ Fixation data not found: {fixation_csv}")
            return
        
        print("Loading existing fixation data...")
        fixation_df = pd.read_csv(fixation_csv)
        print(f"   Loaded {len(fixation_df)} fixation segments")
        
        # Load video frames for visualization
        print("Loading video frames...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"   Loaded {len(frames)} frames")
        
        # Parse gaze data
        print("Parsing gaze data...")
        gaze_data = parse_gtea_gaze(gaze_path)
        gaze_type_map = {0: 'untracked', 1: 'fixation', 2: 'saccade', 3: 'unknown', 4: 'truncated'}
        gaze_text_data = [[gaze_data[i, 0], gaze_data[i, 1], gaze_type_map[int(gaze_data[i, 2])]] for i in range(len(gaze_data))]
        df = pd.DataFrame(gaze_text_data, columns=["px", "py", "gaze_type"])
        df['frame_idx'] = df.index
        
        # Generate visualization only
        print("Generating visualization...")
        viz_path = os.path.join(video_output_dir, f"{video_name}_gaze_visualization.mp4")
        visualize_gaze_green_dot_red_fov(frames, df, viz_path, fov_radius=100, fps=24)
        print(f"   ✅ Visualization saved: {viz_path}")
        return
    
    try:
        # 1. Load video
        print("1. Loading video frames...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"   Total frames loaded: {len(frames)}")
        
        # 2. Parse gaze data
        print("2. Parsing gaze data...")
        if not os.path.exists(gaze_path):
            print(f"   ❌ Gaze file not found: {gaze_path}")
            return
        
        gaze_data = parse_gtea_gaze(gaze_path)
        
        # Check if gaze data is empty
        if gaze_data.size == 0:
            print(f"   ❌ No valid gaze data found, skipping {video_name}")
            return
        
        # 3. Create dataframe (노트북 방식과 동일)
        print("3. Creating dataframe...")
        gaze_type_map = {
            0: 'untracked',
            1: 'fixation',
            2: 'saccade',
            3: 'unknown',
            4: 'truncated'
        }
        gaze_text_data = [
            [gaze_data[i, 0], gaze_data[i, 1], gaze_type_map[int(gaze_data[i, 2])]]
            for i in range(len(gaze_data))
        ]

        # DataFrame 생성
        df = pd.DataFrame(gaze_text_data, columns=["px", "py", "gaze_type"])     # frame-wise gaze axis 

        # timestamp 컬럼 추가 (frame_idx를 fps로 나누어 초 단위로 변환 후 00:00:00 형식으로 변환)
        fps = 24
        df['frame_idx'] = df.index
        df['time_seconds'] = df['frame_idx'] / fps

        # time을 ms 단위로 변환 (action 데이터와 동일한 형식)
        df['time_ms'] = df['time_seconds'] * 1000

        df['timestamp'] = df['time_seconds'].apply(seconds_to_timestamp)
        df = map_actions_to_gaze(df, action_data, video_name)  # NOTE added: Map actions to gaze data
        print('Columns: ', df.columns)
        
        # 4. Extract fixations
        print("4. Extracting fixations...")
        if len(df) == 0:
            print("   ⚠️ Empty dataframe, skipping fixation extraction")
            fixations = []
        else:
            fixations = extract_fixation_segments(df, radius_thresh=0.15, duration_thresh=2, gap_thresh=0.01)
        
        # 5. Extract confusion segments
        print("5. Extracting confusion segments...")
        if len(df) == 0:
            print("   ⚠️ Empty dataframe, skipping confusion extraction")
            confusion_segments = []
        else:
            confusion_segments = extract_confusion_segments(df, duration_thresh=0.5)
        
        # ================================
        # 6. Create fixation dataset
        print("6. Creating fixation dataset...")
        if action_data is not None and fixations:
            fixation_dataset = create_segment_dataset_with_actions(fixations, df, action_data, video_name)
        else:
            fixation_dataset = pd.DataFrame(fixations) if fixations else pd.DataFrame()
        
        # 7. Save results
        print("7. Saving results...")
        
        # Save DataFrame
        df_path = os.path.join(video_output_dir, f"{video_name}_total_metadata.csv")
        df.to_csv(df_path, index=False)
        print(f"   ✅ Gaze dataframe saved: {df_path}")
        
        # Save fixation dataset
        if not fixation_dataset.empty:
            fixation_path = os.path.join(video_output_dir, f"{video_name}_fixation_dataset.csv")
            fixation_dataset.to_csv(fixation_path, index=False)
            print(f"   ✅ Fixation dataset saved: {fixation_path}")
        
        # Save confusion dataset (disabled - not needed)
        # if confusion_segments:
        #     if action_data is not None:
        #         confusion_df = create_segment_dataset_with_actions(confusion_segments, df, action_data, video_name)
        #     else:
        #         confusion_df = pd.DataFrame(confusion_segments)
        #     confusion_path = os.path.join(video_output_dir, f"{video_name}_confusion_dataset.csv")
        #     confusion_df.to_csv(confusion_path, index=False)
        #     print(f"   ✅ Confusion dataset saved: {confusion_path}")
        
        # 8. Create timeline visualization
        if not skip_viz:
            print("8. Creating timeline visualization...")
            if not fixation_dataset.empty or confusion_segments:
                # Get confusion DataFrame for plotting
                if confusion_segments:
                    if action_data is not None:
                        confusion_df_for_plot = create_segment_dataset_with_actions(confusion_segments, df, action_data, video_name)
                    else:
                        confusion_df_for_plot = pd.DataFrame(confusion_segments)
                else:
                    confusion_df_for_plot = pd.DataFrame()
                    
                # Create timeline plot
                video_duration = df['time_seconds'].max() if len(df) > 0 else None
                timeline_path = os.path.join(video_output_dir, f"{video_name}_timeline.png")
                plot_gaze_segments(fixation_dataset, confusion_df_for_plot, video_duration, timeline_path)
            else:
                print("   ⚠️ No fixation or confusion data to visualize")
        else:
            print("8. ⚡ Skipping timeline visualization (--no-viz)")

        # 9. Create visualization video (only if we have valid data)
        if not skip_viz:
            if len(df) > 0 and len(frames) > 0:
                print("9. Creating visualization video with green dot + red FOV...")
                viz_path = os.path.join(video_output_dir, f"{video_name}_gaze_visualization.mp4")
                visualize_gaze_green_dot_red_fov(frames, df, viz_path, fov_radius=100, fps=24)
            else:
                print("9. Skipping visualization video (no valid gaze data)")
        else:
            print("9. ⚡ Skipping visualization video (--no-viz)")
        
        print(f"✅ {video_name} processing completed!")
        
    except Exception as e:
        print(f"❌ Error processing {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()


def process_single_video_ego4d(video_path, gaze_path, output_dir, action_data=None, fps=30, skip_viz=False, viz_only=False):
    """Process a single Ego4D video
    
    Args:
        video_path: Path to video file
        gaze_path: Path to gaze data file
        output_dir: Output directory
        action_data: Action annotation data (optional)
        fps: Frames per second
        skip_viz: Skip visualization generation
        viz_only: Only regenerate visualization (requires existing fixation data)
    """
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    if viz_only:
        print(f"Processing (VIZ-ONLY): {video_name}")
    else:
        print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Create output directory
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # In viz-only mode, load existing fixation data and skip to visualization
    if viz_only:
        fixation_csv = os.path.join(video_output_dir, f"{video_name}_fixation_dataset.csv")
        if not os.path.exists(fixation_csv):
            print(f"   ❌ Fixation data not found: {fixation_csv}")
            return
        
        print("Loading existing fixation data...")
        fixation_df = pd.read_csv(fixation_csv)
        print(f"   Loaded {len(fixation_df)} fixation segments")
        
        # Load video frames for visualization
        print("Loading video frames...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"   Loaded {len(frames)} frames")
        
        # Parse gaze data (ego4d format)
        print("Parsing gaze data...")
        gaze_data = parse_ego4d_gaze(gaze_path, fps=fps)
        if gaze_data is None or gaze_data.size == 0:
            print("   ❌ Failed to parse gaze data")
            return
        
        # Convert numpy array to DataFrame
        gaze_type_map = {0: 'untracked', 1: 'fixation', 2: 'saccade', 3: 'unknown', 4: 'truncated'}
        gaze_text_data = [[gaze_data[i, 0], gaze_data[i, 1], gaze_type_map[int(gaze_data[i, 2])]] for i in range(len(gaze_data))]
        df = pd.DataFrame(gaze_text_data, columns=["px", "py", "gaze_type"])
        df['frame_idx'] = df.index
        
        # Generate visualization only
        print("Generating visualization...")
        viz_path = os.path.join(video_output_dir, f"{video_name}_gaze_visualization.mp4")
        visualize_gaze_green_dot_red_fov(frames, df, viz_path, fov_radius=100, fps=fps)
        print(f"   ✅ Visualization saved: {viz_path}")
        return
    
    try:
        # 1. Load video
        print("1. Loading video frames...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"   Total frames loaded: {len(frames)}")
        
        # 2. Parse gaze data (Ego4D format)
        print("2. Parsing Ego4D gaze data...")
        if not os.path.exists(gaze_path):
            print(f"   ❌ Gaze file not found: {gaze_path}")
            return
        
        gaze_data = parse_ego4d_gaze(gaze_path, fps=fps)
        
        # Check if gaze data is empty
        if gaze_data.size == 0:
            print(f"   ❌ No valid gaze data found, skipping {video_name}")
            return
        
        # 3. Create dataframe
        print("3. Creating dataframe...")
        gaze_type_map = {
            0: 'untracked',
            1: 'fixation',
            2: 'saccade',
            3: 'unknown',
            4: 'truncated'
        }
        gaze_text_data = [
            [gaze_data[i, 0], gaze_data[i, 1], gaze_type_map[int(gaze_data[i, 2])]]
            for i in range(len(gaze_data))
        ]

        # DataFrame 생성
        df = pd.DataFrame(gaze_text_data, columns=["px", "py", "gaze_type"])

        # timestamp 컬럼 추가
        df['frame_idx'] = df.index
        df['time_seconds'] = df['frame_idx'] / fps

        # time을 ms 단위로 변환
        df['time_ms'] = df['time_seconds'] * 1000

        df['timestamp'] = df['time_seconds'].apply(seconds_to_timestamp)
        
        # Map actions if available (Ego4D might not have action data)
        if action_data is not None:
            df = map_actions_to_gaze(df, action_data, video_name)
        print('Columns: ', df.columns)
        
        # 4. Extract fixations
        print("4. Extracting fixations...")
        if len(df) == 0:
            print("   ⚠️ Empty dataframe, skipping fixation extraction")
            fixations = []
        else:
            # Adjust radius_thresh for Ego4D resolution (1088x1080 vs EGTEA 640x480)
            # EGTEA: 0.15 * 640 = 96 pixels → Ego4D: 96 / 1088 ≈ 0.088
            fixations = extract_fixation_segments(df, radius_thresh=0.088, duration_thresh=2, gap_thresh=0.01)
        
        # 5. Extract confusion segments
        print("5. Extracting confusion segments...")
        if len(df) == 0:
            print("   ⚠️ Empty dataframe, skipping confusion extraction")
            confusion_segments = []
        else:
            confusion_segments = extract_confusion_segments(df, duration_thresh=0.5)
        
        # 6. Create fixation dataset
        print("6. Creating fixation dataset...")
        if action_data is not None and fixations:
            fixation_dataset = create_segment_dataset_with_actions(fixations, df, action_data, video_name)
        else:
            fixation_dataset = pd.DataFrame(fixations) if fixations else pd.DataFrame()
        
        # 7. Save results
        print("7. Saving results...")
        
        # Save DataFrame
        df_path = os.path.join(video_output_dir, f"{video_name}_total_metadata.csv")
        df.to_csv(df_path, index=False)
        print(f"   ✅ Gaze dataframe saved: {df_path}")
        
        # Save fixation dataset
        if not fixation_dataset.empty:
            fixation_path = os.path.join(video_output_dir, f"{video_name}_fixation_dataset.csv")
            fixation_dataset.to_csv(fixation_path, index=False)
            print(f"   ✅ Fixation dataset saved: {fixation_path}")
        
        # Save confusion dataset (disabled - not needed)
        # if confusion_segments:
        #     if action_data is not None:
        #         confusion_df = create_segment_dataset_with_actions(confusion_segments, df, action_data, video_name)
        #     else:
        #         confusion_df = pd.DataFrame(confusion_segments)
        #     confusion_path = os.path.join(video_output_dir, f"{video_name}_confusion_dataset.csv")
        #     confusion_df.to_csv(confusion_path, index=False)
        #     print(f"   ✅ Confusion dataset saved: {confusion_path}")
        
        # 8. Create timeline visualization
        if not skip_viz:
            print("8. Creating timeline visualization...")
            if not fixation_dataset.empty or confusion_segments:
                # Get confusion DataFrame for plotting
                if confusion_segments:
                    if action_data is not None:
                        confusion_df_for_plot = create_segment_dataset_with_actions(confusion_segments, df, action_data, video_name)
                    else:
                        confusion_df_for_plot = pd.DataFrame(confusion_segments)
                else:
                    confusion_df_for_plot = pd.DataFrame()
                    
                # Create timeline plot
                video_duration = df['time_seconds'].max() if len(df) > 0 else None
                timeline_path = os.path.join(video_output_dir, f"{video_name}_timeline.png")
                plot_gaze_segments(fixation_dataset, confusion_df_for_plot, video_duration, timeline_path)
            else:
                print("   ⚠️ No fixation or confusion data to visualize")
        else:
            print("8. ⚡ Skipping timeline visualization (--no-viz)")

        # 9. Create visualization video
        if not skip_viz:
            if len(df) > 0 and len(frames) > 0:
                print("9. Creating visualization video with green dot + red FOV...")
                viz_path = os.path.join(video_output_dir, f"{video_name}_gaze_visualization.mp4")
                # Adjust FOV radius for Ego4D resolution
                # EGTEA: 100 pixels / 640 = 15.6% of width
                # Ego4D: 0.156 * 1088 ≈ 170 pixels
                visualize_gaze_green_dot_red_fov(frames, df, viz_path, fov_radius=170, fps=fps)
            else:
                print("9. Skipping visualization video (no valid gaze data)")
        else:
            print("9. ⚡ Skipping visualization video (--no-viz)")
        
        print(f"✅ {video_name} processing completed!")
        
    except Exception as e:
        print(f"❌ Error processing {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()


def process_single_video_egoexo(video_path, gaze_path, output_dir, action_data=None, fps=30, skip_viz=False, viz_only=False):
    """Process a single EgoExoLearn video
    
    Args:
        video_path: Path to video file
        gaze_path: Path to gaze data file (.npy)
        output_dir: Output directory
        action_data: Action annotation data (optional)
        fps: Frames per second
        skip_viz: Skip visualization generation
        viz_only: Only regenerate visualization (requires existing fixation data)
    """
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    if viz_only:
        print(f"Processing (VIZ-ONLY): {video_name}")
    else:
        print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Create output directory
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # In viz-only mode, load existing fixation data and skip to visualization
    if viz_only:
        fixation_csv = os.path.join(video_output_dir, f"{video_name}_fixation_dataset.csv")
        if not os.path.exists(fixation_csv):
            print(f"   ❌ Fixation data not found: {fixation_csv}")
            return
        
        print("Loading existing fixation data...")
        fixation_df = pd.read_csv(fixation_csv)
        print(f"   Loaded {len(fixation_df)} fixation segments")
        
        # Load video frames for visualization
        print("Loading video frames...")
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"   Loaded {len(frames)} frames ({frame_width}x{frame_height})")
        
        # Parse gaze data (egoexo format - .npy)
        print("Parsing gaze data...")
        gaze_data = parse_egoexo_gaze(gaze_path, fps=fps)
        if gaze_data is None or gaze_data.size == 0:
            print("   ❌ Failed to parse gaze data")
            return
        
        # Convert numpy array to DataFrame
        gaze_type_map = {0: 'untracked', 1: 'fixation', 2: 'saccade', 3: 'unknown', 4: 'truncated'}
        gaze_text_data = [[gaze_data[i, 0], gaze_data[i, 1], gaze_type_map[int(gaze_data[i, 2])]] for i in range(len(gaze_data))]
        df = pd.DataFrame(gaze_text_data, columns=["px", "py", "gaze_type"])
        df['frame_idx'] = df.index
        
        # Generate visualization only
        print("Generating visualization...")
        viz_path = os.path.join(video_output_dir, f"{video_name}_gaze_visualization.mp4")
        visualize_gaze_green_dot_red_fov(frames, df, viz_path, fov_radius=100, fps=fps)
        print(f"   ✅ Visualization saved: {viz_path}")
        return
    
    try:
        # 1. Load video and get resolution
        print("1. Loading video frames...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get video resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"   Total frames loaded: {len(frames)}")
        print(f"   Video resolution: {frame_width}x{frame_height}")
        
        # 2. Parse gaze data (EgoExoLearn .npy format)
        print("2. Parsing EgoExoLearn gaze data...")
        if not os.path.exists(gaze_path):
            print(f"   ❌ Gaze file not found: {gaze_path}")
            return
        
        gaze_data = parse_egoexo_gaze(gaze_path, fps=fps)
        
        # Check if gaze data is empty
        if gaze_data.size == 0:
            print(f"   ❌ No valid gaze data found, skipping {video_name}")
            return
        
        # 3. Create dataframe
        print("3. Creating dataframe...")
        gaze_type_map = {
            0: 'untracked',
            1: 'fixation',
            2: 'saccade',
            3: 'unknown',
            4: 'truncated'
        }
        gaze_text_data = [
            [gaze_data[i, 0], gaze_data[i, 1], gaze_type_map[int(gaze_data[i, 2])]]
            for i in range(len(gaze_data))
        ]

        # DataFrame 생성
        df = pd.DataFrame(gaze_text_data, columns=["px", "py", "gaze_type"])

        # timestamp 컬럼 추가
        df['frame_idx'] = df.index
        df['time_seconds'] = df['frame_idx'] / fps

        # time을 ms 단위로 변환
        df['time_ms'] = df['time_seconds'] * 1000

        df['timestamp'] = df['time_seconds'].apply(seconds_to_timestamp)
        
        # Map actions if available
        if action_data is not None:
            df = map_actions_to_gaze(df, action_data, video_name)
        print('Columns: ', df.columns)
        
        # 4. Extract fixations
        print("4. Extracting fixations...")
        if len(df) == 0:
            print("   ⚠️ Empty dataframe, skipping fixation extraction")
            fixations = []
        else:
            # Calculate radius_thresh dynamically based on video resolution
            # Target: 96 pixels (same as EGTEA baseline)
            target_pixels = 96
            radius_thresh = target_pixels / frame_width
            print(f"   Calculated radius_thresh: {radius_thresh:.4f} ({target_pixels} pixels for {frame_width}px width)")
            fixations = extract_fixation_segments(df, radius_thresh=radius_thresh, duration_thresh=2, gap_thresh=0.01)
        
        # 5. Extract confusion segments
        print("5. Extracting confusion segments...")
        if len(df) == 0:
            print("   ⚠️ Empty dataframe, skipping confusion extraction")
            confusion_segments = []
        else:
            confusion_segments = extract_confusion_segments(df, duration_thresh=0.5)
        
        # 6. Create fixation dataset
        print("6. Creating fixation dataset...")
        if action_data is not None and fixations:
            fixation_dataset = create_segment_dataset_with_actions(fixations, df, action_data, video_name)
        else:
            fixation_dataset = pd.DataFrame(fixations) if fixations else pd.DataFrame()
        
        # 7. Save results
        print("7. Saving results...")
        
        # Save DataFrame
        df_path = os.path.join(video_output_dir, f"{video_name}_total_metadata.csv")
        df.to_csv(df_path, index=False)
        print(f"   ✅ Gaze dataframe saved: {df_path}")
        
        # Save fixation dataset
        if not fixation_dataset.empty:
            fixation_path = os.path.join(video_output_dir, f"{video_name}_fixation_dataset.csv")
            fixation_dataset.to_csv(fixation_path, index=False)
            print(f"   ✅ Fixation dataset saved: {fixation_path}")
        
        # Save confusion dataset (disabled - not needed)
        # if confusion_segments:
        #     if action_data is not None:
        #         confusion_df = create_segment_dataset_with_actions(confusion_segments, df, action_data, video_name)
        #     else:
        #         confusion_df = pd.DataFrame(confusion_segments)
        #     confusion_path = os.path.join(video_output_dir, f"{video_name}_confusion_dataset.csv")
        #     confusion_df.to_csv(confusion_path, index=False)
        #     print(f"   ✅ Confusion dataset saved: {confusion_path}")
        
        # 8. Create timeline visualization
        if not skip_viz:
            print("8. Creating timeline visualization...")
            if not fixation_dataset.empty or confusion_segments:
                # Get confusion DataFrame for plotting
                if confusion_segments:
                    if action_data is not None:
                        confusion_df_for_plot = create_segment_dataset_with_actions(confusion_segments, df, action_data, video_name)
                    else:
                        confusion_df_for_plot = pd.DataFrame(confusion_segments)
                else:
                    confusion_df_for_plot = pd.DataFrame()
                    
                # Create timeline plot
                video_duration = df['time_seconds'].max() if len(df) > 0 else None
                timeline_path = os.path.join(video_output_dir, f"{video_name}_timeline.png")
                plot_gaze_segments(fixation_dataset, confusion_df_for_plot, video_duration, timeline_path)
            else:
                print("   ⚠️ No fixation or confusion data to visualize")
        else:
            print("8. ⚡ Skipping timeline visualization (--no-viz)")

        # 9. Create visualization video
        if not skip_viz:
            if len(df) > 0 and len(frames) > 0:
                print("9. Creating visualization video with green dot + red FOV...")
                viz_path = os.path.join(video_output_dir, f"{video_name}_gaze_visualization.mp4")
                # Calculate FOV radius dynamically based on resolution
                # Target: ~15.6% of width (170 pixels for 1088px width baseline)
                fov_radius = int(0.156 * frame_width)
                print(f"   Using FOV radius: {fov_radius} pixels")
                visualize_gaze_green_dot_red_fov(frames, df, viz_path, fov_radius=fov_radius, fps=fps)
            else:
                print("9. Skipping visualization video (no valid gaze data)")
        else:
            print("9. ⚡ Skipping visualization video (--no-viz)")
        
        print(f"✅ {video_name} processing completed!")
        
    except Exception as e:
        print(f"❌ Error processing {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()


def process_single_video_holoassist(video_path, gaze_path, output_dir, action_data=None, fps=24.46, skip_viz=False, viz_only=False, video_name=None):
    """Process a single HoloAssist video
    
    Args:
        video_path: Path to video file
        gaze_path: Path to gaze data file (.csv)
        output_dir: Output directory
        action_data: Action annotation data (optional)
        fps: Frames per second
        skip_viz: Skip visualization generation
        viz_only: Only regenerate visualization (requires existing fixation data)
        video_name: Video name (if None, extracted from path)
    """
    # Use provided video_name or extract from parent directory (session name)
    if video_name is None:
        # Get session name from parent directory of video file (e.g., R005-7July-GoPro)
        video_name = Path(video_path).parent.parent.name
    print(f"\n{'='*60}")
    if viz_only:
        print(f"Processing (VIZ-ONLY): {video_name}")
    else:
        print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Create output directory
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # In viz-only mode, load existing fixation data and skip to visualization
    if viz_only:
        fixation_csv = os.path.join(video_output_dir, f"{video_name}_fixation_dataset.csv")
        if not os.path.exists(fixation_csv):
            print(f"   ❌ Fixation data not found: {fixation_csv}")
            return
        
        print("Loading existing fixation data...")
        fixation_df = pd.read_csv(fixation_csv)
        print(f"   Loaded {len(fixation_df)} fixation segments")
        
        # Load video frames for visualization
        print("Loading video frames...")
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"   Loaded {len(frames)} frames ({frame_width}x{frame_height})")
        
        # Parse gaze data (holoassist format - .csv)
        print("Parsing gaze data...")
        gaze_data = parse_holoassist_gaze(gaze_path, fps=fps, video_resolution=(frame_width, frame_height))
        if gaze_data is None or gaze_data.size == 0:
            print("   ❌ Failed to parse gaze data")
            return
        
        # Convert numpy array to DataFrame
        gaze_type_map = {0: 'untracked', 1: 'fixation', 2: 'saccade', 3: 'unknown', 4: 'truncated'}
        gaze_text_data = [[gaze_data[i, 0], gaze_data[i, 1], gaze_type_map[int(gaze_data[i, 2])]] for i in range(len(gaze_data))]
        df = pd.DataFrame(gaze_text_data, columns=["px", "py", "gaze_type"])
        df['frame_idx'] = df.index
        
        # Generate visualization only
        print("Generating visualization...")
        viz_path = os.path.join(video_output_dir, f"{video_name}_gaze_visualization.mp4")
        visualize_gaze_green_dot_red_fov(frames, df, viz_path, fov_radius=100, fps=fps)
        print(f"   ✅ Visualization saved: {viz_path}")
        return
    
    try:
        # 1. Load video and get resolution
        print("1. Loading video frames...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get video resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"   Total frames loaded: {len(frames)}")
        print(f"   Video resolution: {frame_width}x{frame_height}")
        
        # 2. Parse gaze data (HoloAssist CSV format)
        print("2. Parsing HoloAssist gaze data...")
        if not os.path.exists(gaze_path):
            print(f"   ❌ Gaze file not found: {gaze_path}")
            return
        
        gaze_data = parse_holoassist_gaze(gaze_path, fps=fps, video_resolution=(frame_width, frame_height))
        
        # Check if gaze data is empty
        if gaze_data.size == 0:
            print(f"   ❌ No valid gaze data found, skipping {video_name}")
            return
        
        # 3. Create dataframe
        print("3. Creating dataframe...")
        gaze_type_map = {
            0: 'untracked',
            1: 'fixation',
            2: 'saccade',
            3: 'unknown',
            4: 'truncated'
        }
        gaze_text_data = [
            [gaze_data[i, 0], gaze_data[i, 1], gaze_type_map[int(gaze_data[i, 2])]]
            for i in range(len(gaze_data))
        ]

        # DataFrame 생성
        df = pd.DataFrame(gaze_text_data, columns=["px", "py", "gaze_type"])

        # timestamp 컬럼 추가
        df['frame_idx'] = df.index
        df['time_seconds'] = df['frame_idx'] / fps

        # time을 ms 단위로 변환
        df['time_ms'] = df['time_seconds'] * 1000

        df['timestamp'] = df['time_seconds'].apply(seconds_to_timestamp)
        
        # Map actions if available (HoloAssist has fine-grained action annotations)
        if action_data is not None:
            df = map_actions_to_gaze(df, action_data, video_name)
        print('Columns: ', df.columns)
        
        # 4. Extract fixations
        print("4. Extracting fixations...")
        if len(df) == 0:
            print("   ⚠️ Empty dataframe, skipping fixation extraction")
            fixations = []
        else:
            # Calculate radius_thresh dynamically based on video resolution
            # Target: 96 pixels (same as EGTEA baseline)
            target_pixels = 96
            radius_thresh = target_pixels / frame_width
            print(f"   Calculated radius_thresh: {radius_thresh:.4f} ({target_pixels} pixels for {frame_width}px width)")
            fixations = extract_fixation_segments(df, radius_thresh=radius_thresh, duration_thresh=2, gap_thresh=0.01)
        
        # 5. Extract confusion segments
        print("5. Extracting confusion segments...")
        if len(df) == 0:
            print("   ⚠️ Empty dataframe, skipping confusion extraction")
            confusion_segments = []
        else:
            confusion_segments = extract_confusion_segments(df, duration_thresh=0.5)
        
        # 6. Create fixation dataset
        print("6. Creating fixation dataset...")
        if action_data is not None and fixations:
            fixation_dataset = create_segment_dataset_with_actions(fixations, df, action_data, video_name)
        else:
            fixation_dataset = pd.DataFrame(fixations) if fixations else pd.DataFrame()
        
        # 7. Save results
        print("7. Saving results...")
        
        # Save DataFrame
        df_path = os.path.join(video_output_dir, f"{video_name}_total_metadata.csv")
        df.to_csv(df_path, index=False)
        print(f"   ✅ Gaze dataframe saved: {df_path}")
        
        # Save fixation dataset
        if not fixation_dataset.empty:
            fixation_path = os.path.join(video_output_dir, f"{video_name}_fixation_dataset.csv")
            fixation_dataset.to_csv(fixation_path, index=False)
            print(f"   ✅ Fixation dataset saved: {fixation_path}")
        
        # Save confusion dataset (disabled - not needed)
        # if confusion_segments:
        #     if action_data is not None:
        #         confusion_df = create_segment_dataset_with_actions(confusion_segments, df, action_data, video_name)
        #     else:
        #         confusion_df = pd.DataFrame(confusion_segments)
        #     confusion_path = os.path.join(video_output_dir, f"{video_name}_confusion_dataset.csv")
        #     confusion_df.to_csv(confusion_path, index=False)
        #     print(f"   ✅ Confusion dataset saved: {confusion_path}")
        
        # 8. Create timeline visualization
        if not skip_viz:
            print("8. Creating timeline visualization...")
            if not fixation_dataset.empty or confusion_segments:
                # Get confusion DataFrame for plotting
                if confusion_segments:
                    if action_data is not None:
                        confusion_df_for_plot = create_segment_dataset_with_actions(confusion_segments, df, action_data, video_name)
                    else:
                        confusion_df_for_plot = pd.DataFrame(confusion_segments)
                else:
                    confusion_df_for_plot = pd.DataFrame()
                    
                # Create timeline plot
                video_duration = df['time_seconds'].max() if len(df) > 0 else None
                timeline_path = os.path.join(video_output_dir, f"{video_name}_timeline.png")
                plot_gaze_segments(fixation_dataset, confusion_df_for_plot, video_duration, timeline_path)
            else:
                print("   ⚠️ No fixation or confusion data to visualize")
        else:
            print("8. ⚡ Skipping timeline visualization (--no-viz)")

        # 9. Create visualization video
        if not skip_viz:
            if len(df) > 0 and len(frames) > 0:
                print("9. Creating visualization video with green dot + red FOV...")
                viz_path = os.path.join(video_output_dir, f"{video_name}_gaze_visualization.mp4")
                # Calculate FOV radius dynamically based on resolution
                # Target: ~15.6% of width (170 pixels for 1088px width baseline)
                fov_radius = int(0.156 * frame_width)
                print(f"   Using FOV radius: {fov_radius} pixels")
                visualize_gaze_green_dot_red_fov(frames, df, viz_path, fov_radius=fov_radius, fps=fps)
            else:
                print("9. Skipping visualization video (no valid gaze data)")
        else:
            print("9. ⚡ Skipping visualization video (--no-viz)")
        
        print(f"✅ {video_name} processing completed!")
        
    except Exception as e:
        print(f"❌ Error processing {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main processing function"""
    # Path configuration - Update these paths to your local setup
    base_dir = "path/to/EGTEA_gaze"  # Set your EGTEA gaze dataset path
    video_dir = os.path.join(base_dir, "downloaded_videos")
    gaze_dir = os.path.join(base_dir, "gaze_data", "gaze_data")
    output_dir = os.path.join(base_dir, "processed_results_v2")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load action data 
    action_data = None
    action_file_path = os.path.join(base_dir, "raw_annotations/action_labels.csv")  # Modify to actual path
    if os.path.exists(action_file_path):
        print("Loading action data...")
        action_data = pd.read_csv(action_file_path, sep=';')
        print(f"Action data loaded: {action_data.shape}")
    else:
        print("Action data not found, proceeding without action information")
    
    # Find all video files
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    print(f"\nFound {len(video_files)} video files")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        gaze_path = os.path.join(gaze_dir, f"{video_name}.txt")
        
        print(f"\n[{i}/{len(video_files)}] Processing {video_name}...")
        
        if not os.path.exists(gaze_path):
            print(f"❌ Gaze file not found for {video_name}, skipping...")
            continue
        
        process_single_video(video_path, gaze_path, output_dir, action_data)
    
    print(f"\n{'='*60}")
    print("All videos processed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")

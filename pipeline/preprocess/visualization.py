"""
Visualization functions for EGTEA gaze processing
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import re
from pathlib import Path


def plot_gaze_segments(fixation_df, confusion_df, video_duration=None, save_path=None):
    """
    Plot fixation and confusion segments on a timeline
    
    Args:
        fixation_df: DataFrame with fixation segments
        confusion_df: DataFrame with confusion segments
        video_duration: Optional video duration to set x-axis limit
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(15, 2))

    # Optional: Set video end time
    if video_duration:
        ax.set_xlim(0, video_duration)

    # Determine column names (support both formats)
    start_col = 'start_time_seconds' if 'start_time_seconds' in fixation_df.columns else 'start_time'
    duration_col = 'duration_seconds' if 'duration_seconds' in fixation_df.columns else 'duration'

    # Plot fixation as red bars
    for i, (_, row) in enumerate(fixation_df.iterrows()):
        start = row[start_col]
        duration = row[duration_col]
        ax.broken_barh([(start, duration)], (1, 0.8), facecolors='tab:red', label='Fixation' if i == 0 else "")

    # Determine column names for confusion (support both formats)
    if not confusion_df.empty:
        conf_start_col = 'start_time_seconds' if 'start_time_seconds' in confusion_df.columns else 'start_time'
        conf_duration_col = 'duration_seconds' if 'duration_seconds' in confusion_df.columns else 'duration'
    
    # Plot confusion as green bars
    for i, (_, row) in enumerate(confusion_df.iterrows()):
        start = row[conf_start_col]
        duration = row[conf_duration_col]
        ax.broken_barh([(start, duration)], (0, 0.8), facecolors='tab:green', label='Confusion' if i == 0 else "")

    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(['Confusion', 'Fixation'])
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Gaze Timeline: Fixation vs. Confusion')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    ax.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Timeline plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()  # Free memory


def visualize_gaze_with_trail(frames, gaze_df, output_path, fps=24, trail_length=30):
    """
    Visualize gaze points and trails on video frames and save as video
    
    Args:
        frames: List of video frames
        gaze_df: Gaze dataframe with px, py, gaze_type columns
        output_path: Output video file path
        fps: Frame rate
        trail_length: Trail length (number of frames)
    """
    if len(frames) == 0:
        print("No frames to process")
        return
        
    if len(gaze_df) == 0:
        print("No gaze data to visualize")
        return
    
    # Video information
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Color definitions (BGR format)
    colors = {
        'fixation': (0, 255, 128),      # Bright green
        'saccade': (255, 255, 0),       # Cyan
        'unknown': (0, 165, 255),       # Orange
        'untracked': (128, 128, 128)    # Gray
    }
    
    print(f"Processing {len(frames)} frames with trail visualization...")
    
    # List to store trail points
    trail_points = []
    
    for frame_idx in range(len(frames)):
        # Copy current frame
        frame = frames[frame_idx].copy()
        
        # Check gaze data for this frame
        if frame_idx < len(gaze_df):
            gaze_row = gaze_df.iloc[frame_idx]
            px_norm = gaze_row['px']
            py_norm = gaze_row['py']
            gaze_type = gaze_row['gaze_type']
            
            # Validate gaze coordinates
            if pd.notna(px_norm) and pd.notna(py_norm) and gaze_type != 'untracked':
                # Convert normalized coordinates to pixel coordinates
                px = int(px_norm * width)
                py = int(py_norm * height)
                
                # Validate pixel coordinates
                if 0 <= px < width and 0 <= py < height:
                    # Add current gaze point to trail
                    trail_points.append({
                        'point': (px, py),
                        'type': gaze_type,
                        'frame': frame_idx
                    })
                    
                    # Limit trail length
                    if len(trail_points) > trail_length:
                        trail_points.pop(0)
        
        # Draw trail (older points are more transparent)
        for i, trail_point in enumerate(trail_points):
            point = trail_point['point']
            point_type = trail_point['type']
            
            # Validate point coordinates before drawing
            if isinstance(point, tuple) and len(point) == 2:
                px, py = point
                if 0 <= px < width and 0 <= py < height:
                    # Calculate transparency (newer points are more opaque)
                    alpha = (i + 1) / len(trail_points)
                    color = colors.get(point_type, (255, 255, 255))
                    
                    # Draw trail points (size also changes gradually)
                    radius = int(3 + alpha * 5)
                    cv2.circle(frame, point, radius, color, -1)
        
        # Highlight current gaze point
        if trail_points:
            current_point = trail_points[-1]
            point = current_point['point']
            point_type = current_point['type']
            
            # Validate current point before drawing
            if isinstance(point, tuple) and len(point) == 2:
                px, py = point
                if 0 <= px < width and 0 <= py < height:
                    color = colors.get(point_type, (255, 255, 255))
                    
                    # Display current point prominently
                    cv2.circle(frame, point, 12, color, -1)  # Filled circle
                    cv2.circle(frame, point, 15, color, 3)   # Border circle
                    
                    # Display gaze type text
                    text_x = min(px + 20, width - 100)  # Prevent text from going off screen
                    text_y = max(py - 20, 30)
                    cv2.putText(frame, point_type, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display frame number
        cv2.putText(frame, f"Frame: {frame_idx + 1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add frame to video
        out.write(frame)
        
        # Print progress
        if (frame_idx + 1) % 100 == 0:
            print(f"Processed {frame_idx + 1}/{len(frames)} frames")
    
    # Complete video saving
    out.release()
    print(f"Video with trail saved to: {output_path}")


def extract_and_save_gifs(dataset, frames, fps, save_dir, tag, gaze_df=None):
    """
    Extract and save GIF segments from video frames for given dataset
    
    Args:
        dataset: DataFrame with segment information (start_time_seconds, end_time_seconds, etc.)
        frames: List of video frames
        fps: Frame rate
        save_dir: Directory to save GIFs
        tag: Tag for naming (e.g., 'fixation', 'confusion')
        gaze_df: Optional gaze dataframe for overlay visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    total_frames = len(frames)

    # Color definitions for gaze visualization (BGR format)
    colors = {
        'fixation': (0, 255, 128),      # Bright green
        'saccade': (255, 255, 0),       # Cyan
        'unknown': (0, 165, 255),       # Orange
        'untracked': (128, 128, 128)    # Gray
    }

    for idx, row in dataset.iterrows():
        try:
            start_sec = row['start_time_seconds']
            end_sec = row['end_time_seconds']
            duration = end_sec - start_sec
            print(f"[{tag}_{idx}] ⏱ {start_sec:.2f}s → {end_sec:.2f}s (duration: {duration:.2f}s)")

            start_idx = int(start_sec * fps)
            end_idx = int(end_sec * fps)
            print(f"[{tag}_{idx}] Frame range: {start_idx} → {end_idx}")

            if start_idx >= end_idx or start_idx >= total_frames:
                print(f"⚠️ Skipped {tag}_{idx}: Invalid frame range ({start_idx} → {end_idx})")
                continue
            end_idx = min(end_idx, total_frames)

            selected_frames = frames[start_idx:end_idx]

            if not selected_frames:
                print(f"⚠️ Skipped {tag}_{idx}: No frames selected")
                continue

            # Include additional info in filename
            center_x = row.get('center_x', 0)
            center_y = row.get('center_y', 0)
            action_info = row.get('action_caption', 'no_action')[:20] if row.get('action_caption') else 'no_action'
            
            # Remove all characters that are invalid for filenames
            action_info = re.sub(r'[<>:"/\\|?*;, ]', '_', action_info)
            action_info = re.sub(r'_+', '_', action_info)  # Replace multiple underscores with single
            action_info = action_info.strip('_')  # Remove leading/trailing underscores
            
            filename = f"{tag}_{idx}_dur{duration:.1f}s_pos({center_x:.2f},{center_y:.2f})_{action_info}.gif"
            output_path = os.path.join(save_dir, filename)

            gif_frames = []
            height, width, _ = selected_frames[0].shape

            # Add gaze overlay if gaze data is provided
            for frame_offset, frame in enumerate(selected_frames):
                frame_copy = frame.copy()
                current_frame_idx = start_idx + frame_offset
                
                # Add gaze visualization if gaze data is available
                if gaze_df is not None and current_frame_idx < len(gaze_df):
                    gaze_row = gaze_df.iloc[current_frame_idx]
                    px_norm = gaze_row.get('px')
                    py_norm = gaze_row.get('py')
                    gaze_type = gaze_row.get('gaze_type', 'unknown')
                    
                    if pd.notna(px_norm) and pd.notna(py_norm):
                        # Convert normalized coordinates to pixel coordinates
                        px = int(px_norm * width)
                        py = int(py_norm * height)
                        
                        if 0 <= px < width and 0 <= py < height:
                            color = colors.get(gaze_type, (255, 255, 255))
                            
                            # Draw gaze point
                            cv2.circle(frame_copy, (px, py), 8, color, -1)  # Filled circle
                            cv2.circle(frame_copy, (px, py), 12, color, 2)  # Border circle
                            
                            # Add gaze type text
                            text_x = min(px + 15, width - 80)
                            text_y = max(py - 15, 25)
                            cv2.putText(frame_copy, gaze_type, (text_x, text_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Add segment info overlay
                cv2.putText(frame_copy, f"{tag.upper()} #{idx}", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_copy, f"Time: {start_sec + frame_offset/fps:.2f}s", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Convert BGR to RGB for GIF
                rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                gif_frames.append(rgb_frame)

            # Save as GIF with slower fps for better viewing
            imageio.mimsave(output_path, gif_frames, fps=8)
            print(f"✅ Saved GIF: {output_path}")

        except Exception as e:
            print(f"❌ Failed {tag}_{idx}: {e}")
            import traceback
            traceback.print_exc()


def visualize_gaze_green_dot(frames, gaze_df, output_path, fps=24):
    """
    Visualize gaze points as bright green dots on video frames and save as video
    Simple visualization with only green dots (no text, no trail) for model recognition
    
    Args:
        frames: List of video frames
        gaze_df: Gaze dataframe with px, py, gaze_type columns
        output_path: Output video file path
        fps: Frame rate
    """
    if len(frames) == 0:
        print("No frames to process")
        return
        
    if len(gaze_df) == 0:
        print("No gaze data to visualize")
        return
    
    # Video information
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Bright green color (BGR format) - highly visible for models
    bright_green = (0, 255, 0)  # Pure bright green
    dark_outline = (0, 0, 0)     # Black outline for contrast
    
    print(f"Processing {len(frames)} frames with green dot visualization...")
    
    for frame_idx in range(len(frames)):
        # Copy current frame
        frame = frames[frame_idx].copy()
        
        # Check gaze data for this frame
        if frame_idx < len(gaze_df):
            gaze_row = gaze_df.iloc[frame_idx]
            px_norm = gaze_row['px']
            py_norm = gaze_row['py']
            gaze_type = gaze_row['gaze_type']
            
            # Validate gaze coordinates
            if pd.notna(px_norm) and pd.notna(py_norm) and gaze_type != 'untracked':
                # Convert normalized coordinates to pixel coordinates
                px = int(px_norm * width)
                py = int(py_norm * height)
                
                # Validate pixel coordinates
                if 0 <= px < width and 0 <= py < height:
                    # Draw black outline for better visibility
                    cv2.circle(frame, (px, py), 18, dark_outline, 3)  # Outer black border
                    
                    # Draw bright green filled circle
                    cv2.circle(frame, (px, py), 15, bright_green, -1)  # Filled green circle
                    
                    # Draw inner black circle for definition
                    cv2.circle(frame, (px, py), 12, dark_outline, 2)  # Inner border
                    
                    # Draw center bright green dot
                    cv2.circle(frame, (px, py), 10, bright_green, -1)  # Center dot
        
        # Add frame to video
        out.write(frame)
        
        # Print progress
        if (frame_idx + 1) % 100 == 0:
            print(f"Processed {frame_idx + 1}/{len(frames)} frames")
    
    # Complete video saving
    out.release()
    print(f"Video with green dot saved to: {output_path}")


def visualize_gaze_green_dot_red_fov(frames, gaze_df, output_path, fov_radius=100, fps=24):
    """
    Visualize gaze points as bright green dots with red FOV circle on video frames
    
    Args:
        frames: List of video frames
        gaze_df: Gaze dataframe with px, py, gaze_type columns
        output_path: Output video file path
        fov_radius: FOV radius in pixels (default: 100)
        fps: Frame rate
    """
    if len(frames) == 0:
        print("No frames to process")
        return
        
    if len(gaze_df) == 0:
        print("No gaze data to visualize")
        return
    
    # Video information
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Color definitions (BGR format)
    bright_green = (0, 255, 0)   # Pure bright green for gaze point
    dark_outline = (0, 0, 0)     # Black outline for contrast
    red_fov = (0, 0, 255)        # Bright red for FOV area
    
    print(f"Processing {len(frames)} frames with green dot + red FOV visualization (radius={fov_radius}px)...")
    
    for frame_idx in range(len(frames)):
        # Copy current frame
        frame = frames[frame_idx].copy()
        
        # Check gaze data for this frame
        if frame_idx < len(gaze_df):
            gaze_row = gaze_df.iloc[frame_idx]
            px_norm = gaze_row['px']
            py_norm = gaze_row['py']
            gaze_type = gaze_row['gaze_type']
            
            # Validate gaze coordinates
            if pd.notna(px_norm) and pd.notna(py_norm) and gaze_type != 'untracked':
                # Convert normalized coordinates to pixel coordinates
                px = int(px_norm * width)
                py = int(py_norm * height)
                
                # Validate pixel coordinates
                if 0 <= px < width and 0 <= py < height:
                    # Draw red FOV circle (outer area)
                    cv2.circle(frame, (px, py), fov_radius, red_fov, 3)  # Red FOV border
                    
                    # Draw simple gaze point: black outline + green filled circle
                    cv2.circle(frame, (px, py), 10, dark_outline, 2)  # Black outline
                    cv2.circle(frame, (px, py), 8, bright_green, -1)   # Green filled circle
        
        # Add frame to video
        out.write(frame)
        
        # Print progress
        if (frame_idx + 1) % 100 == 0:
            print(f"Processed {frame_idx + 1}/{len(frames)} frames")
    
    # Complete video saving
    out.release()
    print(f"Video with green dot + red FOV saved to: {output_path}")


def seconds_to_timestamp(seconds):
    """Convert seconds to timestamp format HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"





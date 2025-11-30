"""
Gaze Metadata Processing - Multi-Dataset Version
Supports EGTEA, Ego4D, and EgoExoLearn datasets
Uses functions from the preprocess module for better code organization
"""

import pandas as pd
import os
import glob
import argparse
import json
from pathlib import Path

# Import functions from preprocess module
from preprocess import process_single_video, process_single_video_ego4d, process_single_video_egoexo, process_single_video_holoassist

# Get pipeline directory dynamically
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

#   python step1_extract_fixation.py --dataset egtea
#   python step1_extract_fixation.py --dataset ego4d --fps 30
#   python step1_extract_fixation.py --dataset egoexo --fps 30
#   python step1_extract_fixation.py --dataset egoexo --fps 30 --no-viz  # Skip visualization (faster)
#   python step1_extract_fixation.py --dataset holoassist --fps 24.46 --no-viz
#   python step1_extract_fixation.py --dataset holoassist --fps 24.46 --viz-only  # Only regenerate visualization

def process_egtea(skip_viz=False, viz_only=False):
    """Process EGTEA dataset"""
    print("🚀 Starting EGTEA Gaze Metadata Processing")
    print("=" * 60)
    if viz_only:
        print("🎨 VISUALIZATION ONLY MODE (regenerating visualization)")
    elif skip_viz:
        print("⚡ Visualization DISABLED (faster processing)")
    else:
        print("🎨 Visualization ENABLED")
    
    # Path configuration
    base_dir = os.path.join(PIPELINE_DIR, "raw_gaze_dataset", "egtea")
    video_dir = os.path.join(base_dir, "videos")
    gaze_dir = os.path.join(base_dir, "gaze_data")
    output_dir = os.path.join(PIPELINE_DIR, "final_data", "egtea", "metadata")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load action data 
    action_data = None
    action_file_path = os.path.join(base_dir, "raw_annotations", "action_labels.csv")
    if os.path.exists(action_file_path):
        print("Loading action data...")
        action_data = pd.read_csv(action_file_path, sep=';')
        print(f"Action data loaded: {action_data.shape}")
    else:
        print("Action data not found, proceeding without action information")
    
    # Find all video files
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    print(f"\nFound {len(video_files)} video files")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        gaze_path = os.path.join(gaze_dir, f"{video_name}.txt")
        
        print(f"\n[{i}/{len(video_files)}] Processing {video_name}...")
        
        # Check if already processed
        output_check = os.path.join(output_dir, video_name, f"{video_name}_fixation_dataset.csv")
        viz_check = os.path.join(output_dir, video_name, f"{video_name}_gaze_visualization.mp4")
        
        if viz_only:
            # In viz-only mode, skip if visualization already exists
            if os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (visualization already exists)")
                continue
            # Check if fixation data exists (required for viz-only mode)
            if not os.path.exists(output_check):
                print(f"❌ Fixation data not found for {video_name}, skipping...")
                continue
        else:
            # Normal mode: skip if output already exists
            if os.path.exists(output_check) and os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (already processed)")
                continue
        
        if not os.path.exists(gaze_path):
            print(f"❌ Gaze file not found for {video_name}, skipping...")
            continue
        
        process_single_video(video_path, gaze_path, output_dir, action_data, skip_viz=skip_viz, viz_only=viz_only)
    
    print(f"\n{'='*60}")
    print("All videos processed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


def process_ego4d(fps=30, skip_viz=False, viz_only=False):
    """Process Ego4D dataset"""
    print("🚀 Starting Ego4D Gaze Metadata Processing")
    print("=" * 60)
    if viz_only:
        print("🎨 VISUALIZATION ONLY MODE (regenerating visualization)")
    elif skip_viz:
        print("⚡ Visualization DISABLED (faster processing)")
    else:
        print("🎨 Visualization ENABLED")
    
    # Path configuration
    base_dir = os.path.join(PIPELINE_DIR, "raw_gaze_dataset", "ego4d", "v2")
    video_dir = os.path.join(base_dir, "gaze_videos", "v2", "full_scale")
    gaze_dir = os.path.join(base_dir, "gaze")
    output_dir = os.path.join(PIPELINE_DIR, "final_data", "ego4d", "metadata")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Ego4D doesn't have action data by default
    action_data = None
    print("Note: Ego4D processing without action data")
    
    # Find all video files
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    print(f"\nFound {len(video_files)} video files")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        gaze_path = os.path.join(gaze_dir, f"{video_name}.csv")
        
        print(f"\n[{i}/{len(video_files)}] Processing {video_name}...")
        
        # Check if already processed
        output_check = os.path.join(output_dir, video_name, f"{video_name}_fixation_dataset.csv")
        viz_check = os.path.join(output_dir, video_name, f"{video_name}_gaze_visualization.mp4")
        
        if viz_only:
            # In viz-only mode, skip if visualization already exists
            if os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (visualization already exists)")
                continue
            # Check if fixation data exists (required for viz-only mode)
            if not os.path.exists(output_check):
                print(f"❌ Fixation data not found for {video_name}, skipping...")
                continue
        else:
            # Normal mode: skip if both files exist
            if os.path.exists(output_check) and os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (already processed)")
                continue
        
        if not os.path.exists(gaze_path):
            print(f"❌ Gaze file not found for {video_name}, skipping...")
            continue
        
        process_single_video_ego4d(video_path, gaze_path, output_dir, action_data, fps=fps, skip_viz=skip_viz, viz_only=viz_only)
    
    print(f"\n{'='*60}")
    print("All videos processed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


def process_egoexo(fps=30, skip_viz=False, viz_only=False):
    """Process EgoExoLearn dataset"""
    print("🚀 Starting EgoExoLearn Gaze Metadata Processing")
    print("=" * 60)
    if viz_only:
        print("🎨 VISUALIZATION ONLY MODE (regenerating visualization)")
    elif skip_viz:
        print("⚡ Visualization DISABLED (faster processing)")
    else:
        print("🎨 Visualization ENABLED")
    
    # Path configuration
    base_dir = os.path.join(PIPELINE_DIR, "raw_gaze_dataset", "egoexolearn", "full")
    video_dir = base_dir  # Videos are in the root directory
    gaze_dir = os.path.join(base_dir, "gazes_30fps_npy")
    annotation_dir = os.path.join(base_dir, "annotation")
    output_dir = os.path.join(PIPELINE_DIR, "final_data", "egoexo", "metadata")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    print(f"\nFound {len(video_files)} video files")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        gaze_path = os.path.join(gaze_dir, f"{video_name}.npy")
        annotation_path = os.path.join(annotation_dir, f"{video_name}.json")
        
        print(f"\n[{i}/{len(video_files)}] Processing {video_name}...")
        
        # Check if already processed
        output_check = os.path.join(output_dir, video_name, f"{video_name}_fixation_dataset.csv")
        viz_check = os.path.join(output_dir, video_name, f"{video_name}_gaze_visualization.mp4")
        
        if viz_only:
            # In viz-only mode, skip if visualization already exists
            if os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (visualization already exists)")
                continue
            # Check if fixation data exists (required for viz-only mode)
            if not os.path.exists(output_check):
                print(f"❌ Fixation data not found for {video_name}, skipping...")
                continue
        else:
            # Normal mode: skip if both files exist
            if os.path.exists(output_check) and os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (already processed)")
                continue
        
        if not os.path.exists(gaze_path):
            print(f"❌ Gaze file not found for {video_name}, skipping...")
            continue
        
        # Load action data (annotations) for this video
        action_data = None
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r') as f:
                    action_data = json.load(f)
                print(f"   ✅ Loaded {len(action_data)} annotations")
            except Exception as e:
                print(f"   ⚠️ Failed to load annotation: {e}")
                action_data = None
        else:
            print(f"   ⚠️ No annotation file found")
        
        process_single_video_egoexo(video_path, gaze_path, output_dir, action_data, fps=fps, skip_viz=skip_viz, viz_only=viz_only)
    
    print(f"\n{'='*60}")
    print("All videos processed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


def process_egoexo_lab(fps=30, skip_viz=False, viz_only=False):
    """Process EgoExoLearn Lab dataset with fine-grained annotations"""
    print("🚀 Starting EgoExoLearn Lab Gaze Metadata Processing")
    print("=" * 60)
    if viz_only:
        print("🎨 VISUALIZATION ONLY MODE (regenerating visualization)")
    elif skip_viz:
        print("⚡ Visualization DISABLED (faster processing)")
    else:
        print("🎨 Visualization ENABLED")
    
    # Path configuration
    base_dir = os.path.join(PIPELINE_DIR, "raw_gaze_dataset", "egoexolearn", "full")
    video_dir = base_dir  # Videos are in the root directory
    gaze_dir = os.path.join(base_dir, "gazes_30fps_npy")
    fine_annotation_file = os.path.join(PIPELINE_DIR, "raw_gaze_dataset", "egoexolearn", "annotations", "fine_annotation_trainvaltest_en.csv")
    output_dir = os.path.join(PIPELINE_DIR, "final_data", "egoexo", "metadata", "lab")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load fine-grained annotation data
    print("Loading fine-grained annotations...")
    fine_annotations_df = pd.read_csv(fine_annotation_file)
    print(f"Total annotations loaded: {len(fine_annotations_df)}")
    
    # Filter lab scene only
    lab_annotations = fine_annotations_df[fine_annotations_df['scene'] == 'lab'].copy()
    print(f"Lab annotations: {len(lab_annotations)}")
    
    # Get unique lab video IDs
    lab_video_ids = lab_annotations['video_uid'].unique()
    print(f"Found {len(lab_video_ids)} unique lab videos")
    
    # Process each lab video
    processed = 0
    skipped = 0
    
    for i, video_name in enumerate(sorted(lab_video_ids), 1):
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        gaze_path = os.path.join(gaze_dir, f"{video_name}.npy")
        
        print(f"\n[{i}/{len(lab_video_ids)}] Processing {video_name}...")
        
        # Check if already processed
        output_check = os.path.join(output_dir, video_name, f"{video_name}_fixation_dataset.csv")
        viz_check = os.path.join(output_dir, video_name, f"{video_name}_gaze_visualization.mp4")
        
        if viz_only:
            if os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (visualization already exists)")
                skipped += 1
                continue
            if not os.path.exists(output_check):
                print(f"❌ Fixation data not found for {video_name}, skipping...")
                skipped += 1
                continue
        else:
            if os.path.exists(output_check) and os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (already processed)")
                skipped += 1
                continue
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            skipped += 1
            continue
        
        if not os.path.exists(gaze_path):
            print(f"❌ Gaze file not found: {gaze_path}")
            skipped += 1
            continue
        
        # Get fine-grained annotations for this video
        video_annotations = lab_annotations[lab_annotations['video_uid'] == video_name].copy()
        
        # Convert fine-grained annotations to action_data format (list of dicts)
        action_data = []
        for _, row in video_annotations.iterrows():
            action_data.append({
                'start': row['start_sec'],
                'end': row['end_sec'],
                'textAttribute_en': row['narration_en_no_hand_prompt'],  # Using no_hand_prompt version
                'annotation_id': row['annotation_id'],
                'subset': row['subset']
            })
        
        print(f"   ✅ Loaded {len(action_data)} fine-grained annotations")
        
        # Process the video
        process_single_video_egoexo(video_path, gaze_path, output_dir, action_data, fps=fps, skip_viz=skip_viz, viz_only=viz_only)
        processed += 1
    
    print(f"\n{'='*60}")
    print("EgoExo Lab Processing Complete!")
    print(f"{'='*60}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


def process_egoexo_kitchen(fps=30, skip_viz=False, viz_only=False):
    """Process EgoExoLearn Kitchen dataset with fine-grained annotations"""
    print("🚀 Starting EgoExoLearn Kitchen Gaze Metadata Processing")
    print("=" * 60)
    if viz_only:
        print("🎨 VISUALIZATION ONLY MODE (regenerating visualization)")
    elif skip_viz:
        print("⚡ Visualization DISABLED (faster processing)")
    else:
        print("🎨 Visualization ENABLED")
    
    # Path configuration
    base_dir = os.path.join(PIPELINE_DIR, "raw_gaze_dataset", "egoexolearn", "full")
    video_dir = base_dir  # Videos are in the root directory
    gaze_dir = os.path.join(base_dir, "gazes_30fps_npy")
    fine_annotation_file = os.path.join(PIPELINE_DIR, "raw_gaze_dataset", "egoexolearn", "annotations", "fine_annotation_trainvaltest_en.csv")
    output_dir = os.path.join(PIPELINE_DIR, "final_data", "egoexo", "metadata", "kitchen_160")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video IDs from kitchen_160 directory (only process existing ones)
    print("Scanning kitchen_160 directory for existing videos...")
    all_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    print(f"Found {len(all_dirs)} directories in kitchen_160")
    
    # Load fine-grained annotation data
    print("Loading fine-grained annotations...")
    fine_annotations_df = pd.read_csv(fine_annotation_file)
    print(f"Total annotations loaded: {len(fine_annotations_df)}")
    
    # Filter annotations for videos that exist in kitchen_160 directory
    kitchen_annotations = fine_annotations_df[fine_annotations_df['video_uid'].isin(all_dirs)].copy()
    print(f"Annotations for kitchen_160 videos: {len(kitchen_annotations)}")
    
    # Get unique kitchen video IDs (from kitchen_160 directory)
    kitchen_video_ids = sorted(all_dirs)
    print(f"Processing {len(kitchen_video_ids)} videos from kitchen_160 directory")
    
    # Process each kitchen video
    processed = 0
    skipped = 0
    
    for i, video_name in enumerate(kitchen_video_ids, 1):
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        gaze_path = os.path.join(gaze_dir, f"{video_name}.npy")
        
        print(f"\n[{i}/{len(kitchen_video_ids)}] Processing {video_name}...")
        
        # Check if already processed
        output_check = os.path.join(output_dir, video_name, f"{video_name}_fixation_dataset.csv")
        viz_check = os.path.join(output_dir, video_name, f"{video_name}_gaze_visualization.mp4")
        
        if viz_only:
            if os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (visualization already exists)")
                skipped += 1
                continue
            if not os.path.exists(output_check):
                print(f"❌ Fixation data not found for {video_name}, skipping...")
                skipped += 1
                continue
        else:
            if os.path.exists(output_check) and os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {video_name} (already processed)")
                skipped += 1
                continue
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            skipped += 1
            continue
        
        if not os.path.exists(gaze_path):
            print(f"❌ Gaze file not found: {gaze_path}")
            skipped += 1
            continue
        
        # Get fine-grained annotations for this video
        video_annotations = kitchen_annotations[kitchen_annotations['video_uid'] == video_name].copy()
        
        # Convert fine-grained annotations to action_data format (list of dicts)
        action_data = []
        for _, row in video_annotations.iterrows():
            action_data.append({
                'start': row['start_sec'],
                'end': row['end_sec'],
                'textAttribute_en': row['narration_en_no_hand_prompt'],  # Using no_hand_prompt version
                'annotation_id': row['annotation_id'],
                'subset': row['subset']
            })
        
        print(f"   ✅ Loaded {len(action_data)} fine-grained annotations")
        
        # Process the video
        process_single_video_egoexo(video_path, gaze_path, output_dir, action_data, fps=fps, skip_viz=skip_viz, viz_only=viz_only)
        processed += 1
    
    print(f"\n{'='*60}")
    print("EgoExo Kitchen Processing Complete!")
    print(f"{'='*60}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


def process_holoassist(fps=24.46, skip_viz=False, viz_only=False):
    """Process HoloAssist dataset"""
    print("🚀 Starting HoloAssist Gaze Metadata Processing")
    print("=" * 60)
    if viz_only:
        print("🎨 VISUALIZATION ONLY MODE (regenerating visualization)")
    elif skip_viz:
        print("⚡ Visualization DISABLED (faster processing)")
    else:
        print("🎨 Visualization ENABLED")
    
    # Path configuration
    base_dir = os.path.join(PIPELINE_DIR, "raw_gaze_dataset", "holoassist", "full")
    output_dir = os.path.join(PIPELINE_DIR, "final_data", "holoassist", "metadata")
    annotation_file = os.path.join(base_dir, "data-annnotation-trainval-v1_1.json")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load annotation data
    annotation_data = None
    annotated_video_names = set()
    if os.path.exists(annotation_file):
        print("Loading annotation data...")
        with open(annotation_file, 'r') as f:
            annotation_data = json.load(f)
        print(f"Annotation data loaded: {len(annotation_data)} videos")
        
        # Create set of video names that have annotations
        annotated_video_names = set([v.get('video_name') for v in annotation_data if 'video_name' in v])
        print(f"Videos with annotations: {len(annotated_video_names)}")
    else:
        print("⚠️ Annotation file not found, proceeding without action information")
    
    # Find all session directories with Export_py
    session_dirs = sorted([d for d in Path(base_dir).iterdir() if d.is_dir()])
    valid_sessions = []
    skipped_no_annotation = 0
    
    for session_dir in session_dirs:
        export_py = session_dir / "Export_py"
        if not export_py.exists():
            continue
        
        video_file = export_py / "Video_pitchshift.mp4"
        gaze_csv = export_py / f"{session_dir.name}_gaze_2d.csv"
        
        if video_file.exists() and gaze_csv.exists():
            # Skip if no annotation available
            if annotation_data and session_dir.name not in annotated_video_names:
                skipped_no_annotation += 1
                continue
            
            valid_sessions.append({
                'name': session_dir.name,
                'video_path': str(video_file),
                'gaze_path': str(gaze_csv)
            })
    
    print(f"\nFound {len(valid_sessions)} valid sessions with gaze data AND annotations")
    if skipped_no_annotation > 0:
        print(f"⏭️  Skipped {skipped_no_annotation} sessions without annotations")
    
    # Process each session
    for i, session in enumerate(valid_sessions, 1):
        session_name = session['name']
        video_path = session['video_path']
        gaze_path = session['gaze_path']
        
        print(f"\n[{i}/{len(valid_sessions)}] Processing {session_name}...")
        
        # Check if already processed
        output_check = os.path.join(output_dir, session_name, f"{session_name}_fixation_dataset.csv")
        viz_check = os.path.join(output_dir, session_name, f"{session_name}_gaze_visualization.mp4")
        
        if viz_only:
            # In viz-only mode, skip if visualization already exists
            if os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {session_name} (visualization already exists)")
                continue
            # Check if fixation data exists (required for viz-only mode)
            if not os.path.exists(output_check):
                print(f"❌ Fixation data not found for {session_name}, skipping...")
                continue
        else:
            # Normal mode: skip if both files exist
            if os.path.exists(output_check) and os.path.exists(viz_check):
                print(f"⏭️  SKIPPING: {session_name} (already processed)")
                continue
        
        # Find action data for this video
        video_action_data = None
        if annotation_data is not None:
            for video_anno in annotation_data:
                if video_anno.get('video_name') == session_name:
                    video_action_data = video_anno.get('events', [])
                    # Filter for fine-grained actions only
                    video_action_data = [e for e in video_action_data if e.get('label') == 'Fine grained action']
                    print(f"   ✅ Loaded {len(video_action_data)} fine-grained actions")
                    break
        
        if video_action_data is None:
            print(f"   ⚠️ No fine-grained action annotations found for {session_name}")
        
        # Pass session_name explicitly to avoid using "Video_pitchshift" as video_name
        process_single_video_holoassist(video_path, gaze_path, output_dir, video_action_data, fps=fps, skip_viz=skip_viz, viz_only=viz_only, video_name=session_name)
    
    print(f"\n{'='*60}")
    print("All videos processed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


def main():
    """Main processing function with dataset selection"""
    parser = argparse.ArgumentParser(description='Process gaze metadata for different datasets')
    parser.add_argument('--dataset', type=str, choices=['egtea', 'ego4d', 'egoexo', 'egoexo-lab', 'kitchen', 'holoassist'], required=True,
                        help='Dataset to process: egtea, ego4d, egoexo, egoexo-lab, kitchen, or holoassist')
    parser.add_argument('--fps', type=float, default=30,
                        help='FPS for videos (default: 30, HoloAssist: 24.46)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation (faster processing)')
    parser.add_argument('--viz-only', action='store_true',
                        help='Only regenerate visualization (requires existing fixation data)')
    
    args = parser.parse_args()
    
    # Check for conflicting flags
    if args.no_viz and args.viz_only:
        print("❌ Error: --no-viz and --viz-only are mutually exclusive")
        return
    
    if args.dataset == 'egtea':
        process_egtea(skip_viz=args.no_viz, viz_only=args.viz_only)
    elif args.dataset == 'ego4d':
        process_ego4d(fps=args.fps, skip_viz=args.no_viz, viz_only=args.viz_only)
    elif args.dataset == 'egoexo':
        process_egoexo(fps=args.fps, skip_viz=args.no_viz, viz_only=args.viz_only)
    elif args.dataset == 'egoexo-lab':
        process_egoexo_lab(fps=args.fps, skip_viz=args.no_viz, viz_only=args.viz_only)
    elif args.dataset == 'kitchen':
        process_egoexo_kitchen(fps=args.fps, skip_viz=args.no_viz, viz_only=args.viz_only)
    elif args.dataset == 'holoassist':
        process_holoassist(fps=args.fps, skip_viz=args.no_viz, viz_only=args.viz_only)
    else:
        print(f"Unknown dataset: {args.dataset}")
        print("Please choose 'egtea', 'ego4d', 'egoexo', 'egoexo-lab', 'kitchen', or 'holoassist'")


if __name__ == "__main__":
    main()

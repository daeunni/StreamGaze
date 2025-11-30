"""
Step 3: QA Generation for Egocentric Videos
Generate QA tasks (Present, Past, Future) from fixation data with object annotations

Usage:

    python step3_qa_gen_v2.py --dataset egoexo \
        --metadata-file path/to/metadata.csv \
        --video-base-dir path/to/EgoExoLearn \
        --save-path os.path.join(PIPELINE_DIR, 'final_data') + '/'egoexo/qa/FINAL_afterhuma_egokitchen2 \
        --action-file path/to/EgoExoLearn/annotation
    

    # ===== Egoexo with custom paths (all tasks) =====
    CUDA_VISIBLE_DEVICES=0,1,2 python step3_qa_gen_v2.py --dataset egoexo \
        --metadata-file os.path.join(PIPELINE_DIR, 'final_data') + '/'egoexo/total_metadata_after_human_all_omithuman_FINAL.csv \
        --video-base-dir path/to/EgoExoLearn \
        --save-path os.path.join(PIPELINE_DIR, 'final_data') + '/'egoexo/qa/FINAL_afterhuman \
        --action-file path/to/EgoExoLearn/annotation
    
    CUDA_VISIBLE_DEVICES=3,4,5 python step3_qa_gen_v2.py --dataset egoexo \
        --metadata-file os.path.join(PIPELINE_DIR, 'final_data') + '/'egoexo/total_metadata_before_human_all_omithuman_FINAL.csv \
        --video-base-dir path/to/EgoExoLearn \
        --save-path os.path.join(PIPELINE_DIR, 'final_data') + '/'egoexo/qa/FINAL_beforehuman \
        --action-file path/to/EgoExoLearn/annotation
    
    CUDA_VISIBLE_DEVICES=6,7 python step3_qa_gen_v2.py --dataset ego4d 

    CUDA_VISIBLE_DEVICES=0,1,2,3 python step3_qa_gen_v2.py --dataset holoassist 


"""

import os
# Get pipeline directory dynamically  
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

import pdb
import pandas as pd
import nncore
import numpy as np
import cv2, json, random, pdb
from pprint import pprint
import matplotlib.pyplot as plt
from openai import AzureOpenAI
import os, imageio
from datetime import datetime
from PIL import Image
from tqdm import tqdm 
from openai import OpenAI
from datetime import timedelta
import base64
import re
# ours 
from qa_generation.past import (
    generate_next_after_group_qa,
    generate_never_gazed_qa,
    generate_transition_pattern_qa,
    generate_scene_reconstruction_qa
)
from qa_generation.future import (
    generate_future_action_qa,
    generate_object_remind_qa
)
from qa_generation.present import Present_object_identity_attribute
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
import torch

# Helper function for safe JSON parsing
def safe_json_parse_from_csv(json_str):
    if pd.isna(json_str) or not json_str:
        return None
        
    try:
        # First try JSON
        return json.loads(json_str)
    except:
        try:
            # Try JSON after converting single quotes to double quotes
            if isinstance(json_str, str):
                json_str_fixed = json_str.replace("'", '"')
                # Remove newline characters and clean spaces
                json_str_fixed = json_str_fixed.replace('\n', ' ').replace('\r', ' ')
                import re
                json_str_fixed = re.sub(r'\s+', ' ', json_str_fixed).strip()
                return json.loads(json_str_fixed)
        except:
            try:
                # Try evaluating as Python literal with ast.literal_eval
                import ast
                return ast.literal_eval(json_str)
            except:
                return None

# Extract representative object names and create a new column
def extract_representative_object_names(fixation_df, dataset='egtea'):
    """
    Extract only object_identity from representative_object to create new column
    Supports different column names per dataset
    """
    object_names = []
    
    # Determine column name based on dataset
    if dataset in ['ego4d', 'egoexo', 'holoassist']:
        col_name = 'exact_gaze_object'
    else:  # egtea
        col_name = 'representative_object'
    
    for _, row in fixation_df.iterrows():
        if pd.notna(row.get(col_name)):
            try:
                rep_obj = safe_json_parse_from_csv(row[col_name])
                if rep_obj and rep_obj.get("object_identity"):
                    object_names.append(rep_obj["object_identity"].strip())
                else:
                    object_names.append(None)
            except Exception as e:
                print(f"Error parsing {col_name}: {e}")
                object_names.append(None)
        else:
            object_names.append(None)
    
    return object_names

# Filter rows to keep only the longest duration for each representative_object_name
def filter_duplicate_objects_by_duration(fixation_df):
    """
    Keep only the row with longest duration among rows with same representative_object_name
    """
    print("Filtering duplicate objects by duration...")
    
    # Filter only rows with duration
    valid_rows = fixation_df[fixation_df['duration'].notna()].copy()
    print(f"Rows with valid duration: {len(valid_rows)}")
    
    # Group by representative_object_name
    object_groups = {}
    for idx, row in valid_rows.iterrows():
        obj_name = row.get('representative_object_name')
        if pd.notna(obj_name):
            if obj_name not in object_groups:
                object_groups[obj_name] = []
            object_groups[obj_name].append((idx, row))
    
    print(f"Unique objects: {len(object_groups)}")
    
    # Check duplicate objects
    duplicates = {obj: rows for obj, rows in object_groups.items() if len(rows) > 1}
    if duplicates:
        print(f"Objects with duplicates: {len(duplicates)}")
        for obj, rows in duplicates.items():
            print(f"  {obj}: {len(rows)} rows")
    
    # Select row with longest duration for each object
    selected_indices = []
    for object_name, rows in object_groups.items():
        if len(rows) == 1:
            # Add as is if no duplicate
            selected_indices.append(rows[0][0])
        else:
            # If duplicate, select one with longest duration
            longest_row = max(rows, key=lambda x: x[1]['duration'])
            selected_indices.append(longest_row[0])
            
            # Print duration info of selected row
            selected_duration = longest_row[1]['duration']
            print(f"  {object_name}: Selected row with duration {selected_duration}")
    
    # Create new dataframe with selected rows
    filtered_df = fixation_df.loc[selected_indices].copy()
    
    print(f"Original rows: {len(fixation_df)}")
    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Removed: {len(fixation_df) - len(filtered_df)} duplicate rows")
    
    return filtered_df

def create_scanpath_dictionaries(fixation_dataset, video_actions, dataset='egtea'):
    """
    Create scanpath dictionary with 2D list structure: [FOV objects, Outside objects]
    Returns: scanpath_obj_dict (2D list structure)
    Key format: (start_time, end_time) tuple to preserve duration information
    """
    import ast
    
    # Determine column names based on dataset
    if dataset in ['ego4d', 'egoexo', 'holoassist']:
        gaze_obj_col = 'exact_gaze_object'
        time_col = 'start_time'
        end_time_col = 'end_time'
    else:  # egtea
        gaze_obj_col = 'representative_object'
        time_col = 'episode_start_time'
        end_time_col = 'episode_end_time'
    
    # Initialize scanpath dictionary with 2D list: [inside FOV, outside FOV]
    scanpath_obj_dict = {}
    
    # Collect objects from fixation dataset
    for idx, row in fixation_dataset.iterrows():
        start_time = row[time_col]
        end_time = row[end_time_col]
        time_key = (start_time, end_time)  # Use (start, end) tuple as key
        
        # Initialize: [inside FOV, outside FOV]
        if time_key not in scanpath_obj_dict:
            scanpath_obj_dict[time_key] = [[], []]
        
        # Parse representative_object (inside FOV)
        rep_obj = row[gaze_obj_col]
        if isinstance(rep_obj, str) and rep_obj:
            try:
                rep_obj_dict = ast.literal_eval(rep_obj)
                if isinstance(rep_obj_dict, dict) and 'object_identity' in rep_obj_dict:
                    obj_name = rep_obj_dict['object_identity'].lower()  # Normalize to lowercase
                    if obj_name not in scanpath_obj_dict[time_key][0]:  # Prevent duplicates
                        scanpath_obj_dict[time_key][0].append(obj_name)
            except:
                pass
        
        # Parse other_objects_in_cropped_area (other objects inside FOV)
        other_objs = row['other_objects_in_cropped_area']
        if isinstance(other_objs, str) and other_objs and other_objs != '[]':
            try:
                other_objs_list = ast.literal_eval(other_objs)
                if isinstance(other_objs_list, list):
                    for obj in other_objs_list:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            obj_name = obj['object_identity'].lower()  # Normalize to lowercase
                            if obj_name not in scanpath_obj_dict[time_key][0]:  # Prevent duplicates
                                scanpath_obj_dict[time_key][0].append(obj_name)
            except:
                pass
        
        # Parse other_objects_outside_fov (outside FOV)
        outside_objs = row['other_objects_outside_fov']
        if isinstance(outside_objs, str) and outside_objs and outside_objs != '[]':
            try:
                outside_objs_list = ast.literal_eval(outside_objs)
                if isinstance(outside_objs_list, list):
                    for obj in outside_objs_list:
                        if isinstance(obj, dict) and 'object_identity' in obj:
                            obj_name = obj['object_identity'].lower()  # Normalize to lowercase
                            if obj_name not in scanpath_obj_dict[time_key][1]:  # Prevent duplicates
                                scanpath_obj_dict[time_key][1].append(obj_name)
            except:
                pass
    
    # Add action labels as string entries (not 2D lists)
    if video_actions is not None and len(video_actions) > 0:
        for _, action_row in video_actions.iterrows():
            # Use (start, end) tuple as key for actions too
            action_start = action_row['start_time_sec']
            action_end = action_row['end_time_sec']
            action_time_key = (action_start, action_end)
            action_label = action_row['Action Label'].strip()
            
            # Add action label directly as string (different from object lists)
            if action_time_key not in scanpath_obj_dict:
                scanpath_obj_dict[action_time_key] = action_label
            else:
                # If timestamp exists with objects, keep objects and log warning
                # Actions should have unique timestamps in practice
                pass
    
    return scanpath_obj_dict


class Qwen3VLClient:
    """Wrapper class to use Qwen3-VL like OpenAI client"""
    
    def __init__(self, model_name="Qwen/Qwen3-VL-30B-A3B-Instruct"):
        print(f"Loading Qwen3-VL model: {model_name}...")
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded successfully!")
    
    class ChatCompletions:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor
        
        def create(self, model=None, messages=None, temperature=0.3, max_tokens=300, **kwargs):
            """OpenAI-compatible interface for chat completions"""
            # Convert OpenAI format messages to Qwen3-VL format
            qwen_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    
                    # Simple text-only message
                    qwen_messages.append({
                        'role': role,
                        'content': [{'type': 'text', 'text': content}]
                    })
            
            # Preparation for inference
            inputs = self.processor.apply_chat_template(
                qwen_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Return OpenAI-like response
            class Choice:
                def __init__(self, text):
                    self.message = type('Message', (), {'content': text})()
            
            class Response:
                def __init__(self, text):
                    self.choices = [Choice(text)]
            
            return Response(output_text[0] if output_text else "")
    
    @property
    def chat(self):
        class Chat:
            def __init__(self, model, processor):
                self.completions = Qwen3VLClient.ChatCompletions(model, processor)
        return Chat(self.model, self.processor)


if __name__ == "__main__": 
    ''' 
    Generate QA tasks for all videos
    '''
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate QA tasks for egocentric videos')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['egtea', 'ego4d', 'egoexo', 'holoassist'],
                       help='Dataset to process')
    parser.add_argument('--metadata-file', type=str, default=None,
                       help='Path to unified metadata CSV file')
    parser.add_argument('--video-base-dir', type=str, default=None,
                       help='Base directory containing video files')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Directory to save QA results')
    parser.add_argument('--action-file', type=str, default=None,
                       help='Path to action labels file (CSV or JSON)')
    parser.add_argument('--use-human-verified', action='store_true',
                       help='Use human-verified metadata from dataset/metadata/{dataset}.csv')
    parser.add_argument('--start-pct', type=int, default=0,
                       help='Start processing from this percentage (0-100)')
    parser.add_argument('--tasks', type=str, nargs='+', 
                       choices=['present', 'past', 'future', 'all'],
                       default=['all'],
                       help='Task types to generate: present, past, future, or all (default: all)')
    args = parser.parse_args()
    
    # Process task selection
    if 'all' in args.tasks:
        generate_present = True
        generate_past = True
        generate_future = True
    else:
        generate_present = 'present' in args.tasks
        generate_past = 'past' in args.tasks
        generate_future = 'future' in args.tasks
    
    # Initialize Qwen3-VL model (replaces GPT-4o)
    client = Qwen3VLClient(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct")
    
    # Dataset-specific default configurations
    default_configs = {
        'egtea': {
            'metadata_file': os.path.join(PIPELINE_DIR, 'final_data', 'egtea', 'total_metadata.csv'),
            'video_base_dir': os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egtea', 'videos'),
            'save_path': os.path.join(PIPELINE_DIR, 'final_data', 'egtea', 'qa_raw'),
            'action_file': os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egtea', 'raw_annotations', 'action_labels.csv')
        },
        'ego4d': {
            'metadata_file': os.path.join(PIPELINE_DIR, 'final_data', 'ego4d', 'total_metadata.csv'),
            'video_base_dir': os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'ego4d', 'v2', 'gaze_videos', 'v2', 'full_scale'),
            'save_path': os.path.join(PIPELINE_DIR, 'final_data', 'ego4d', 'qa_raw'),
            'action_file': None
        },
        'egoexo': {
            'metadata_file': os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'total_metadata.csv'),
            'video_base_dir': os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egoexolearn', 'full'),
            'save_path': os.path.join(PIPELINE_DIR, 'final_data', 'egoexo', 'qa_raw'),
            'action_file': os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'egoexolearn', 'annotations')  # Directory for per-video JSON files
        },
        'holoassist': {
            'metadata_file': os.path.join(PIPELINE_DIR, 'final_data', 'holoassist', 'total_metadata.csv'),
            'video_base_dir': os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'holoassist', 'full'),
            'save_path': os.path.join(PIPELINE_DIR, 'final_data', 'holoassist', 'qa_raw'),
            'action_file': os.path.join(PIPELINE_DIR, 'raw_gaze_dataset', 'holoassist', 'full', 'data-annnotation-trainval-v1_1.json')
        }
    }
    
    # Use command-line arguments if provided, otherwise use defaults
    config = default_configs[args.dataset]
    
    # Use human-verified metadata if flag is set
    if args.use_human_verified:
        human_verified_path = os.path.join(os.path.dirname(PIPELINE_DIR), 'dataset', 'metadata', f'{args.dataset}.csv')
        print(f"📋 Using human-verified metadata: {human_verified_path}")
        metadata_file = human_verified_path
    else:
        # Use auto-generated metadata from Step 2.5
        auto_metadata_path = os.path.join(PIPELINE_DIR, 'final_data', args.dataset, 'total_metadata.csv')
        metadata_file = args.metadata_file if args.metadata_file else auto_metadata_path
    
    video_base_dir = args.video_base_dir if args.video_base_dir else config['video_base_dir']
    save_path = args.save_path if args.save_path else config['save_path']
    action_file = args.action_file if args.action_file else config['action_file']
    
    print(f"\n{'='*60}")
    print(f"Configuration for {args.dataset}:")
    print(f"  Metadata type: {'🧑 Human-verified' if args.use_human_verified else '🤖 Auto-generated (Step 2.5)'}")
    print(f"  Metadata file: {metadata_file}")
    print(f"  Video base dir: {video_base_dir}")
    print(f"  Save path: {save_path}")
    print(f"  Action file: {action_file}")
    print(f"  Task types: {', '.join(args.tasks)}")
    print(f"    - Generate Present: {generate_present}")
    print(f"    - Generate Past: {generate_past}")
    print(f"    - Generate Future: {generate_future}")
    print(f"{'='*60}\n")
    
    # Load unified metadata file
    print(f"Loading unified metadata from: {metadata_file}")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    all_fixation_data = pd.read_csv(metadata_file,
                                     sep=',',
                                     quotechar='"',
                                     skipinitialspace=True,
                                     on_bad_lines='skip',
                                     engine='python')
    print(f"Loaded metadata with {len(all_fixation_data)} rows")
    print(f"Columns: {list(all_fixation_data.columns)}")
    
    # Get unique video sources (tasks)
    # Handle different column names: prioritize 'video_name', then others
    if 'video_name' in all_fixation_data.columns:
        video_col = 'video_name'
    elif 'video_source' in all_fixation_data.columns:
        video_col = 'video_source'
    elif 'video_id' in all_fixation_data.columns:
        video_col = 'video_id'
    else:
        video_col = 'source_file'
    
    tasks = sorted(all_fixation_data[video_col].unique())
    
    # Apply start_pct slicing
    if args.start_pct > 0:
        start_idx = int(len(tasks) * args.start_pct / 100)
        tasks = tasks[start_idx:]
        print(f"🎯 Starting from {args.start_pct}% position (index {start_idx}/{len(tasks) + start_idx})")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Load action labels (once for all videos)
    all_action_labels = None
    holoassist_annotations = None
    egoexo_annotation_dir = None
    
    if args.dataset == 'egtea' and action_file and os.path.exists(action_file):
        print(f"Loading action labels from: {action_file}")
        all_action_labels = pd.read_csv(action_file,
                                        sep=';',
                                        skipinitialspace=True,
                                        skiprows=1,
                                        names=['Clip ID', 'Clip Prefix', 'Video Session', 'Starting Time (ms)', 'Ending Time (ms)', 
                                               'Action Label', 'Verb Label', 'Noun Label(s)'])
        all_action_labels.columns = all_action_labels.columns.str.strip()
        all_action_labels['start_time_sec'] = all_action_labels['Starting Time (ms)'] / 1000.0
        all_action_labels['end_time_sec'] = all_action_labels['Ending Time (ms)'] / 1000.0
        print(f"✓ Loaded {len(all_action_labels)} action labels")
    elif args.dataset == 'egoexo' and action_file and os.path.exists(action_file):
        print(f"EgoExo annotation directory: {action_file}")
        egoexo_annotation_dir = action_file
        print(f"✓ Will load per-video JSON annotations from {egoexo_annotation_dir}")
    elif args.dataset == 'holoassist' and action_file and os.path.exists(action_file):
        print(f"Loading HoloAssist annotations from: {action_file}")
        with open(action_file, 'r') as f:
            holoassist_annotations = json.load(f)
        print(f"✓ Loaded HoloAssist annotations")
    else:
        print(f"No action labels file for {args.dataset}")
    
    # Initialize aggregated task lists (across all videos)
    all_present_ident_tasks_easy = []
    all_present_ident_tasks_hard = []
    all_present_attr_tasks = []
    all_past_next_after_group_tasks = []
    all_past_never_gazed_tasks = []
    all_past_transition_pattern_tasks = []
    all_past_scene_reconstruction_tasks = []
    all_future_action_tasks = []
    all_future_remind_easy_tasks = []
    all_future_remind_hard_tasks = []
    
    # Load checkpoint if exists
    checkpoint_file = f"{save_path}/{args.dataset}_checkpoint.json"
    processed_videos = set()
    if os.path.exists(checkpoint_file):
        print(f"📂 Loading checkpoint from {checkpoint_file}")
        checkpoint = nncore.load(checkpoint_file)
        processed_videos = set(checkpoint.get('processed_videos', []))
        all_present_ident_tasks_easy = checkpoint.get('present_ident_easy', [])
        all_present_ident_tasks_hard = checkpoint.get('present_ident_hard', [])
        all_present_attr_tasks = checkpoint.get('present_attr', [])
        all_past_next_after_group_tasks = checkpoint.get('past_next_after_group', [])
        all_past_never_gazed_tasks = checkpoint.get('past_never_gazed', [])
        all_past_transition_pattern_tasks = checkpoint.get('past_transition_pattern', [])
        all_past_scene_reconstruction_tasks = checkpoint.get('past_scene_reconstruction', [])
        all_future_action_tasks = checkpoint.get('future_action', [])
        all_future_remind_easy_tasks = checkpoint.get('future_remind_easy', [])
        all_future_remind_hard_tasks = checkpoint.get('future_remind_hard', [])
        print(f"✓ Loaded checkpoint with {len(processed_videos)} processed videos")
        print(f"✓ Resuming from video {len(processed_videos) + 1}/{len(tasks)}")
    
    print(f"Found {len(tasks)} videos to process")
    print(f"Results will be saved to: {save_path}")
    print(f"Dataset: {args.dataset}")
    
    # Checkpoint save interval
    CHECKPOINT_INTERVAL = 10

    for video_idx, video_name in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Processing video {video_idx + 1}/{len(tasks)}: {video_name}")
        print(f"{'='*60}")
        
        # Skip if already processed
        if video_name in processed_videos:
            print(f"⏭️  SKIPPING: Already processed in checkpoint")
            continue
        
        # Construct video path and categories based on dataset
        video_categories = args.dataset.upper()  # EGTEA, EGO4D, EGOEXO, HOLOASSIST
        
        if args.dataset == 'egtea':
            # EGTEA uses .avi format
            video_path = os.path.join(video_base_dir, f'{video_name}.avi')
            # Fallback to .mp4 if .avi doesn't exist
            if not os.path.exists(video_path):
                video_path = os.path.join(video_base_dir, f'{video_name}.mp4')
        elif args.dataset == 'ego4d':
            # Extract UUID from CSV filename: 'uuid_fixation_merged_filtered_v2.csv' -> 'uuid'
            video_id = video_name.split('_fixation_')[0] if '_fixation_' in video_name else video_name.replace('.csv', '')
            video_path = os.path.join(video_base_dir, f'{video_id}.mp4')
        elif args.dataset == 'egoexo':
            # EgoExo format: Extract UUID from CSV filename like ego4d
            video_id = video_name.split('_fixation_')[0] if '_fixation_' in video_name else video_name.replace('.csv', '')
            video_path = os.path.join(video_base_dir, f'{video_id}.mp4')
        elif args.dataset == 'holoassist':
            # HoloAssist video structure: video_name/Export_py/Video_pitchshift.mp4
            video_path = os.path.join(video_base_dir, video_name, 'Export_py', 'Video_pitchshift.mp4')
        
        # Check if video path exists
        if video_path is None or not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            print(f"⏭️  Skipping video {video_name}...")
            continue

        # Filter fixation data for this video from unified metadata
        fixation_dataset = all_fixation_data[all_fixation_data[video_col] == video_name].copy()
        
        if len(fixation_dataset) == 0:
            print(f"❌ No fixation data found for video: {video_name}")
            print(f"⏭️  Skipping video {video_name}...")
            continue
        
        print(f"Fixation dataset shape: {fixation_dataset.shape}")
        
        # Unified metadata file already uses EGTEA column format (video_source, episode_start_time, etc.)
        # Check if required columns exist
        required_cols = ['episode_start_time', 'episode_end_time', 'duration', 'representative_object']
        missing_cols = [col for col in required_cols if col not in fixation_dataset.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"⏭️  Skipping video {video_name}...")
            continue
        
        print(f"✓ Using unified metadata with EGTEA column format")
        
        # Filter action labels for this video
        video_actions = pd.DataFrame()
        if args.dataset == 'egtea' and all_action_labels is not None:
            video_actions = all_action_labels[all_action_labels['Video Session'].str.strip() == video_name].copy()
            print(f"Found {len(video_actions)} actions for {video_name}")
        elif args.dataset == 'egoexo' and egoexo_annotation_dir is not None:
            # Load per-video JSON annotation file
            annotation_path = os.path.join(egoexo_annotation_dir, f"{video_name}.json")
            if os.path.exists(annotation_path):
                try:
                    with open(annotation_path, 'r') as f:
                        action_data = json.load(f)
                    print(f"   ✅ Loaded {len(action_data)} annotations from {video_name}.json")
                    
                    # Convert to DataFrame format
                    action_list = []
                    for annot in action_data:
                        start_sec = annot.get('start', 0)
                        end_sec = annot.get('end', 0)
                        action_label = annot.get('textAttribute_en', '')
                        if action_label:
                            action_list.append({
                                'start_time_sec': start_sec,
                                'end_time_sec': end_sec,
                                'Action Label': action_label
                            })
                    video_actions = pd.DataFrame(action_list)
                    print(f"Found {len(video_actions)} actions for {video_name}")
                except Exception as e:
                    print(f"   ⚠️ Failed to load annotation for {video_name}: {e}")
                    video_actions = pd.DataFrame()
            else:
                print(f"   ⚠️ No annotation file found: {annotation_path}")
        elif args.dataset == 'holoassist' and holoassist_annotations is not None:
            # Filter for current video (same structure as step1_extract_fixation.py)
            video_action_data = None
            for video_anno in holoassist_annotations:
                if video_anno.get('video_name') == video_name:
                    video_action_data = video_anno.get('events', [])
                    # Filter for fine-grained actions only (same as step1)
                    video_action_data = [e for e in video_action_data if e.get('label') == 'Fine grained action']
                    print(f"   ✅ Found {len(video_action_data)} fine-grained actions")
                    break
            
            if video_action_data:
                # Convert to DataFrame format
                action_list = []
                for event in video_action_data:
                    if 'attributes' in event:
                        attrs = event['attributes']
                        start_sec = event.get('start', 0)
                        end_sec = event.get('end', 0)
                        verb = attrs.get('Verb', '')
                        noun = attrs.get('Noun', '')
                        if verb and verb != 'none' and noun and noun != 'none':
                            action_label = f"{verb} {noun}"
                            action_list.append({
                                'start_time_sec': start_sec,
                                'end_time_sec': end_sec,
                                'Action Label': action_label
                            })
                video_actions = pd.DataFrame(action_list)
                print(f"Found {len(video_actions)} actions for {video_name}")
            else:
                print(f"   ⚠️ No fine-grained action annotations found for {video_name}")
                video_actions = pd.DataFrame()
        else:
            print(f"No action labels available for {args.dataset}")


        '''
        QA Generation
        '''
        # After normalization, all datasets use 'episode_start_time' and 'representative_object'
        fixation_dataset['representative_object_name'] = extract_representative_object_names(fixation_dataset, dataset='egtea')
        fixation_for_present_tasks = filter_duplicate_objects_by_duration(fixation_dataset)
        
        # Only sort if rows exist
        if len(fixation_for_present_tasks) > 0:
            fixation_for_present_tasks = fixation_for_present_tasks.sort_values(by='episode_start_time')
        else:
            print(f"⚠️ No valid fixation data after filtering for {video_name}")
            print(f"⏭️ Skipping QA generation for this video...")
            continue

        # Create scanpath dictionary with 2D list structure
        scanpath_obj_dict = create_scanpath_dictionaries(
            fixation_dataset, video_actions, dataset='egtea'
        )
        
        # Debug: Check scanpath_obj_dict structure
        total_timestamps = len(scanpath_obj_dict)
        action_timestamps = sum(1 for v in scanpath_obj_dict.values() if isinstance(v, str))
        object_timestamps = sum(1 for v in scanpath_obj_dict.values() if isinstance(v, list))
        print(f"  Scanpath dict: {total_timestamps} total timestamps")
        print(f"    - {action_timestamps} action timestamps")
        print(f"    - {object_timestamps} object timestamps")
        
        # Present tasks (conditional)
        if generate_present:
            present_ident_tasks_easy, present_ident_tasks_hard, present_attr_tasks = Present_object_identity_attribute(fixation_for_present_tasks, video_path)
        else:
            present_ident_tasks_easy, present_ident_tasks_hard, present_attr_tasks = [], [], []
        
        # Past tasks (conditional)
        if generate_past:
            past_next_after_group_tasks = generate_next_after_group_qa(scanpath_obj_dict, video_path)
            past_never_gazed_tasks = generate_never_gazed_qa(scanpath_obj_dict, video_path)
            past_transition_pattern_tasks = generate_transition_pattern_qa(scanpath_obj_dict, video_path, num_samples=5)
            past_scene_reconstruction_tasks = generate_scene_reconstruction_qa(scanpath_obj_dict, video_path)
        else:
            past_next_after_group_tasks = []
            past_never_gazed_tasks = []
            past_transition_pattern_tasks = []
            past_scene_reconstruction_tasks = []
        
        # Future tasks (conditional)
        if generate_future:
            print(f"\n  [Future Task Generation]")
            # future_action_tasks = []
            future_action_tasks = generate_future_action_qa(scanpath_obj_dict, video_path, video_categories=video_categories)
            print(f"    → Generated {len(future_action_tasks)} action prediction tasks")
            
            future_remind_easy_tasks, future_remind_hard_tasks = generate_object_remind_qa(
                scanpath_obj_dict=scanpath_obj_dict,
                video_path=video_path,
                video_categories=video_categories
            )
            print(f"    → Generated {len(future_remind_easy_tasks)} easy remind tasks")
            print(f"    → Generated {len(future_remind_hard_tasks)} hard remind tasks")
        else:
            future_action_tasks = []
            future_remind_easy_tasks = []
            future_remind_hard_tasks = []

        print(f"\n  [Summary] Generated tasks for {video_name}:")
        print(f"  Present identification tasks (easy): {len(present_ident_tasks_easy)}")
        print(f"  Present identification tasks (hard): {len(present_ident_tasks_hard)}")
        print(f"  Present attribute tasks: {len(present_attr_tasks)}")
        print(f"  Past next after group tasks: {len(past_next_after_group_tasks)}")
        print(f"  Past never gazed tasks: {len(past_never_gazed_tasks)}")
        print(f"  Past transition pattern tasks: {len(past_transition_pattern_tasks)}")
        print(f"  Past scene reconstruction tasks: {len(past_scene_reconstruction_tasks)}")
        print(f"  Future action tasks: {len(future_action_tasks)}")
        print(f"  Future remind easy tasks: {len(future_remind_easy_tasks)}")
        print(f"  Future remind hard tasks: {len(future_remind_hard_tasks)}")

        # Aggregate tasks from this video to the global lists
        all_present_ident_tasks_easy.extend(present_ident_tasks_easy)
        all_present_ident_tasks_hard.extend(present_ident_tasks_hard)
        all_present_attr_tasks.extend(present_attr_tasks)
        all_past_next_after_group_tasks.extend(past_next_after_group_tasks)
        all_past_never_gazed_tasks.extend(past_never_gazed_tasks)
        all_past_transition_pattern_tasks.extend(past_transition_pattern_tasks)
        all_past_scene_reconstruction_tasks.extend(past_scene_reconstruction_tasks)
        all_future_action_tasks.extend(future_action_tasks)
        all_future_remind_easy_tasks.extend(future_remind_easy_tasks)
        all_future_remind_hard_tasks.extend(future_remind_hard_tasks)
        
        # Mark video as processed
        processed_videos.add(video_name)
        
        # Save checkpoint every CHECKPOINT_INTERVAL videos
        if len(processed_videos) % CHECKPOINT_INTERVAL == 0:
            print(f"\n💾 Saving checkpoint ({len(processed_videos)} videos processed)...")
            checkpoint_data = {
                'processed_videos': list(processed_videos),
                'present_ident_easy': all_present_ident_tasks_easy,
                'present_ident_hard': all_present_ident_tasks_hard,
                'present_attr': all_present_attr_tasks,
                'past_next_after_group': all_past_next_after_group_tasks,
                'past_never_gazed': all_past_never_gazed_tasks,
                'past_transition_pattern': all_past_transition_pattern_tasks,
                'past_scene_reconstruction': all_past_scene_reconstruction_tasks,
                'future_action': all_future_action_tasks,
                'future_remind_easy': all_future_remind_easy_tasks,
                'future_remind_hard': all_future_remind_hard_tasks,
                'timestamp': datetime.now().isoformat()
            }
            nncore.dump(checkpoint_data, checkpoint_file, indent=2)
            print(f"✓ Checkpoint saved to {checkpoint_file}")
            
            # Save intermediate JSON files as well
            print(f"💾 Saving intermediate JSON files...")
            if generate_present:
                nncore.dump(all_present_ident_tasks_easy, f"{save_path}/{args.dataset}_present_ident_tasks.json", indent=2)
                nncore.dump(all_present_ident_tasks_hard, f"{save_path}/{args.dataset}_present_ident_tasks_hard.json", indent=2)
                nncore.dump(all_present_attr_tasks, f"{save_path}/{args.dataset}_present_attr_tasks.json", indent=2)
            if generate_past:
                nncore.dump(all_past_next_after_group_tasks, f"{save_path}/{args.dataset}_past_next_after_group_tasks.json", indent=2)
                nncore.dump(all_past_never_gazed_tasks, f"{save_path}/{args.dataset}_past_never_gazed_tasks.json", indent=2)
                nncore.dump(all_past_transition_pattern_tasks, f"{save_path}/{args.dataset}_past_transition_pattern_tasks.json", indent=2)
                nncore.dump(all_past_scene_reconstruction_tasks, f"{save_path}/{args.dataset}_past_scene_reconstruction_tasks.json", indent=2)
            if generate_future:
                nncore.dump(all_future_action_tasks, f"{save_path}/{args.dataset}_future_action_tasks.json", indent=2)
                nncore.dump(all_future_remind_easy_tasks, f"{save_path}/{args.dataset}_future_remind_easy_tasks.json", indent=2)
                nncore.dump(all_future_remind_hard_tasks, f"{save_path}/{args.dataset}_future_remind_hard_tasks.json", indent=2)
            print(f"✓ Intermediate JSON files saved")
            
            if generate_future:
                print(f"  → Accumulated future action tasks: {len(all_future_action_tasks)}")
                print(f"  → Accumulated future remind easy tasks: {len(all_future_remind_easy_tasks)}")
                print(f"  → Accumulated future remind hard tasks: {len(all_future_remind_hard_tasks)}")

    # Save aggregated tasks after processing all videos
    print(f"\n{'='*60}")
    print("Saving aggregated results from all videos...")
    print(f"{'='*60}")
    
    print(f"\nTotal task counts across all videos:")
    if generate_present:
        print(f"  Present identification tasks (easy): {len(all_present_ident_tasks_easy)}")
        print(f"  Present identification tasks (hard): {len(all_present_ident_tasks_hard)}")
        print(f"  Present attribute tasks: {len(all_present_attr_tasks)}")
    if generate_past:
        print(f"  Past next after group tasks: {len(all_past_next_after_group_tasks)}")
        print(f"  Past never gazed tasks: {len(all_past_never_gazed_tasks)}")
        print(f"  Past transition pattern tasks: {len(all_past_transition_pattern_tasks)}")
        print(f"  Past scene reconstruction tasks: {len(all_past_scene_reconstruction_tasks)}")
    if generate_future:
        print(f"  Future action tasks: {len(all_future_action_tasks)}")
        print(f"  Future remind easy tasks: {len(all_future_remind_easy_tasks)}")
        print(f"  Future remind hard tasks: {len(all_future_remind_hard_tasks)}")
    
    # Save Present tasks (conditional)
    if generate_present:
        nncore.dump(all_present_ident_tasks_easy, f"{save_path}/{args.dataset}_present_ident_tasks.json", indent=2)
        nncore.dump(all_present_ident_tasks_hard, f"{save_path}/{args.dataset}_present_ident_tasks_hard.json", indent=2)
        nncore.dump(all_present_attr_tasks, f"{save_path}/{args.dataset}_present_attr_tasks.json", indent=2)
    
    # Save Past tasks (conditional)
    if generate_past:
        nncore.dump(all_past_next_after_group_tasks, f"{save_path}/{args.dataset}_past_next_after_group_tasks.json", indent=2)
        nncore.dump(all_past_never_gazed_tasks, f"{save_path}/{args.dataset}_past_never_gazed_tasks.json", indent=2)
        nncore.dump(all_past_transition_pattern_tasks, f"{save_path}/{args.dataset}_past_transition_pattern_tasks.json", indent=2)
        nncore.dump(all_past_scene_reconstruction_tasks, f"{save_path}/{args.dataset}_past_scene_reconstruction_tasks.json", indent=2)
    
    # Save Future tasks (conditional)
    if generate_future:
        print(f"\n📝 Saving Future tasks:")
        print(f"  → Action tasks: {len(all_future_action_tasks)} tasks → {save_path}/{args.dataset}_future_action_tasks.json")
        nncore.dump(all_future_action_tasks, f"{save_path}/{args.dataset}_future_action_tasks.json", indent=2)
        
        print(f"  → Remind easy tasks: {len(all_future_remind_easy_tasks)} tasks → {save_path}/{args.dataset}_future_remind_easy_tasks.json")
        nncore.dump(all_future_remind_easy_tasks, f"{save_path}/{args.dataset}_future_remind_easy_tasks.json", indent=2)
        
        print(f"  → Remind hard tasks: {len(all_future_remind_hard_tasks)} tasks → {save_path}/{args.dataset}_future_remind_hard_tasks.json")
        nncore.dump(all_future_remind_hard_tasks, f"{save_path}/{args.dataset}_future_remind_hard_tasks.json", indent=2)
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "start_pct": args.start_pct,
        "total_videos_processed": len(tasks),
        "task_types_generated": args.tasks,
        "task_counts": {}
    }
    
    # Add task counts conditionally
    if generate_present:
        summary["task_counts"]["present_ident_easy"] = len(all_present_ident_tasks_easy)
        summary["task_counts"]["present_ident_hard"] = len(all_present_ident_tasks_hard)
        summary["task_counts"]["present_attr"] = len(all_present_attr_tasks)
    if generate_past:
        summary["task_counts"]["past_next_after_group"] = len(all_past_next_after_group_tasks)
        summary["task_counts"]["past_never_gazed"] = len(all_past_never_gazed_tasks)
        summary["task_counts"]["past_transition_pattern"] = len(all_past_transition_pattern_tasks)
        summary["task_counts"]["past_scene_reconstruction"] = len(all_past_scene_reconstruction_tasks)
    if generate_future:
        summary["task_counts"]["future_action"] = len(all_future_action_tasks)
        summary["task_counts"]["future_remind_easy"] = len(all_future_remind_easy_tasks)
        summary["task_counts"]["future_remind_hard"] = len(all_future_remind_hard_tasks)
    
    nncore.dump(summary, f"{save_path}/task_summary.json", indent=2)
    
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✓ Checkpoint file removed: {checkpoint_file}")
    
    print(f"\n{'='*60}")
    print(f"🎉 All videos processed successfully!")
    print(f"Results saved in: {save_path}")
    print(f"{'='*60}")
    print(f"\nSaved files:")
    if generate_present:
        print(f"  - {args.dataset}_present_ident_tasks.json")
        print(f"  - {args.dataset}_present_ident_tasks_hard.json")
        print(f"  - {args.dataset}_present_attr_tasks.json")
    if generate_past:
        print(f"  - {args.dataset}_past_next_after_group_tasks.json")
        print(f"  - {args.dataset}_past_never_gazed_tasks.json")
        print(f"  - {args.dataset}_past_transition_pattern_tasks.json")
        print(f"  - {args.dataset}_past_scene_reconstruction_tasks.json")
    if generate_future:
        print(f"  - {args.dataset}_future_action_tasks.json")
        print(f"  - {args.dataset}_future_remind_easy_tasks.json")
        print(f"  - {args.dataset}_future_remind_hard_tasks.json")
    print("  - task_summary.json")
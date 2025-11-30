#!/usr/bin/env python3
"""
Run All QA Filtering Pipeline

This script applies all filtering functions to QA JSON files.

Input/Output for each filter function:
- Input: data (list of dicts), log_file (optional file handle)
- Output: (filtered_data, stats)
  - filtered_data: list of filtered QA items
  - stats: dict with filtering statistics

Usage:
    python run_all_filtering.py --input_dir <qa_json_dir> --output_dir <output_dir>
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Import all filtering functions
from filtering import (
    filter_future_action,
    filter_future_remind_hard,
    filter_future_remind_easy,
    filter_present_attr,
    filter_present_ident,
    filter_past_next_after_group,
    filter_past_scene_reconstruction,
    filter_past_transition_pattern,
)


# Task type to filter function mapping
TASK_FILTERS = {
    'future_action': filter_future_action,
    'future_remind_hard': filter_future_remind_hard,
    'future_remind_easy': filter_future_remind_easy,
    'present_attr': filter_present_attr,
    'present_ident': filter_present_ident,
    'past_next_after_group': filter_past_next_after_group,
    'past_scene_reconstruction': filter_past_scene_reconstruction,
    'past_transition_pattern': filter_past_transition_pattern,
}


def detect_task_type(filename):
    """Detect task type from filename"""
    filename_lower = filename.lower()
    
    # Future tasks
    if 'future_action' in filename_lower or 'action_prediction' in filename_lower:
        return 'future_action'
    elif 'remind_hard' in filename_lower:
        return 'future_remind_hard'
    elif 'remind_easy' in filename_lower or 'gaze_triggered' in filename_lower:
        return 'future_remind_easy'
    
    # Present tasks
    elif 'present_attr' in filename_lower or 'attribute_recognition' in filename_lower:
        return 'present_attr'
    elif 'present_ident' in filename_lower or 'identification' in filename_lower:
        return 'present_ident'
    
    # Past tasks
    elif 'next_after_group' in filename_lower or 'gaze_sequence' in filename_lower:
        return 'past_next_after_group'
    elif 'scene_reconstruction' in filename_lower or 'scene_recall' in filename_lower:
        return 'past_scene_reconstruction'
    elif 'transition_pattern' in filename_lower or 'object_transition' in filename_lower:
        return 'past_transition_pattern'
    
    return None


def load_json(json_path):
    """Load JSON data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, output_path):
    """Save JSON data"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_filtering(input_file, output_file, log_file, task_type=None):
    """Run filtering on a single file"""
    print(f"\n{'='*80}")
    print(f"Processing: {input_file}")
    print(f"{'='*80}")
    
    # Load data
    data = load_json(input_file)
    print(f"  📥 Loaded {len(data)} items")
    
    # Detect task type if not provided
    if task_type is None:
        task_type = detect_task_type(os.path.basename(input_file))
    
    if task_type is None:
        print(f"  ⚠️  Could not detect task type. Skipping.")
        return None
    
    print(f"  🔍 Task type: {task_type}")
    
    # Get filter function
    filter_func = TASK_FILTERS.get(task_type)
    if filter_func is None:
        print(f"  ⚠️  No filter function for task type: {task_type}")
        return None
    
    # Open log file
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"Filtering Log: {input_file}\n")
        log.write(f"Task Type: {task_type}\n")
        log.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"{'='*80}\n\n")
        
        # Run filtering
        print(f"  🚀 Running {filter_func.__name__}...")
        filtered_data, stats = filter_func(data, log)
        
        # Write stats to log
        log.write(f"\n{'='*80}\n")
        log.write(f"Filtering Statistics:\n")
        log.write(f"{'='*80}\n")
        for key, value in stats.items():
            log.write(f"{key}: {value}\n")
    
    # Print stats
    print(f"\n  📊 Statistics:")
    for key, value in stats.items():
        print(f"    - {key}: {value}")
    
    # Save filtered data
    save_json(filtered_data, output_file)
    print(f"  💾 Saved {len(filtered_data)} filtered items to: {output_file}")
    print(f"  📝 Log saved to: {log_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Run QA filtering pipeline')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input QA JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save filtered QA JSON files')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory to save log files (default: output_dir/logs)')
    parser.add_argument('--file_pattern', type=str, default='*.json',
                        help='File pattern to match (default: *.json)')
    
    args = parser.parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / 'logs'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"QA Filtering Pipeline")
    print(f"{'='*80}")
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Log Directory:    {log_dir}")
    print(f"{'='*80}\n")
    
    # Find all JSON files
    json_files = sorted(input_dir.glob(args.file_pattern))
    
    if not json_files:
        print(f"❌ No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files\n")
    
    # Process each file
    all_stats = {}
    success_count = 0
    skip_count = 0
    
    for json_file in json_files:
        filename = json_file.name
        output_file = output_dir / filename
        log_file = log_dir / f"{json_file.stem}.log"
        
        try:
            stats = run_filtering(json_file, output_file, log_file)
            if stats is not None:
                all_stats[filename] = stats
                success_count += 1
            else:
                skip_count += 1
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            skip_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"Total files:      {len(json_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped:          {skip_count}")
    print(f"{'='*80}\n")
    
    # Save summary
    summary_file = output_dir / 'filtering_summary.json'
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'total_files': len(json_files),
        'success_count': success_count,
        'skip_count': skip_count,
        'per_file_stats': all_stats
    }
    save_json(summary, summary_file)
    print(f"📊 Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()


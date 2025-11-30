"""
Data analysis utility functions
"""

import os
import pandas as pd
from collections import Counter


def save_frequency_analysis(other_object_counter, output_path, video_name):
    """Save frequency analysis results to a text file"""
    with open(output_path, 'w') as f:
        f.write(f"Frequency Analysis Results for {video_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"🔍 Total unique other objects found: {len(other_object_counter)}\n")
        f.write(f"📊 Total other object instances: {sum(other_object_counter.values())}\n\n")
        
        f.write("📦 TOP 20 OTHER OBJECTS BY FREQUENCY:\n")
        for i, (obj_name, count) in enumerate(other_object_counter.most_common(20), 1):
            f.write(f"  {i:2d}. {obj_name}: {count} times\n")
        f.write("\n")
        
        f.write("📦 ALL OTHER OBJECTS FREQUENCY:\n")
        for obj_name, count in sorted(other_object_counter.items()):
            f.write(f"  {obj_name}: {count}\n")
        f.write("\n")
        
        single_occurrence = [obj for obj, count in other_object_counter.items() if count == 1]
        f.write(f"🔸 Objects appearing only once ({len(single_occurrence)} objects):\n")
        for obj in sorted(single_occurrence):
            f.write(f"  - {obj}\n")


def count_total_fixations(base_dir):
    """Count total number of fixations across all videos"""
    total = 0
    video_fixations = {}
    
    for video_name in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, video_name)):
            continue
            
        try:
            fixation_path = os.path.join(base_dir, video_name, f'{video_name}_fixation_dataset.csv')
            if os.path.exists(fixation_path):
                df = pd.read_csv(fixation_path)
                count = len(df)
                total += count
                video_fixations[video_name] = count
        except Exception as e:
            print(f"Error counting fixations for {video_name}: {e}")
            video_fixations[video_name] = 0
            
    return total, video_fixations

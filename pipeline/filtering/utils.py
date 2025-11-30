"""
Common utility functions for filtering
"""

import re
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

# Qwen3VL Model Configuration
print("Loading Qwen3VL-30B model...")
qwen_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct", 
    dtype="auto", 
    device_map="auto"
)
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
print("Qwen3VL-30B model loaded successfully!")

# Human-related keywords
HUMAN_KEYWORDS = ['hand', 'hands', 'finger', 'fingers', 'arm', 'arms', 
                  'foot', 'feet', 'leg', 'legs', 'body', 'face', 'head', 'person']


def parse_time_to_seconds(time_str):
    """Convert MM:SS or HH:MM:SS to seconds"""
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def seconds_to_time_str(seconds):
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def is_human_related(obj_name):
    """Check if object is human-related"""
    obj_lower = str(obj_name).lower()
    return any(keyword in obj_lower for keyword in HUMAN_KEYWORDS)


def get_qwen_model():
    """Get Qwen3VL model"""
    return qwen_model

def get_qwen_processor():
    """Get Qwen3VL processor"""
    return qwen_processor


def extract_object_name(option_str):
    """Extract object name from option string like 'A. spoon' -> 'spoon'"""
    if '. ' in option_str:
        return option_str.split('. ', 1)[1]
    return option_str


def normalize_object_name(obj_name):
    """Normalize object name: lowercase + underscore to space"""
    obj_name = obj_name.lower()
    obj_name = obj_name.replace('_', ' ')
    return obj_name


def get_gaze_visualization_path(original_video_path):
    """Convert original video path to gaze_visualization path"""
    return original_video_path.replace('/videos/', '/gaze_visualization/')


def time_to_seconds(time_str):
    """Convert time string (MM:SS) to seconds - alias for parse_time_to_seconds"""
    return parse_time_to_seconds(time_str)


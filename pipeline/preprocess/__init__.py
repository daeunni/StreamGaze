# Preprocess module for EGTEA gaze processing
"""
This module contains functions for processing EGTEA gaze data and interacting with Gemini API.
"""

# Data analysis functions
from .data_analysis import count_total_fixations, save_frequency_analysis

# Gaze processing functions  
from .gaze_processing import (
    parse_gtea_gaze,
    parse_ego4d_gaze,
    parse_egoexo_gaze,
    parse_holoassist_gaze,
    extract_fixation_segments,
    extract_confusion_segments,
    detect_saccade_segments_ivt_dispersion
)

# Gemini API functions
# from .gemini_api import (
#     load_gemini_api_key,
#     setup_gemini_client,
#     call_gemini_api,
#     extract_objects_and_scene_from_video_clip_gemini,
#     extract_objects_and_scene_from_video_clip_gemini_threaded,
#     video_to_base64,
#     pil_to_base64
# )

# Video processing functions
from .video_processing import (
    process_single_video,
    process_single_video_ego4d,
    process_single_video_egoexo,
    process_single_video_holoassist,
    main as process_all_videos
)

# Visualization functions
from .visualization import (
    plot_gaze_segments,
    visualize_gaze_with_trail,
    visualize_gaze_green_dot,
    visualize_gaze_green_dot_red_fov,
    extract_and_save_gifs,
    seconds_to_timestamp
)

# All exports
__all__ = [
    # Data analysis
    'count_total_fixations',
    'save_frequency_analysis',
    
    # Gaze processing
    'parse_gtea_gaze',
    'parse_ego4d_gaze',
    'parse_egoexo_gaze',
    'parse_holoassist_gaze',
    'extract_fixation_segments', 
    'extract_confusion_segments',
    'detect_saccade_segments_ivt_dispersion',
    
    # Gemini API
    # 'load_gemini_api_key',
    # 'setup_gemini_client', 
    # 'call_gemini_api',
    # 'extract_objects_and_scene_from_video_clip_gemini',
    # 'extract_objects_and_scene_from_video_clip_gemini_threaded',
    # 'video_to_base64',
    # 'pil_to_base64',
    
    # Video processing
    'process_single_video',
    'process_single_video_ego4d',
    'process_single_video_egoexo',
    'process_single_video_holoassist',
    'process_all_videos',
    
    # Visualization
    'plot_gaze_segments',
    'visualize_gaze_with_trail',
    'visualize_gaze_green_dot',
    'visualize_gaze_green_dot_red_fov',
    'extract_and_save_gifs',
    'seconds_to_timestamp'
]

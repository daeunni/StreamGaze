"""
Action mapping functions for EGTEA gaze processing
"""
import pandas as pd
import numpy as np
import json


def map_actions_to_gaze(df, action_data, video_name):
    """Map action information to gaze data"""
    # Simple implementation - just return the original df for now
    return df


def find_overlapping_action(start_time, end_time, actions_list):
    """
    Find all actions that overlap with the given time segment
    
    Args:
        start_time: Segment start time in seconds
        end_time: Segment end time in seconds
        actions_list: List of action dictionaries (for EgoExoLearn/HoloAssist JSON format)
                     or DataFrame (for EGTEA CSV format)
    
    Returns:
        List of action texts (empty list if no overlap)
    """
    if actions_list is None or len(actions_list) == 0:
        return []
    
    overlapping_actions = []
    
    # Handle JSON format (list of dicts) - EgoExoLearn or HoloAssist
    if isinstance(actions_list, list):
        for action in actions_list:
            action_start = action['start']
            action_end = action['end']
            
            # Check if there's overlap between segment and action
            if not (end_time < action_start or start_time > action_end):
                # There's overlap - extract the action text
                
                # EgoExoLearn format: uses 'textAttribute_en' key
                if 'textAttribute_en' in action:
                    action_text = action.get('textAttribute_en', '')
                    if action_text:
                        overlapping_actions.append(action_text)
                
                # HoloAssist format: uses attributes with Verb + Noun
                elif 'attributes' in action:
                    attrs = action['attributes']
                    verb = attrs.get('Verb', '')
                    noun = attrs.get('Noun', '')
                    
                    # Build action text from verb + noun
                    if verb and verb != 'none' and noun and noun != 'none':
                        action_text = f"{verb} {noun}"
                        overlapping_actions.append(action_text)
                    elif verb and verb != 'none':
                        overlapping_actions.append(verb)
                    elif noun and noun != 'none':
                        overlapping_actions.append(noun)
    
    # Handle EGTEA DataFrame format (not implemented yet)
    elif isinstance(actions_list, pd.DataFrame):
        # TODO: Implement EGTEA action mapping if needed
        pass
    
    return overlapping_actions


def create_segment_dataset_with_actions(segments, df_with_actions, action_data, video_name):
    """
    Create segment-based dataset and map action information
    
    Args:
        segments: List of time-based segments (fixations, confusion segments, etc.)
        df_with_actions: Frame-wise DataFrame with action mapping
        action_data: Original action data (DataFrame for EGTEA, list of dicts for EgoExoLearn)
        video_name: Video name (e.g., 'OP01-R01-PastaSalad')
    
    Returns:
        segment_dataset: Segment-based DataFrame with action mapping
    """
    if not segments:
        print("❌ No segment data available.")
        return pd.DataFrame()
    
    print(f"Processing {len(segments)} segments...")
    
    segment_data = []
    
    # Get actions for current video (EGTEA style)
    if isinstance(action_data, pd.DataFrame):
        current_video_actions = action_data[action_data[' Video Session'] == f' {video_name}'].copy() if action_data is not None else pd.DataFrame()
    else:
        # For EgoExoLearn, action_data is already the list for this video
        current_video_actions = action_data
    
    for idx, segment in enumerate(segments):
        # Basic segment information
        start_time = segment['start_time']
        end_time = segment['end_time']
        center_x = segment['center_x']
        center_y = segment['center_y']
        duration = end_time - start_time
        
        # Find all overlapping actions
        overlapping_actions = find_overlapping_action(start_time, end_time, current_video_actions)
        
        # Format action caption based on data type
        if isinstance(current_video_actions, list):
            # EgoExoLearn: Store as JSON list for multiple actions
            action_caption = json.dumps(overlapping_actions, ensure_ascii=False)
        elif isinstance(current_video_actions, pd.DataFrame):
            # EGTEA: Store as string (legacy format)
            action_caption = overlapping_actions[0] if overlapping_actions else 'unknown'
        else:
            # No action data
            action_caption = 'unknown'
        
        # Create segment entry with _seconds suffix for consistency
        segment_entry = {
            'segment_id': f'segment_{idx}',
            'start_time_seconds': start_time,
            'end_time_seconds': end_time,
            'duration_seconds': duration,
            'center_x': center_x,
            'center_y': center_y,
            'action_caption': action_caption  # Keep same column name as EGTEA
        }
        
        segment_data.append(segment_entry)
    
    return pd.DataFrame(segment_data)


def seconds_to_timestamp(seconds):
    """Convert seconds to timestamp format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

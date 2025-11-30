"""
Past Task Filtering (Next After Group, Scene Reconstruction, Transition Pattern)
"""

from tqdm import tqdm
from .utils import is_human_related, parse_time_to_seconds


def filter_past_next_after_group(data, log_file=None):
    """Filter past next after group tasks"""
    stats = {
        'initial': len(data),
        'filtered_human_objects': 0,
        'final': 0
    }
    
    filtered_data = []
    
    for item in tqdm(data, desc="Filtering past_next_after_group"):
        q = item['questions'][0]
        
        # Check for human objects in options
        has_human = False
        for opt in q.get('options', []):
            # Extract object names from options like "A. spoon"
            obj = opt.split('. ', 1)[1] if '. ' in opt else opt
            if is_human_related(obj):
                has_human = True
                if log_file:
                    log_file.write(f"[FILTERED - HUMAN OBJECT] {obj}\n")
                break
        
        if has_human:
            stats['filtered_human_objects'] += 1
            continue
        
        filtered_data.append(item)
    
    stats['final'] = len(filtered_data)
    return filtered_data, stats


def filter_past_scene_reconstruction(data, log_file=None):
    """Filter past scene reconstruction tasks"""
    stats = {
        'initial': len(data),
        'filtered_short_clip': 0,
        'filtered_human_objects': 0,
        'filtered_gazed_in_options': 0,
        'final': 0
    }
    
    filtered_data = []
    
    for item in tqdm(data, desc="Filtering past_scene_reconstruction"):
        q = item['questions'][0]
        
        # Filter 1: Check clip duration (skip very short clips < 5s)
        clip = q.get('input_video_clip', [0, 100])
        duration = clip[1] - clip[0]
        
        if duration < 5:
            stats['filtered_short_clip'] += 1
            if log_file:
                log_file.write(f"[FILTERED - SHORT CLIP] duration={duration}s\n")
            continue
        
        # Filter 2: Check for human objects in gazed_objects
        gazed_objects = q.get('gazed_objects', [])
        has_human = any(is_human_related(obj) for obj in gazed_objects)
        
        if has_human:
            stats['filtered_human_objects'] += 1
            if log_file:
                human_objs = [obj for obj in gazed_objects if is_human_related(obj)]
                log_file.write(f"[FILTERED - HUMAN OBJECTS] {human_objs}\n")
            continue
        
        # Filter 3: Check if gazed_objects appear in options (should not)
        gazed_set = set(obj.lower() for obj in gazed_objects)
        
        has_overlap = False
        for opt in q.get('options', []):
            # Extract object from option like "A. spoon"
            obj = opt.split('. ', 1)[1] if '. ' in opt else opt
            if obj.lower() in gazed_set:
                has_overlap = True
                if log_file:
                    log_file.write(f"[FILTERED - GAZED IN OPTIONS] {obj}\n")
                break
        
        if has_overlap:
            stats['filtered_gazed_in_options'] += 1
            continue
        
        filtered_data.append(item)
    
    stats['final'] = len(filtered_data)
    return filtered_data, stats


def has_consecutive_identical_groups(sequence):
    """Check if sequence has consecutive identical groups"""
    for i in range(len(sequence) - 1):
        if set(sequence[i]) == set(sequence[i+1]):
            return True
    return False


def filter_past_transition_pattern(data, log_file=None):
    """Filter past transition pattern tasks"""
    stats = {
        'initial': len(data),
        'filtered_consecutive_identical': 0,
        'filtered_short_sequence': 0,
        'final': 0
    }
    
    filtered_data = []
    
    for item in tqdm(data, desc="Filtering past_transition_pattern"):
        q = item['questions'][0]
        
        correct_sequence = q.get('correct_sequence', [])
        
        # Filter 1: Check for consecutive identical groups
        if has_consecutive_identical_groups(correct_sequence):
            stats['filtered_consecutive_identical'] += 1
            if log_file:
                log_file.write(f"[FILTERED - CONSECUTIVE IDENTICAL] {correct_sequence}\n")
            continue
        
        # Filter 2: Check sequence length (should have at least 2 groups)
        if len(correct_sequence) < 2:
            stats['filtered_short_sequence'] += 1
            if log_file:
                log_file.write(f"[FILTERED - SHORT SEQUENCE] {correct_sequence}\n")
            continue
        
        filtered_data.append(item)
    
    stats['final'] = len(filtered_data)
    return filtered_data, stats

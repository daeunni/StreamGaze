"""
Present Object Identification Task Filtering

Complete implementation with similar object fixing
"""

from tqdm import tqdm
from .utils import parse_time_to_seconds, is_human_related

# Similar object pairs (keep first, remove second if both appear)
SIMILAR_PAIRS = [
    ('counter', 'countertop'),
    ('tomato', 'tomatoes'),
    ('onion', 'onions'),
    ('pepper', 'peppers'),
    ('knife', 'knives'),
    ('plate', 'plates'),
    ('bowl', 'bowls'),
    ('spoon', 'spoons'),
    ('fork', 'forks'),
    ('cup', 'cups'),
]


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


def has_similar_objects(options):
    """Check if options contain similar objects"""
    objects = [extract_object_name(opt).lower() for opt in options]
    similar_found = []
    
    for pair in SIMILAR_PAIRS:
        indices = []
        for i, obj in enumerate(objects):
            if obj == pair[0] or obj == pair[1]:
                indices.append((i, obj))
        
        # If both members of the pair are found
        if len(indices) >= 2:
            pair_objs = [obj for _, obj in indices]
            if pair[0] in pair_objs and pair[1] in pair_objs:
                similar_found.append((pair, indices))
    
    return similar_found


def fix_similar_options(question_data, similar_pairs):
    """Fix similar objects in options by removing the second member of each pair"""
    if not similar_pairs:
        return question_data['options'], question_data['answer'], False
    
    options = question_data['options'].copy()
    answer = question_data['answer']
    answer_idx = ord(answer) - ord('A')
    
    # Collect all indices to remove
    indices_to_remove = set()
    
    for pair, indices in similar_pairs:
        # Remove second member (usually plural/longer)
        to_remove_obj = pair[1]
        
        for idx, obj in indices:
            if obj == to_remove_obj:
                indices_to_remove.add(idx)
    
    if not indices_to_remove:
        return options, answer, False
    
    # Build new options list without removed indices
    new_options = []
    old_to_new_idx = {}
    
    for i, opt in enumerate(options):
        if i not in indices_to_remove:
            new_idx = len(new_options)
            old_to_new_idx[i] = new_idx
            letter = chr(65 + new_idx)
            obj_name = extract_object_name(opt)
            new_options.append(f"{letter}. {obj_name}")
    
    # Need 4 options, skip if we can't maintain that
    if len(new_options) < 4:
        return None, None, True
    
    # Update answer letter
    new_answer_idx = old_to_new_idx.get(answer_idx)
    if new_answer_idx is None:
        return None, None, True
    
    new_answer = chr(65 + new_answer_idx)
    
    return new_options, new_answer, True


def filter_present_ident(data, log_file=None):
    """Filter present identification tasks"""
    stats = {
        'initial': len(data),
        'filtered_zero_duration': 0,
        'filtered_human_objects': 0,
        'fixed_similar_objects': 0,
        'normalized_names': 0,
        'final': 0
    }
    
    filtered_data = []
    
    for item in tqdm(data, desc="Filtering present_ident"):
        question_data = item['questions'][0]
        
        # Extract video clip info
        clip = question_data.get('input_video_clip', [0, 10])
        duration = clip[1] - clip[0]
        
        # Filter 1: Remove zero duration clips
        if duration == 0:
            stats['filtered_zero_duration'] += 1
            if log_file:
                log_file.write(f"[FILTERED - ZERO DURATION]\n")
            continue
        
        # Filter 2: Remove human-related objects
        obj_name = question_data.get('object', '')
        if is_human_related(obj_name):
            stats['filtered_human_objects'] += 1
            if log_file:
                log_file.write(f"[FILTERED - HUMAN OBJECT] {obj_name}\n")
            continue
        
        # Normalize object name
        original_obj = question_data['object']
        normalized_obj = normalize_object_name(original_obj)
        if original_obj != normalized_obj:
            question_data['object'] = normalized_obj
            stats['normalized_names'] += 1
        
        # Normalize options
        normalized_options = []
        for opt in question_data['options']:
            letter = opt.split('. ')[0] if '. ' in opt else ''
            obj = extract_object_name(opt)
            normalized = normalize_object_name(obj)
            if letter:
                normalized_options.append(f"{letter}. {normalized}")
            else:
                normalized_options.append(normalized)
        question_data['options'] = normalized_options
        
        # Filter 3: Fix similar objects in options
        similar_pairs = has_similar_objects(question_data['options'])
        if similar_pairs:
            fixed_options, fixed_answer, was_modified = fix_similar_options(question_data, similar_pairs)
            
            if fixed_options is None:
                # Couldn't fix properly, skip
                if log_file:
                    log_file.write(f"[FILTERED - COULD NOT FIX SIMILAR] {question_data['object']}\n")
                continue
            
            if was_modified:
                question_data['options'] = fixed_options
                question_data['answer'] = fixed_answer
                stats['fixed_similar_objects'] += 1
                
                if log_file:
                    log_file.write(f"[FIXED - SIMILAR OBJECTS] {question_data['object']}\n")
        
        filtered_data.append(item)
    
    stats['final'] = len(filtered_data)
    return filtered_data, stats

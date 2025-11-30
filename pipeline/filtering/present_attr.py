"""
Present Object Attribute Task Filtering

Complete implementation with Qwen3VL verification
"""

import os
import base64
import cv2
from tqdm import tqdm
from .utils import parse_time_to_seconds, is_human_related, get_qwen_model, get_qwen_processor

# Ambiguous attribute types (too subjective - remove entirely)
AMBIGUOUS_ATTRIBUTE_TYPES = ['usage', 'position', 'size']

# Confusing attribute pairs
CONFUSING_COLOR_PAIRS = [
    ('red', 'pink'), ('red', 'dark red'), ('blue', 'light blue'), ('blue', 'dark blue'),
    ('white', 'silver'), ('white', 'light gray'), ('black', 'dark gray'), ('black', 'dark'),
    ('brown', 'tan'), ('brown', 'dark brown'), ('green', 'dark green'), ('green', 'lime'),
]

CONFUSING_MATERIAL_PAIRS = [
    ('metal', 'metallic'), ('wood', 'wooden'), ('plastic', 'synthetic'), ('glass', 'transparent'),
]

CONFUSING_TEXTURE_PAIRS = [
    ('smooth', 'glossy'), ('smooth', 'polished'), ('rough', 'coarse'), ('rough', 'textured'),
]


def extract_frames_from_video(video_path, start_sec, end_sec, num_frames=8):
    """Extract evenly spaced frames from video segment"""
    if not os.path.exists(video_path):
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        duration_frames = end_frame - start_frame
        if duration_frames <= 0:
            cap.release()
            return None
        
        frame_indices = [int(start_frame + (i * duration_frames / (num_frames - 1))) 
                        for i in range(num_frames)]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (512, 384))
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames.append(frame_base64)
        
        cap.release()
        return frames
        
    except Exception as e:
        print(f"   ✗ Error extracting frames: {e}")
        return None


def normalize_object_name(obj_name):
    """Normalize object name: lowercase + underscore to space"""
    obj_name = obj_name.lower()
    obj_name = obj_name.replace('_', ' ')
    return obj_name


def extract_option_value(option_str):
    """Extract value from option string like 'A. red' -> 'red'"""
    if '. ' in option_str:
        return option_str.split('. ', 1)[1]
    return option_str


def has_confusing_pairs(options, attribute_type):
    """Check if options contain confusing pairs"""
    option_values = [extract_option_value(opt).lower() for opt in options]
    
    if attribute_type == 'color':
        pairs = CONFUSING_COLOR_PAIRS
    elif attribute_type == 'material':
        pairs = CONFUSING_MATERIAL_PAIRS
    elif attribute_type == 'texture':
        pairs = CONFUSING_TEXTURE_PAIRS
    else:
        return []
    
    confusing = []
    for pair in pairs:
        if pair[0] in option_values and pair[1] in option_values:
            confusing.append(pair)
    
    return confusing


def get_replacement_option_qwen3vl(frames, question_data, confusing_pair):
    """Use Qwen3VL to suggest a replacement option"""
    model = get_qwen_model()
    processor = get_qwen_processor()
    
    object_name = question_data.get('object_name', '')
    attribute_type = question_data.get('attribute_type', '')
    current_options = [extract_option_value(opt) for opt in question_data['options']]
    answer_letter = question_data['answer']
    answer_idx = ord(answer_letter) - ord('A')
    correct_answer = current_options[answer_idx]
    
    prompt = f"""You are helping maintain appropriate difficulty in multiple-choice questions for egocentric video understanding.

**Current Situation:**
- Object: {object_name}
- Attribute type: {attribute_type}
- Current options: {', '.join(current_options)}
- Correct answer: {correct_answer}
- Problem: "{confusing_pair[0]}" and "{confusing_pair[1]}" are too similar (unfair)

**Your Task:**
Replace the confusing option ("{confusing_pair[1]}") with a better alternative that:

1. **Maintains moderate difficulty** (most important!)
   - Should be similar enough to be plausible
   - Should be distinguishable if watched carefully
   - Avoid extreme contrasts that make it too easy

2. **Good examples:**
   - For red tomato: orange, brown, burgundy (similar hue, but distinguishable) ✅
   - For red tomato: blue, purple, white (too different, too easy) ❌
   
3. **Context-appropriate:**
   - For color: use similar color family
   - For material: use related materials
   - For texture: use related textures

4. **Not already in options:** {', '.join(current_options)}

**Watch the video frames to see the actual object and suggest ONE appropriate replacement.**

Reply with ONLY the replacement word, nothing else. No explanations, no quotes.
"""
    
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image", "image": f"data:image/jpeg;base64,{frame}"})
    
    messages = [{"role": "user", "content": content}]
    
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        replacement = output_text[0].strip()
        replacement = replacement.strip('"').strip("'").strip().lower()
        
        return replacement
        
    except Exception as e:
        print(f"   ✗ Qwen3VL API Error: {e}")
        return None


def filter_present_attr(data, log_file=None):
    """Filter present attribute tasks with Qwen3VL"""
    stats = {
        'initial': len(data),
        'filtered_zero_duration': 0,
        'filtered_human_objects': 0,
        'filtered_ambiguous_type': 0,
        'fixed_confusing_options': 0,
        'normalized_names': 0,
        'final': 0
    }
    
    filtered_data = []
    
    for item in tqdm(data, desc="Filtering present_attr"):
        question_data = item['questions'][0]
        video_path = item.get('video_path', item.get('video'))
        
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
        obj_name = question_data.get('object_name', '')
        if is_human_related(obj_name):
            stats['filtered_human_objects'] += 1
            if log_file:
                log_file.write(f"[FILTERED - HUMAN OBJECT] {obj_name}\n")
            continue
        
        # Filter 3: Remove ambiguous attribute types
        attribute_type = question_data.get('attribute_type', '')
        if attribute_type in AMBIGUOUS_ATTRIBUTE_TYPES:
            stats['filtered_ambiguous_type'] += 1
            if log_file:
                log_file.write(f"[FILTERED - AMBIGUOUS TYPE] {attribute_type}\n")
            continue
        
        # Normalize object name
        original_obj = question_data['object_name']
        normalized_obj = normalize_object_name(original_obj)
        if original_obj != normalized_obj:
            question_data['object_name'] = normalized_obj
            stats['normalized_names'] += 1
        
        # Normalize options
        normalized_options = []
        for opt in question_data['options']:
            letter = opt.split('. ')[0] if '. ' in opt else ''
            value = extract_option_value(opt)
            normalized = normalize_object_name(value) if '_' in value or value[0].isupper() else value.lower()
            if letter:
                normalized_options.append(f"{letter}. {normalized}")
            else:
                normalized_options.append(normalized)
        question_data['options'] = normalized_options
        
        # Filter 4: Fix confusing pairs
        confusing_pairs = has_confusing_pairs(question_data['options'], attribute_type)
        
        if confusing_pairs:
            # Extract frames for Qwen3VL
            response_time = item['response_time']
            time_range = response_time.strip('[]').split(' - ')
            start_sec = parse_time_to_seconds(time_range[0])
            end_sec = parse_time_to_seconds(time_range[1])
            
            frames = extract_frames_from_video(video_path, start_sec, end_sec, 8)
            
            if frames is None:
                if log_file:
                    log_file.write(f"[ERROR] Failed to extract frames\n")
                continue
            
            # Fix each confusing pair
            for pair in confusing_pairs:
                replacement = get_replacement_option_qwen3vl(frames, question_data, pair)
                
                if replacement and replacement not in [extract_option_value(opt).lower() for opt in question_data['options']]:
                    option_values = [extract_option_value(opt).lower() for opt in question_data['options']]
                    
                    if pair[1] in option_values:
                        replace_idx = option_values.index(pair[1])
                        letter = chr(65 + replace_idx)
                        question_data['options'][replace_idx] = f"{letter}. {replacement}"
                        
                        stats['fixed_confusing_options'] += 1
                        
                        if log_file:
                            log_file.write(f"[FIXED] {pair[0]} vs {pair[1]} -> {replacement}\n")
        
        filtered_data.append(item)
    
    stats['final'] = len(filtered_data)
    return filtered_data, stats

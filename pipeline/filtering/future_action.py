"""
Future Action Prediction Task Filtering

Filters:
1. Time gap (3s < gap <= 60s)
2. Context-answer overlap
3. Generic context
4. Polish truncated options with Qwen3VL
"""

import os
import re
from tqdm import tqdm
from .utils import parse_time_to_seconds, seconds_to_time_str, get_qwen_model, get_qwen_processor


def is_generic_context(context_objects):
    """Check if context consists only of generic/furniture objects"""
    generic_keywords = [
        'counter', 'sink', 'cabinet', 'stove', 'table', 'floor', 'wall', 
        'ceiling', 'curtain', 'countertop', 'outlet', 'window', 'furniture',
        'shelf', 'drawer', 'rack', 'gas_stove'
    ]
    
    if not context_objects:
        return True
    
    all_generic = all(
        any(keyword in str(obj).lower() for keyword in generic_keywords)
        for obj in context_objects
    )
    
    return all_generic


def has_context_answer_overlap(context_objects, answer_action):
    """Check if context objects appear in answer action"""
    context_objects_lower = [str(obj).lower() for obj in context_objects]
    answer_lower = answer_action.lower()
    
    for obj in context_objects_lower:
        obj_clean = re.sub(r'[0-9\-_]', ' ', obj).strip()
        words = obj_clean.split()
        
        for word in words:
            if len(word) > 3 and word in answer_lower:
                return True
    
    return False


def needs_polishing(options):
    """Check if options need polishing"""
    for opt in options:
        if len(opt) > 100 and not opt.strip()[-1] in '.!?,;':
            return True
        if len(opt) > 200:
            return True
    return False


def polish_options_with_qwen3vl(options):
    """Polish options with Qwen3VL"""
    if not options:
        return options
    
    try:
        model = get_qwen_model()
        processor = get_qwen_processor()
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        
        prompt = f"""Polish these multiple-choice options for a future action prediction task.

Requirements:
1. Fix truncated text (e.g., "stir-f" → "stir-fry")
2. Keep options concise (max 2 sentences or ~100 characters)
3. Maintain key information about the action
4. Keep the same letter prefix (A., B., C., D.)
5. Use clear, natural English

Options:
{options_text}

Return ONLY the polished options in the same format, one per line."""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        polished_text = output_text[0].strip()
        polished_lines = [line.strip() for line in polished_text.split('\n') if line.strip()]
        
        if len(polished_lines) == len(options):
            return polished_lines
        else:
            return options
            
    except Exception as e:
        print(f"  ⚠️  Qwen3VL polishing failed: {e}")
        return options


def filter_future_action(data, log_file=None):
    """Filter future action tasks"""
    min_time_gap = 3
    max_time_gap = 60
    
    stats = {
        'initial': len(data),
        'filtered_time_gap_too_short': 0,
        'filtered_time_gap_too_long': 0,
        'filtered_generic_context': 0,
        'filtered_context_answer_overlap': 0,
        'polished_options': 0,
        'final': 0
    }
    
    filtered_data = []
    
    for item in tqdm(data, desc="Filtering future_action"):
        for q in item['questions']:
            video_name = os.path.basename(item['video'])
            
            timestamp_sec = parse_time_to_seconds(q['time_stamp'])
            action_time_sec = parse_time_to_seconds(q['actual_action_time'])
            time_gap = action_time_sec - timestamp_sec
            
            # Filter 1: Time gap <= 3s
            if time_gap <= min_time_gap:
                stats['filtered_time_gap_too_short'] += 1
                if log_file:
                    log_file.write(f"[FILTERED - TIME TOO SHORT] {video_name}: {time_gap}s\n")
                continue
            
            # Filter 2: Time gap > 60s
            if time_gap > max_time_gap:
                stats['filtered_time_gap_too_long'] += 1
                if log_file:
                    log_file.write(f"[FILTERED - TIME TOO LONG] {video_name}: {time_gap}s\n")
                continue
            
            # Filter 3: Generic context
            if is_generic_context(q['context_objects']):
                stats['filtered_generic_context'] += 1
                if log_file:
                    log_file.write(f"[FILTERED - GENERIC CONTEXT] {video_name}\n")
                continue
            
            # Filter 4: Context-answer overlap
            if has_context_answer_overlap(q['context_objects'], q['answer_action']):
                stats['filtered_context_answer_overlap'] += 1
                if log_file:
                    log_file.write(f"[FILTERED - CONTEXT-ANSWER OVERLAP] {video_name}\n")
                continue
            
            # Polish options if needed
            if needs_polishing(q['options']):
                original_options = q['options'].copy()
                q['options'] = polish_options_with_qwen3vl(q['options'])
                stats['polished_options'] += 1
                
                if log_file:
                    log_file.write(f"[POLISHED OPTIONS] {video_name}\n")
            
            # Update response_time
            response_start_sec = timestamp_sec
            response_end_sec = action_time_sec + 5
            item['response_time'] = f"[{seconds_to_time_str(response_start_sec)} - {seconds_to_time_str(response_end_sec)}]"
            
            filtered_data.append(item)
    
    stats['final'] = len(filtered_data)
    return filtered_data, stats


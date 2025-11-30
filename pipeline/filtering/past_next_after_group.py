"""
Past Next After Group Task Filtering

Complete implementation with Qwen3VL for ambiguous distractor replacement
"""

import os
import time
from tqdm import tqdm
from .utils import is_human_related, get_qwen_model, get_qwen_processor


def extract_object_name(option_str):
    """Extract object name from option string like 'A. spoon' -> 'spoon'"""
    if '. ' in option_str:
        return option_str.split('. ', 1)[1]
    return option_str


def check_similarity(obj1, obj2, strict_mode=True):
    """Check if two object names are similar/overlapping"""
    words1 = set(obj1.lower().replace('_', ' ').split())
    words2 = set(obj2.lower().replace('_', ' ').split())
    
    common_words = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'and', 'or'}
    modifier_words = {'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown',
                      'big', 'small', 'large', 'tiny', 'old', 'new', 'clean', 'dirty'}
    
    words1 = words1 - common_words
    words2 = words2 - common_words
    
    overlap = words1.intersection(words2)
    
    if not overlap:
        return False, [], None
    
    only_modifiers = all(word in modifier_words for word in overlap)
    
    if obj1 == obj2:
        return True, list(overlap), "exact_duplicate"
    elif obj1 in obj2 or obj2 in obj1:
        return True, list(overlap), "substring"
    elif len(overlap) > 0:
        if strict_mode and only_modifiers:
            return False, [], None
        
        total_words = len(words1.union(words2))
        overlap_ratio = len(overlap) / total_words if total_words > 0 else 0
        
        if strict_mode and overlap_ratio < 0.5:
            return False, [], None
        
        return True, list(overlap), "word_overlap"
    
    return False, [], None


def find_ambiguous_options(question_data, strict_mode=True):
    """Find ambiguous option pairs in a question"""
    options = question_data['options']
    objects = [extract_object_name(opt) for opt in options]
    
    ambiguous_pairs = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            is_similar, overlap, sim_type = check_similarity(objects[i], objects[j], strict_mode)
            if is_similar:
                ambiguous_pairs.append({
                    'idx1': i,
                    'idx2': j,
                    'obj1': objects[i],
                    'obj2': objects[j],
                    'overlap': overlap,
                    'similarity_type': sim_type
                })
    
    return ambiguous_pairs


def collect_all_objects(data):
    """Collect all unique objects from the dataset"""
    all_objects = set()
    for item in data:
        for q in item['questions']:
            options = q['options']
            objects = [extract_object_name(opt) for opt in options]
            all_objects.update(objects)
            
            if 'answer_object' in q:
                all_objects.add(q['answer_object'])
            
            if 'object_group' in q:
                all_objects.update(q['object_group'])
    
    return list(all_objects)


def get_replacement_object_qwen3vl(ambiguous_obj, answer_obj, object_group, all_objects_pool, used_objects):
    """Use Qwen3VL to suggest a replacement object"""
    model = get_qwen_model()
    processor = get_qwen_processor()
    
    prompt = f"""You are helping to create multiple-choice questions for egocentric video understanding.

Current situation:
- We need to replace an ambiguous distractor (wrong answer option): "{ambiguous_obj}"
- The correct answer is: "{answer_obj}"
- The question asks: "What object does the user gaze at next after looking at {{{', '.join(object_group)}}}?"
- Objects already used in this question: {', '.join(used_objects)}

Requirements for replacement:
1. Must be DISTINCT from "{ambiguous_obj}" and all other options
2. Must be a plausible distractor (could reasonably appear in an egocentric video)
3. Should be a common everyday object
4. Must NOT be semantically similar to the answer "{answer_obj}"
5. Must NOT be in the object group {{{', '.join(object_group)}}}

Available object pool (you can choose from these):
{', '.join(list(all_objects_pool)[:100])}

Please suggest ONE single replacement object name. Reply with ONLY the object name, nothing else.
Format: just the object name with spaces instead of underscores (e.g., "plate" or "cutting board")"""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        suggested_obj = output_text[0].strip()
        suggested_obj = suggested_obj.strip('"').strip("'").strip()
        suggested_obj = suggested_obj.replace('_', ' ')
        
        return suggested_obj
    
    except Exception as e:
        print(f"  [ERROR] Qwen3VL API call failed: {e}")
        # Fallback: return a random object from pool
        for obj in all_objects_pool:
            if obj not in used_objects and obj != answer_obj and obj not in object_group:
                return obj.replace('_', ' ')
        return None


def filter_past_next_after_group(data, log_file=None):
    """Filter past next after group tasks with Qwen3VL"""
    stats = {
        'initial': len(data),
        'filtered_human_objects': 0,
        'fixed_ambiguous': 0,
        'underscore_removed': 0,
        'final': 0
    }
    
    # Collect all objects first
    all_objects_pool = collect_all_objects(data)
    
    filtered_data = []
    
    # Remove underscores from all options
    for item in data:
        for question in item['questions']:
            for i, opt in enumerate(question['options']):
                obj_name = extract_object_name(opt)
                if '_' in obj_name:
                    new_obj_name = obj_name.replace('_', ' ')
                    option_letter = chr(65 + i)
                    question['options'][i] = f"{option_letter}. {new_obj_name}"
                    stats['underscore_removed'] += 1
            
            if 'answer_object' in question and '_' in question['answer_object']:
                question['answer_object'] = question['answer_object'].replace('_', ' ')
            
            if 'object_group' in question:
                question['object_group'] = [obj.replace('_', ' ') for obj in question['object_group']]
    
    # Re-collect after underscore removal
    all_objects_pool = collect_all_objects(data)
    
    for item in tqdm(data, desc="Filtering past_next_after_group"):
        question = item['questions'][0]
        
        # Check for human objects in options
        has_human = False
        for opt in question.get('options', []):
            obj = extract_object_name(opt)
            if is_human_related(obj):
                has_human = True
                if log_file:
                    log_file.write(f"[FILTERED - HUMAN OBJECT] {obj}\n")
                break
        
        if has_human:
            stats['filtered_human_objects'] += 1
            continue
        
        # Find ambiguous pairs
        ambiguous_pairs = find_ambiguous_options(question, strict_mode=True)
        
        if ambiguous_pairs:
            options = question['options']
            objects = [extract_object_name(opt) for opt in options]
            answer_obj = question.get('answer_object', '')
            object_group = question.get('object_group', [])
            
            # Fix ambiguous pairs
            for pair in ambiguous_pairs:
                idx_to_replace = None
                obj_to_replace = None
                
                # Decide which to replace
                if answer_obj == pair['obj1']:
                    idx_to_replace = pair['idx2']
                    obj_to_replace = pair['obj2']
                elif answer_obj == pair['obj2']:
                    idx_to_replace = pair['idx1']
                    obj_to_replace = pair['obj1']
                else:
                    idx_to_replace = pair['idx2']
                    obj_to_replace = pair['obj2']
                
                # Get replacement
                used_objects = set(objects)
                replacement = get_replacement_object_qwen3vl(
                    obj_to_replace, 
                    answer_obj, 
                    object_group, 
                    all_objects_pool, 
                    used_objects
                )
                
                if replacement and replacement not in used_objects:
                    option_letter = chr(65 + idx_to_replace)
                    question['options'][idx_to_replace] = f"{option_letter}. {replacement}"
                    objects[idx_to_replace] = replacement
                    stats['fixed_ambiguous'] += 1
                    
                    if log_file:
                        log_file.write(f"[FIXED] '{obj_to_replace}' -> '{replacement}'\n")
                
                time.sleep(0.5)  # Rate limiting
        
        filtered_data.append(item)
    
    stats['final'] = len(filtered_data)
    return filtered_data, stats


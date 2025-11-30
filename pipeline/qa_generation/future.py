import pandas as pd
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set random seed for reproducibility
random.seed(42)

# ========================================
# Qwen3 Model Setup (Global)
# ========================================
print("Loading Qwen3 model for future.py...")
qwen_model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("✓ Qwen3 model loaded successfully for future.py!")

# Global cache for action conversions (across all videos)
ACTION_CONVERSION_CACHE = {}

def qwen3_generate(prompt, max_tokens=500, temperature=0.3):
    """Helper function to call Qwen3 model"""
    messages = [{"role": "user", "content": prompt}]
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
    
    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output = qwen_tokenizer.decode(output_ids, skip_special_tokens=True)
    return output

# ========================================
# FUTURE PREDICTION QA FUNCTIONS
# ========================================

# 1. Future Action Prediction
def generate_future_action_qa(scanpath_obj_dict, video_path, video_categories='EGTEA', lookback_window=3):
    """
    미래 행동 예측: 최근 N개의 fixations를 보고 다음 action 예측
    Based on the last N fixations, what action will the user do next?
    
    Filtering criteria:
    - Only FOV objects are included in context_objects (outside FOV excluded)
    - Time gap between fixation and action must be between 2-120 seconds
    """
    print("  [Future] Generating action prediction QA...")
    print("    - Filtering: FOV objects only, time gap 2-120 seconds")
    
    def format_time(seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def convert_action_to_natural_language(action_label):
        """Convert action label to natural sentence using Qwen3"""
        try:
            prompt = f"""Convert this action label into a natural, fluent English sentence.

        Action label: {action_label}

        Rules:
        - Make it a clear, concise action description (e.g., "Take an eating utensil", "Cut the tomato")
        - Replace underscores with spaces
        - Keep it simple and natural
        - Return ONLY the sentence, no explanations

        Natural sentence:"""
            
            natural_action = qwen3_generate(prompt, max_tokens=50, temperature=0.3).strip()
            return natural_action if natural_action else action_label
            
        except Exception as e:
            print(f"Qwen3 conversion failed for '{action_label}': {e}")
            return action_label  # fallback to original
    
    # Extract timeline
    timeline = sorted(scanpath_obj_dict.items())
    
    # Find timepoints with actions
    action_indices = []
    for i, (ts, value) in enumerate(timeline):
        if isinstance(value, str):  # action label
            action_indices.append(i)
    
    print(f"    - Found {len(action_indices)} action timestamps in timeline")
    
    if len(action_indices) < 2:
        print(f"    ⚠️  Skipping: Need at least 2 actions, found {len(action_indices)}")
        return []
    
    # Collect all actions (for wrong answer options)
    all_actions = list(set([val for _, val in timeline if isinstance(val, str)]))
    
    print(f"    - Found {len(all_actions)} unique actions")
    
    if len(all_actions) < 4:
        print(f"    ⚠️  Skipping: Need at least 4 unique actions for multiple choice, found {len(all_actions)}")
        return []
    
    # Action conversion cache (convert only when needed)
    action_to_natural = {}
    conversion_count = [0]  # mutable counter for nested function
    
    def get_natural_action(action):
        """Lazy conversion: convert only needed actions"""
        if action not in action_to_natural:
            if action not in ACTION_CONVERSION_CACHE:
                conversion_count[0] += 1
                print(f"      [Convert #{conversion_count[0]}] {action}", end='')
                ACTION_CONVERSION_CACHE[action] = convert_action_to_natural_language(action)
                print(f" → {ACTION_CONVERSION_CACHE[action]}")
            action_to_natural[action] = ACTION_CONVERSION_CACHE[action]
        return action_to_natural[action]
    
    # Calculate response_time (ts is now (start, end) tuple)
    all_time_keys = [time_key for time_key, _ in timeline]
    start_time = min(start for start, end in all_time_keys)
    end_time = max(end for start, end in all_time_keys)
    response_time = f"[{format_time(start_time)} - {format_time(end_time)}]"
    
    questions = []
    seen_sequences = set()  # Prevent duplicate fixation sequences
    min_time_gap = 2.0  # Minimum 2-second interval
    
    # Filtering statistics
    filter_stats = {
        'total_candidates': 0,
        'filtered_no_fov_objects': 0,
        'filtered_no_future_action': 0,
        'filtered_time_gap': 0,
        'filtered_duplicate_sequence': 0,
        'final_qa': 0
    }
    
    # Generate QA based on fixation sequences
    print(f"    - Analyzing {len(timeline)} timeline entries for action prediction...")
    for i in range(len(timeline) - 1):
        time_key, val = timeline[i]
        
        # Skip if not a fixation
        if not isinstance(val, list) or len(val) == 0:
            continue
        
        filter_stats['total_candidates'] += 1
        
        # time_key is (start, end) tuple
        fixation_start, fixation_end = time_key
        
        # Collect fixations within lookback window
        fixation_sequence = []
        fixation_end_time = fixation_end  # Use the end time of this fixation
        for j in range(max(0, i - lookback_window + 1), i + 1):
            t_key, v = timeline[j]
            if isinstance(v, list) and len(v) > 0:
                # v is 2D list [[in FOV], [outside FOV]]
                # Collect only FOV objects (exclude outside FOV)
                if len(v[0]) > 0:  # v[0] = objects in FOV
                    fixation_sequence.extend(v[0])
        
        if len(fixation_sequence) == 0:
            filter_stats['filtered_no_fov_objects'] += 1
            continue
        
        # Remove duplicates while preserving order
        fixation_sequence = list(dict.fromkeys(fixation_sequence))
        
        # Convert fixation sequence to string for duplicate checking
        sequence_key = " → ".join(str(obj) for obj in fixation_sequence[-lookback_window:])
        if sequence_key in seen_sequences:
            filter_stats['filtered_duplicate_sequence'] += 1
            continue
        
        # Find action occurring at least min_time_gap after this fixation
        future_action = None
        future_action_time = None
        max_time_gap = 120.0  # Maximum 2-minute (120-second) limit
        filtered_by_time = False
        
        for j in range(i + 1, len(timeline)):
            action_time_key, action_val = timeline[j]
            if isinstance(action_val, str):  # action found
                # action_time_key is (start, end) tuple
                action_start, action_end = action_time_key
                # Check time gap: action start should be after fixation end + min_time_gap
                time_gap = action_start - fixation_end_time
                
                # Select only actions between min_time_gap and max_time_gap
                if time_gap >= min_time_gap and time_gap <= max_time_gap:
                    future_action = action_val
                    future_action_time = action_start  # Use action start time
                    break
                elif time_gap > max_time_gap:
                    # Filter out if time_gap is too large
                    filtered_by_time = True
                    break
        
        if future_action is None:
            if filtered_by_time:
                filter_stats['filtered_time_gap'] += 1
            else:
                filter_stats['filtered_no_future_action'] += 1
            continue
        
        seen_sequences.add(sequence_key)
        
        # Generate wrong answer options
        wrong_actions = [act for act in all_actions if act != future_action]
        if len(wrong_actions) < 3:
            continue
        
        wrong_sample = random.sample(wrong_actions, 3)
        all_choices = [future_action] + wrong_sample
        random.shuffle(all_choices)
        
        correct_answer = chr(65 + all_choices.index(future_action))
        
        # Generate context & collect timing
        context_objects_with_timing = []
        for j in range(max(0, i - lookback_window + 1), i + 1):
            t_key, v = timeline[j]
            if isinstance(v, list) and len(v) > 0:
                # t_key is (start, end) tuple
                t_start, t_end = t_key
                # v is 2D list [[in FOV], [outside FOV]]
                # Collect only FOV objects (v[0])
                if len(v[0]) > 0:
                    for obj in v[0]:
                        if obj not in [item['object'] for item in context_objects_with_timing]:
                            context_objects_with_timing.append({
                                'object': obj,
                                'timestamp': format_time(t_start)  # Use start time
                            })
        
        context_str = " → ".join(str(obj) for obj in fixation_sequence[-lookback_window:])
        
        # Convert options to natural sentences (only needed ones!)
        natural_options = [f"{chr(65+i)}. {get_natural_action(choice)}" for i, choice in enumerate(all_choices)]
        
        questions.append({
            'task_type': 'Future_Action_Prediction',
            'question': f'Based on the recent fixation pattern ({context_str}), what action will the user do next?',
            'time_stamp': format_time(fixation_end_time),  # fixation end time (question time)
            'answer': correct_answer,
            'options': natural_options,
            'required_ability': 'predictive_reasoning',
            'answer_action': get_natural_action(future_action),
            'context_objects': fixation_sequence[-lookback_window:],
            'context_objects_with_timing': context_objects_with_timing,
            'actual_action_time': format_time(future_action_time)
        })
        filter_stats['final_qa'] += 1
    
    # Return each question as a separate dictionary
    result = [{
        'response_time': response_time,
        'questions': [q],
        'video_categories': video_categories,
        'video': video_path
    } for q in questions]
    
    # Print filtering statistics
    print(f"    ✓ Generated {len(result)} action prediction QA")
    print(f"    ✓ Converted {conversion_count[0]} new actions (cache: {len(ACTION_CONVERSION_CACHE)} total)")
    print(f"    📊 Filtering statistics:")
    print(f"       - Total candidates: {filter_stats['total_candidates']}")
    print(f"       - Filtered (no FOV objects): {filter_stats['filtered_no_fov_objects']}")
    print(f"       - Filtered (no future action): {filter_stats['filtered_no_future_action']}")
    print(f"       - Filtered (time gap > 120s): {filter_stats['filtered_time_gap']}")
    print(f"       - Filtered (duplicate sequence): {filter_stats['filtered_duplicate_sequence']}")
    print(f"       - Final QA: {filter_stats['final_qa']}")
    
    return result


# 2. Object Remind
def generate_object_remind_qa(scanpath_obj_dict, video_path, video_categories='EGTEA'):
    """
    Object Remind: 특정 object가 등장할 때 알려주는 기능
    
    Args:
        scanpath_obj_dict: {timestamp: [fov_objects, outside_objects]}
        
    Returns:
        easy_qa_list: FOV objects (사용자가 직접 본 objects)
        hard_qa_list: Outside objects (시야 밖 objects)
    """
    print("  [Future] Generating object remind QA...")
    
    def format_time(seconds):
        """Convert seconds to MM:SS format"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def filter_furniture_with_qwen3(objects_list):
        """Filter out furniture/fixture objects using Qwen3"""
        try:
            prompt = f"""Given this list of objects, filter OUT furniture, fixtures, large appliances, and structural elements.
            Keep only meaningful interactive objects that would be useful to track.

            Objects: {', '.join(objects_list)}

            Rules:
            - EXCLUDE: countertop, counter, cabinet, microwave, refrigerator, oven, stove, sink, wall, floor, ceiling, curtain, furniture, etc.
            - KEEP: food items, utensils, tools, ingredients, containers, etc.
            - Return as JSON list: ["object1", "object2", ...]

            Filtered objects (JSON list only):"""
            
            result = qwen3_generate(prompt, max_tokens=200, temperature=0.3).strip()
            if '```' in result:
                result = result.split('```')[1].replace('json', '').strip()
            
            filtered = json.loads(result)
            return filtered
            
        except Exception as e:
            print(f"Qwen3 filtering failed: {e}, using fallback")
            # Fallback: keyword-based filtering
            furniture_keywords = ['counter', 'cabinet', 'microwave', 'refrigerator', 'oven', 
                                 'stove', 'sink', 'wall', 'floor', 'ceiling', 'curtain', 
                                 'countertop', 'furniture', 'shelf', 'drawer']
            return [obj for obj in objects_list if not any(kw in obj.lower() for kw in furniture_keywords)]
    
    # 1. Separate scanpath_obj_dict into FOV and Outside
    scanpath_fov_obj = {}
    scanpath_out_obj = {}
    
    # time == key (start, end) tuple / object == value 
    for time_key, objs_list in scanpath_obj_dict.items():
        # Skip if it's an action label (string)
        if isinstance(objs_list, str):
            continue
        
        # objs_list = [fov_objects, outside_objects]
        fov_objects = objs_list[0]
        outside_objects = objs_list[1]
        
        if len(fov_objects) > 0:
            scanpath_fov_obj[time_key] = fov_objects
        
        if len(outside_objects) > 0:
            scanpath_out_obj[time_key] = outside_objects
    
    print(f"    - FOV timestamps: {len(scanpath_fov_obj)}")
    print(f"    - Outside timestamps: {len(scanpath_out_obj)}")
    
    # 2. Extract unique objects
    unique_fov_objects = set()
    for objs in scanpath_fov_obj.values():
        unique_fov_objects.update(objs)
    
    unique_out_objects = set()
    for objs in scanpath_out_obj.values():
        unique_out_objects.update(objs)
    
    print(f"    - Unique FOV objects: {len(unique_fov_objects)}")
    print(f"    - Unique Outside objects: {len(unique_out_objects)}")
    
    # 3. Extract appearance times for each object
    def get_object_appearances(scanpath_dict):
        """Object별 등장 시간 리스트 반환
        Args:
            scanpath_dict: {(start, end): [fov_objects, outside_objects], ...}
        Returns:
            {object_name: [(start1, end1), (start2, end2), ...], ...}
        """
        object_appearances = {}
        for time_key, objects in sorted(scanpath_dict.items()):
            if isinstance(objects, list):
                for obj in objects:
                    if obj not in object_appearances:
                        object_appearances[obj] = []
                    object_appearances[obj].append(time_key)  # time_key is (start, end) tuple
        return object_appearances
    
    fov_appearances = get_object_appearances(scanpath_fov_obj)
    out_appearances = get_object_appearances(scanpath_out_obj)
    
    # 4. Filter furniture
    print(f"    - Filtering furniture from {len(unique_fov_objects)} FOV objects...")
    filtered_fov_objects = filter_furniture_with_qwen3(list(unique_fov_objects))
    print(f"      → {len(filtered_fov_objects)} FOV objects after filtering")
    
    print(f"    - Filtering furniture from {len(unique_out_objects)} Outside objects...")
    filtered_out_objects = filter_furniture_with_qwen3(list(unique_out_objects))
    print(f"      → {len(filtered_out_objects)} Outside objects after filtering")
    
    # 5. Get question start time from entire timeline
    all_time_keys = list(scanpath_fov_obj.keys()) + list(scanpath_out_obj.keys())
    if len(all_time_keys) == 0:
        return [], []
    
    # Question time: video start (start time of first timestamp)
    question_start_time = min(start for start, end in all_time_keys)
    question_time = format_time(question_start_time)
    
    easy_qa_list = []
    hard_qa_list = []
    
    # 6. Generate Easy QA: FOV objects (objects directly gazed at by user)
    # Select only one object per scene
    if len(filtered_fov_objects) > 0:
        # Randomly select one object
        selected_fov_obj = random.choice(filtered_fov_objects)
        print(f"    - Selected 1 FOV object for easy task: {selected_fov_obj}")
        filtered_fov_objects = [selected_fov_obj]  # Keep only one
    
    for obj_name in filtered_fov_objects:
        if obj_name in fov_appearances and len(fov_appearances[obj_name]) > 0:
            # All fixation times (each is (start, end) tuple)
            all_appearance_time_keys = sorted(fov_appearances[obj_name])
            first_appearance_start = all_appearance_time_keys[0][0]  # Start time of first fixation
            
            # Calculate fixation intervals (merge consecutive fixations into intervals)
            fixation_intervals = []
            fixation_intervals_sec = []  # Store as (start_sec, end_sec) format
            if len(all_appearance_time_keys) > 0:
                current_start, current_end = all_appearance_time_keys[0]
                
                for start, end in all_appearance_time_keys[1:]:
                    # New interval if gap is 10+ seconds
                    if start - current_end > 10:
                        fixation_intervals.append({
                            'start': format_time(current_start),
                            'end': format_time(current_end)
                        })
                        fixation_intervals_sec.append((current_start, current_end))
                        current_start, current_end = start, end
                    else:
                        # Merge if consecutive fixation (extend end time)
                        current_end = max(current_end, end)
                
                # Add last interval
                fixation_intervals.append({
                    'start': format_time(current_start),
                    'end': format_time(current_end)
                })
                fixation_intervals_sec.append((current_start, current_end))
            
            # OVO-Bench style: generate multiple evaluation timepoints
            test_info = []
            time_offsets = [-20, -10, 0, +10, +20]   # seconds before/after first appearance (smaller for fixation capture)
            
            # Video length limit: max 5-minute evaluation
            MAX_EVAL_DURATION = 300  # seconds
            max_eval_time = question_start_time + MAX_EVAL_DURATION
            
            # Debug: print fixation intervals
            print(f"      [DEBUG] Object: {obj_name}")
            print(f"      [DEBUG] Fixation intervals (raw): {all_appearance_time_keys[:3]}...")  # Show first 3
            print(f"      [DEBUG] First appearance start: {first_appearance_start}")
            print(f"      [DEBUG] Max eval time: {max_eval_time} (question_start: {question_start_time} + {MAX_EVAL_DURATION}s)")
            
            for offset in time_offsets:
                realtime_sec = first_appearance_start + offset
                
                # Skip if before question start
                if realtime_sec < question_start_time:
                    continue
                
                # Skip if exceeds max evaluation duration
                if realtime_sec > max_eval_time:
                    print(f"      [DEBUG] Skipping realtime={realtime_sec:.2f} (exceeds max {max_eval_time})")
                    continue
                
                # type: check if actually fixating at realtime
                # "Monitor my gaze and alert me when I gaze on the <obj>"
                # → Should alert if gazing at object at realtime (type=1)
                eval_type = 0
                for start_sec, end_sec in all_appearance_time_keys:
                    if start_sec <= realtime_sec <= end_sec:
                        eval_type = 1
                        print(f"      [DEBUG] realtime={realtime_sec:.2f} → type=1 (fixating: [{start_sec:.2f}, {end_sec:.2f}])")
                        break
                
                if eval_type == 0:
                    print(f"      [DEBUG] realtime={realtime_sec:.2f} → type=0 (not fixating)")
                
                test_info.append({
                    'realtime': format_time(int(realtime_sec)),
                    'type': eval_type
                })
            
            # Skip if no valid test points
            if len(test_info) == 0:
                continue
            
            # response_time: entire evaluation range
            last_realtime_str = test_info[-1]['realtime']
            qa_response_time = f"[{question_time} - {last_realtime_str}]"
            
            question_dict = {
                'task_type': 'Object_Remind_Easy',
                'question': f'Monitor my gaze and alert me when I gaze on the <{obj_name}>.',
                'time_stamp': question_time,
                'first_appearance': format_time(first_appearance_start),
                'fixation_intervals': fixation_intervals,  # Added: Fixation interval info
                'test_info': test_info,
                'required_ability': 'object_tracking',
                'difficulty': 'easy',
                'gaze_area': 'FOV',
                'target_object': obj_name
            }
            
            easy_qa_list.append({
                'response_time': qa_response_time,
                'questions': [question_dict],
                'video_categories': video_categories,
                'video': video_path
            })
    
    # 7. Generate Hard QA: Outside objects (only outside FOV, not overlapping with FOV)
    # Exclude objects overlapping with Easy
    hard_only_objects = [obj for obj in filtered_out_objects 
                         if obj not in filtered_fov_objects]
    
    print(f"    - Objects overlapping with FOV: {len(filtered_out_objects) - len(hard_only_objects)}")
    print(f"    - Hard-only objects: {len(hard_only_objects)}")
    
    for obj_name in hard_only_objects:
        if obj_name in out_appearances and len(out_appearances[obj_name]) > 0:
            # All appearance times (each is (start, end) tuple)
            all_appearance_time_keys = sorted(out_appearances[obj_name])
            first_appearance_start = all_appearance_time_keys[0][0]  # Start time of first appearance
            
            # Calculate appearance intervals (merge consecutive appearances into intervals)
            appearance_intervals = []
            appearance_intervals_sec = []  # Store as (start_sec, end_sec) format
            if len(all_appearance_time_keys) > 0:
                current_start, current_end = all_appearance_time_keys[0]
                
                for start, end in all_appearance_time_keys[1:]:
                    # New interval if gap is 10+ seconds
                    if start - current_end > 10:
                        appearance_intervals.append({
                            'start': format_time(current_start),
                            'end': format_time(current_end)
                        })
                        appearance_intervals_sec.append((current_start, current_end))
                        current_start, current_end = start, end
                    else:
                        # Merge if consecutive appearance (extend end time)
                        current_end = max(current_end, end)
                
                # Add last interval
                appearance_intervals.append({
                    'start': format_time(current_start),
                    'end': format_time(current_end)
                })
                appearance_intervals_sec.append((current_start, current_end))
            
            # OVO-Bench style: generate multiple evaluation timepoints
            test_info = []
            time_offsets = [-20, -10, 0, +10, +20]  # seconds before/after first appearance (smaller for fixation capture)
            
            # Video length limit: max 5-minute evaluation
            MAX_EVAL_DURATION = 300  # seconds
            max_eval_time = question_start_time + MAX_EVAL_DURATION
            
            for offset in time_offsets:
                realtime_sec = first_appearance_start + offset
                
                # Skip if before question start
                if realtime_sec < question_start_time:
                    continue
                
                # Skip if exceeds max evaluation duration
                if realtime_sec > max_eval_time:
                    continue
                
                # type: check if object exists in scene at realtime
                # "Watch the scene and alert me when the <obj> appears"
                # → Should alert if object is in scene at realtime (type=1)
                eval_type = 0
                for start_sec, end_sec in all_appearance_time_keys:
                    if start_sec <= realtime_sec <= end_sec:
                        eval_type = 1
                        break
                
                test_info.append({
                    'realtime': format_time(int(realtime_sec)),
                    'type': eval_type
                })
            
            # Skip if no valid test points
            if len(test_info) == 0:
                continue
            
            # response_time: entire evaluation range
            last_realtime_str = test_info[-1]['realtime']
            qa_response_time = f"[{question_time} - {last_realtime_str}]"
            
            # Hard task: proactive prompt
            question_dict = {
                'task_type': 'Object_Remind_Hard',
                'question': f'Watch the scene and alert me when the <{obj_name}> first appears.',
                'time_stamp': question_time,
                'first_appearance': format_time(first_appearance_start),
                'appearance_intervals': appearance_intervals,  # Added: Appearance interval info
                'test_info': test_info,
                'required_ability': 'object_tracking',
                'difficulty': 'hard',
                'gaze_area': 'Outside',
                'target_object': obj_name
            }
            
            hard_qa_list.append({
                'response_time': qa_response_time,
                'questions': [question_dict],
                'video_categories': video_categories,
                'video': video_path
            })
    
    print(f"    ✓ Generated {len(easy_qa_list)} easy remind QA and {len(hard_qa_list)} hard remind QA")
    return easy_qa_list, hard_qa_list


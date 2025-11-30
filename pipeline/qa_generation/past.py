import random

# 1. Next object after group
def generate_next_after_group_qa(scanpath_obj_dict, video_path):
    """What the user gazes at next after viewing an object group"""
    print("  [Task 1] Generating next after group QA...")
    qa_pairs = []
    
    timeline = sorted(scanpath_obj_dict.items())
    if len(timeline) < 2:
        return []
    
    # Collect all objects (excluding action strings)
    all_objects = set()
    for _, objs in timeline:
        # Process only if objs is a list (exclude action strings)
        if isinstance(objs, list) and len(objs) >= 2:
            if isinstance(objs[0], list):  # Objects in FOV
                all_objects.update(objs[0])
            if isinstance(objs[1], list):  # Objects outside FOV
                all_objects.update(objs[1])
    all_objects = list(all_objects)
    
    for i in range(len(timeline) - 1):
        current_time_key, current_objs = timeline[i]
        next_time_key, next_objs = timeline[i + 1]
        
        # Skip if current_objs or next_objs is not a list (i.e., action string)
        if not isinstance(current_objs, list) or not isinstance(next_objs, list):
            continue
        if len(current_objs) < 2 or len(next_objs) < 2:
            continue
        
        # Extract start times from (start, end) tuples
        current_start, current_end = current_time_key if isinstance(current_time_key, tuple) else (current_time_key, current_time_key)
        next_start, next_end = next_time_key if isinstance(next_time_key, tuple) else (next_time_key, next_time_key)
        
        # Use FOV objects at current timepoint as group
        current_group = current_objs[0]
        if not isinstance(current_group, list) or len(current_group) < 1:
            continue
            
        # FOV objects at next timepoint
        next_fov_objs = next_objs[0]
        if not isinstance(next_fov_objs, list) or len(next_fov_objs) == 0:
            continue
        
        # Correct answer: newly appeared object at next timepoint (not in current_group)
        new_objects = [obj for obj in next_fov_objs if obj not in current_group]
        if len(new_objects) == 0:
            continue  # Skip if no new objects
        correct_answer_obj = new_objects[0]
        
        # Generate wrong answers
        wrong_choices = [obj for obj in all_objects if obj != correct_answer_obj and obj not in current_group]
        if len(wrong_choices) < 3:
            continue
        wrong_sample = random.sample(wrong_choices, 3)
        
        # Options
        all_choices = [correct_answer_obj] + wrong_sample
        random.shuffle(all_choices)
        correct_answer = chr(65 + all_choices.index(correct_answer_obj))
        options = [f"{chr(65 + j)}. {obj}" for j, obj in enumerate(all_choices)]
        
        # Time information
        # Answer object show up timestamp
        answer_minutes = int(next_start // 60)
        answer_seconds = int(next_start % 60)
        answer_timestamp = f"{answer_minutes:02d}:{answer_seconds:02d}"
        
        # Object group timestamp (start, end)
        group_start_str = f"{int(current_start//60):02d}:{int(current_start%60):02d}"
        group_end_str = f"{int(current_end//60):02d}:{int(current_end%60):02d}"
        object_group_timestamp = f"({group_start_str}, {group_end_str})"
        
        # Response time: from object_group start to answer_object end (with buffer)
        response_start = max(0, current_start - 2)  # 2s buffer before group
        response_end = next_end + 2  # 2s buffer after answer
        response_time = f"[{int(response_start//60):02d}:{int(response_start%60):02d} - {int(response_end//60):02d}:{int(response_end%60):02d}]"
        
        # Object group as string
        group_str = "{" + ", ".join(current_group) + "}"
        
        qa_pairs.append({
            "response_time": response_time,
            "questions": [{
                "task_type": "Scanpath_Next_After_Group",
                "question": f"What object does the user gaze at next after looking at the {group_str}?",
                "time_stamp": answer_timestamp,  # Answer object timestamp
                "answer": correct_answer,
                "options": options,
                "required_ability": "scanpath_reasoning",
                "object_group": current_group,
                "object_group_timestamp": object_group_timestamp,  # NEW: (start, end)
                "answer_object": correct_answer_obj,
                "answer_timestamp": answer_timestamp  # NEW: when answer appears
            }],
            "video_categories": "EGTEA",
            "video_path": video_path
        })
    
    print(f"    ✓ Generated {len(qa_pairs)} next after group QA")
    return qa_pairs


# 2. Never gazed at
def generate_never_gazed_qa(scanpath_obj_dict, video_path):
    """Find objects that were never gazed at among multiple objects"""
    print("  [Task 2] Generating never gazed QA...")
    qa_pairs = []
    
    timeline = sorted(scanpath_obj_dict.items())
    if len(timeline) < 3:
        return []
    
    # Actually gazed objects (FOV only)
    gazed_objects = set()
    for _, objs in timeline:
        if isinstance(objs, list) and len(objs) >= 1 and isinstance(objs[0], list):
            gazed_objects.update(objs[0])  # FOV only
    
    # All objects in the scene (FOV + outside)
    all_scene_objects = set()
    for _, objs in timeline:
        if isinstance(objs, list) and len(objs) >= 2:
            if isinstance(objs[0], list):
                all_scene_objects.update(objs[0])
            if isinstance(objs[1], list):
                all_scene_objects.update(objs[1])
    
    # Never gazed objects = objects in scene but not gazed at
    never_gazed = list(all_scene_objects - gazed_objects)
    
    if len(never_gazed) == 0 or len(gazed_objects) < 3:
        return []
    
    # Generate questions at each timepoint
    for time_key, objs in timeline[2:]:  # After enough data has accumulated
        # Skip if objs is not a list
        if not isinstance(objs, list) or len(objs) < 2:
            continue
        
        # Extract start time from (start, end) tuple
        ts_start, ts_end = time_key if isinstance(time_key, tuple) else (time_key, time_key)
        
        # Objects gazed at up to this point
        gazed_so_far = set()
        for t_key, o in timeline:
            t_start, t_end = t_key if isinstance(t_key, tuple) else (t_key, t_key)
            if t_start <= ts_start and isinstance(o, list) and len(o) >= 1 and isinstance(o[0], list):
                gazed_so_far.update(o[0])
        
        # Objects not gazed at up to this point
        ungazed_so_far = [obj for obj in never_gazed if obj in all_scene_objects]
        if len(ungazed_so_far) == 0:
            continue
        
        # Correct answer: one of the not-gazed objects
        correct_answer_obj = random.choice(ungazed_so_far)
        
        # Wrong answers: 3 from gazed objects
        gazed_list = list(gazed_so_far)
        if len(gazed_list) < 3:
            continue
        wrong_sample = random.sample(gazed_list, 3)
        
        # Options
        all_choices = [correct_answer_obj] + wrong_sample
        random.shuffle(all_choices)
        correct_answer = chr(65 + all_choices.index(correct_answer_obj))
        options = [f"{chr(65 + j)}. {obj}" for j, obj in enumerate(all_choices)]
        
        # Time information (use start time)
        minutes = int(ts_start // 60)
        seconds = int(ts_start % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Response time: entire segment where objects in options appear
        # Collect all timepoints where wrong_sample (actually gazed objects) appear
        choice_objects_set = set(wrong_sample)  # Only gazed objects (excluding correct answer)
        min_time = float('inf')
        max_time = 0
        
        for t_key, o in timeline:
            t_start, t_end = t_key if isinstance(t_key, tuple) else (t_key, t_key)
            if isinstance(o, list) and len(o) >= 1 and isinstance(o[0], list):
                # If any FOV objects are in choice_objects
                if any(obj in choice_objects_set for obj in o[0]):
                    min_time = min(min_time, t_start)
                    max_time = max(max_time, t_end if isinstance(t_key, tuple) else t_start)
        
        # Add buffer (2s before and after)
        start_time = max(0, min_time - 2)
        end_time = max_time + 2
        response_time = f"[{int(start_time//60):02d}:{int(start_time%60):02d} - {int(end_time//60):02d}:{int(end_time%60):02d}]"
        
        # List of choice objects
        choice_objects_str = ", ".join([obj for obj in all_choices])
        
        qa_pairs.append({
            "response_time": response_time,
            "questions": [{
                "task_type": "Scanpath_Never_Gazed",
                "question": f"Among {{{choice_objects_str}}}, which did the user never gaze at?",
                "time_stamp": time_str,
                "answer": correct_answer,
                "options": options,
                "required_ability": "scanpath_reasoning",
                "answer_object": correct_answer_obj
            }],
            "video_categories": "EGTEA",
            "video_path": video_path
        })
        
        # Generate only a reasonable number (max 3)
        if len(qa_pairs) >= 3:
            break
    
    print(f"    ✓ Generated {len(qa_pairs)} never gazed QA")
    return qa_pairs


# 3. Transition pattern (sequence-level)
def generate_transition_pattern_qa(scanpath_obj_dict, video_path, num_samples=5):
    """Match object group transition patterns"""
    print("  [Task 3] Generating transition pattern QA...")
    qa_pairs = []
    
    timeline = sorted(scanpath_obj_dict.items())
    if len(timeline) < 3:
        return []
    
    for _ in range(num_samples):
        # Select 3 consecutive timepoints
        if len(timeline) < 3:
            continue
        start_idx = random.randint(0, len(timeline) - 3)
        
        # Correct transition sequence
        correct_sequence = []
        for i in range(start_idx, start_idx + 3):
            time_key, objs = timeline[i]
            # Skip if objs is not a list
            if not isinstance(objs, list) or len(objs) < 1:
                continue
            group = objs[0]  # Objects in FOV
            if isinstance(group, list) and len(group) > 0:
                correct_sequence.append(group)
        
        if len(correct_sequence) < 3:
            continue
        
        # Correct answer string
        correct_str = ">".join(["{" + ",".join(g) + "}" for g in correct_sequence])
        
        # Generate wrong answers (shuffle order, replace objects)
        wrong_options = []
        
        # Pool of all objects
        all_objects = set()
        for _, objs in timeline:
            if isinstance(objs, list) and len(objs) >= 1 and isinstance(objs[0], list):
                all_objects.update(objs[0])
        all_objects = list(all_objects)
        
        # Wrong answer 1: shuffle order
        shuffled_seq = correct_sequence.copy()
        random.shuffle(shuffled_seq)
        if shuffled_seq != correct_sequence:
            wrong_str = ">".join(["{" + ",".join(g) + "}" for g in shuffled_seq])
            wrong_options.append(wrong_str)
        
        # Wrong answers 2, 3: replace with random objects
        for _ in range(2):
            fake_seq = []
            for _ in range(3):
                fake_group = random.sample(all_objects, min(2, len(all_objects)))
                fake_seq.append(fake_group)
            fake_str = ">".join(["{" + ",".join(g) + "}" for g in fake_seq])
            if fake_str != correct_str and fake_str not in wrong_options:
                wrong_options.append(fake_str)
        
        if len(wrong_options) < 3:
            continue
        
        # Options
        all_choices = [correct_str] + wrong_options[:3]
        random.shuffle(all_choices)
        correct_answer = chr(65 + all_choices.index(correct_str))
        options = [f"{chr(65 + j)}. {choice}" for j, choice in enumerate(all_choices)]
        
        # Time information (end of sequence)
        start_time_key = timeline[start_idx][0]
        end_time_key = timeline[start_idx + 2][0]
        
        # Extract start times from (start, end) tuples
        start_ts, _ = start_time_key if isinstance(start_time_key, tuple) else (start_time_key, start_time_key)
        end_ts, _ = end_time_key if isinstance(end_time_key, tuple) else (end_time_key, end_time_key)
        
        minutes = int(end_ts // 60)
        seconds = int(end_ts % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        response_time = f"[{int(start_ts//60):02d}:{int(start_ts%60):02d} - {int(end_ts//60):02d}:{int(end_ts%60):02d}]"
        
        qa_pairs.append({
            "response_time": response_time,
            "questions": [{
                "task_type": "Scanpath_Transition_Pattern",
                "question": "Which transition best matches the user's gaze pattern?",
                "time_stamp": time_str,
                "answer": correct_answer,
                "options": options,
                "required_ability": "temporal_reasoning",
                "correct_sequence": correct_sequence
            }],
            "video_categories": "EGTEA",
            "video_path": video_path
        })
    
    print(f"    ✓ Generated {len(qa_pairs)} transition pattern QA")
    return qa_pairs


# 4. Scene Reconstruction
def generate_scene_reconstruction_qa(scanpath_obj_dict, video_path):
    """What background objects were NOT visible when gazing at specific objects"""
    print("  [Task 4] Generating scene reconstruction QA...")
    qa_pairs = []
    
    timeline = sorted(scanpath_obj_dict.items())
    if len(timeline) < 2:
        return []
    
    # Collect all outside objects
    all_outside_objects = set()
    for _, objs in timeline:
        if isinstance(objs, list) and len(objs) >= 2 and isinstance(objs[1], list):
            all_outside_objects.update(objs[1])
    all_outside_objects = list(all_outside_objects)
    
    if len(all_outside_objects) < 4:
        return []
    
    # Generate questions at each timepoint
    for time_key, objs in timeline:
        # Skip if objs is not a list
        if not isinstance(objs, list) or len(objs) < 2:
            continue
        
        # Extract start time from (start, end) tuple
        ts_start, ts_end = time_key if isinstance(time_key, tuple) else (time_key, time_key)
        
        fov_objects = objs[0]
        outside_objects = objs[1]
        
        # Only when there are FOV objects and 3+ outside objects
        if not isinstance(fov_objects, list) or not isinstance(outside_objects, list):
            continue
        if len(fov_objects) == 0 or len(outside_objects) < 3:
            continue
        
        # FOV objects as string (wrapped in braces)
        if len(fov_objects) == 1:
            fov_str = "{" + fov_objects[0] + "}"
        else:
            fov_str = "{" + ", ".join(fov_objects) + "}"
        
        # Correct answer: outside object not visible at this time (sample from other timepoints' outside objects)
        not_visible_objects = [obj for obj in all_outside_objects if obj not in outside_objects]
        if len(not_visible_objects) < 1:
            continue
        correct_answer_obj = random.choice(not_visible_objects)
        
        # Wrong answers: 3 outside objects actually visible at this time
        if len(outside_objects) < 3:
            continue
        wrong_sample = random.sample(outside_objects, 3)
        
        # Options
        all_choices = [correct_answer_obj] + wrong_sample
        random.shuffle(all_choices)
        correct_answer = chr(65 + all_choices.index(correct_answer_obj))
        options = [f"{chr(65 + j)}. {obj}" for j, obj in enumerate(all_choices)]
        
        # Time information (use start time)
        minutes = int(ts_start // 60)
        seconds = int(ts_start % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        start_time = max(0, ts_start - 5)
        end_time = ts_start + 5
        response_time = f"[{int(start_time//60):02d}:{int(start_time%60):02d} - {int(end_time//60):02d}:{int(end_time%60):02d}]"
        
        qa_pairs.append({
            "response_time": response_time,
            "questions": [{
                "task_type": "Scene_Reconstruction",
                "question": f"When the user was gazing at the {fov_str}, which background object was NOT visible?",
                "time_stamp": time_str,
                "answer": correct_answer,
                "options": options,
                "required_ability": "episodic_memory",
                "gazed_objects": fov_objects,
                "answer_object": correct_answer_obj,
                "background_objects": outside_objects
            }],
            "video_categories": "EGTEA",
            "video_path": video_path
        })
        
        # Generate only one per timepoint
        if len(qa_pairs) >= 3:
            break
    
    print(f"    ✓ Generated {len(qa_pairs)} scene reconstruction QA")
    return qa_pairs


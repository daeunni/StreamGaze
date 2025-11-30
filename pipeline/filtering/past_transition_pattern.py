"""
Past Transition Pattern Task Filtering

Complete implementation with Qwen3VL validation
"""

import os
import base64
import cv2
import json
import random
from tqdm import tqdm
from .utils import parse_time_to_seconds, get_qwen_model, get_qwen_processor


def extract_frames_from_video(video_path, start_sec, end_sec, num_frames=16):
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


def has_consecutive_identical_groups(sequence):
    """Check if sequence has consecutive identical groups"""
    for i in range(len(sequence) - 1):
        if set(sequence[i]) == set(sequence[i+1]):
            return True
    return False


def validate_with_qwen3vl(frames, question_data):
    """Validate transition pattern QA with Qwen3VL"""
    model = get_qwen_model()
    processor = get_qwen_processor()
    
    question = question_data['question']
    answer = question_data['answer']
    options = question_data['options']
    correct_sequence = question_data['correct_sequence']
    
    prompt = f"""You are analyzing an egocentric video clip to verify a transition pattern question.

**Task Type:** Transition Pattern (Past)
**Question:** {question}
**Options:** {', '.join(options)}
**Correct Sequence:** {correct_sequence}

**Your Task:**
1. Watch the video frames carefully
2. Verify if the transition pattern matches the correct sequence
3. Check if the options are distinct and reasonable

**Response Format:**
Return ONLY a JSON object:
{{
  "is_valid": 1 or 0,
  "reasoning": "brief explanation"
}}

- is_valid=1: Pattern is correct
- is_valid=0: Pattern is incorrect
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
        
        generated_ids = model.generate(**inputs, max_new_tokens=300)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        response_text = output_text[0].strip()
        
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        result = json.loads(response_text)
        return result.get('is_valid', 0), result.get('reasoning', '')
        
    except Exception as e:
        print(f"   ✗ Qwen3VL API Error: {e}")
        return None, str(e)


def filter_past_transition_pattern(data, log_file=None):
    """Filter past transition pattern tasks"""
    stats = {
        'initial': len(data),
        'filtered_consecutive_identical': 0,
        'filtered_short_sequence': 0,
        'qwen3vl_validated': 0,
        'qwen3vl_invalid': 0,
        'final': 0
    }
    
    filtered_data = []
    
    for item in tqdm(data, desc="Filtering past_transition_pattern"):
        q = item['questions'][0]
        video_path = item.get('video_path', item.get('video'))
        
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
        
        # Optional: Qwen3VL validation
        response_time = item.get('response_time', '[00:00 - 00:10]')
        time_range = response_time.strip('[]').split(' - ')
        start_sec = parse_time_to_seconds(time_range[0])
        end_sec = parse_time_to_seconds(time_range[1])
        
        frames = extract_frames_from_video(video_path, start_sec, end_sec, 16)
        
        if frames is not None:
            is_valid, reasoning = validate_with_qwen3vl(frames, q)
            
            if is_valid is not None:
                stats['qwen3vl_validated'] += 1
                
                if is_valid == 0:
                    stats['qwen3vl_invalid'] += 1
                    if log_file:
                        log_file.write(f"[FILTERED - QWEN3VL INVALID] {reasoning}\n")
                    continue
        
        filtered_data.append(item)
    
    stats['final'] = len(filtered_data)
    return filtered_data, stats


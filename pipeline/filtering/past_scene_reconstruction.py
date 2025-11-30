"""
Past Scene Reconstruction Task Filtering

Complete implementation with Qwen3VL validation
"""

import os
import base64
import cv2
import json
from tqdm import tqdm
from .utils import is_human_related, parse_time_to_seconds, get_qwen_model, get_qwen_processor


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


def validate_with_qwen3vl(frames, question_data):
    """Validate scene reconstruction QA with Qwen3VL"""
    model = get_qwen_model()
    processor = get_qwen_processor()
    
    question = question_data['question']
    answer = question_data['answer']
    options = question_data['options']
    answer_object = question_data['answer_object']
    
    prompt = f"""You are analyzing an egocentric video clip to verify a scene reconstruction question.

**Task Type:** Scene Reconstruction (Past)
**Question:** {question}
**Options:** {', '.join(options)}
**Given Answer:** {answer} ({answer_object})

**Your Task:**
1. Watch the video frames carefully
2. Verify if the question is clear and unambiguous
3. Check if the answer object ({answer_object}) is NOT visible in any of the frames
4. Ensure the options are reasonable and distinct

**Response Format:**
Return ONLY a JSON object:
{{
  "is_valid": 1 or 0,
  "reasoning": "brief explanation"
}}

- is_valid=1: Question is clear and answer is correct
- is_valid=0: Question is unclear or answer is incorrect
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


def filter_past_scene_reconstruction(data, log_file=None):
    """Filter past scene reconstruction tasks"""
    stats = {
        'initial': len(data),
        'filtered_short_clip': 0,
        'filtered_human_objects': 0,
        'filtered_gazed_in_options': 0,
        'qwen3vl_validated': 0,
        'qwen3vl_invalid': 0,
        'final': 0
    }
    
    filtered_data = []
    
    for item in tqdm(data, desc="Filtering past_scene_reconstruction"):
        q = item['questions'][0]
        video_path = item.get('video_path', item.get('video'))
        
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
            obj = opt.split('. ', 1)[1] if '. ' in opt else opt
            if obj.lower() in gazed_set:
                has_overlap = True
                if log_file:
                    log_file.write(f"[FILTERED - GAZED IN OPTIONS] {obj}\n")
                break
        
        if has_overlap:
            stats['filtered_gazed_in_options'] += 1
            continue
        
        # Optional: Qwen3VL validation (can be expensive/slow)
        # Extract frames and validate
        response_time = item['response_time']
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


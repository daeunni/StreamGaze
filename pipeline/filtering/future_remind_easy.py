"""
Future Remind Easy Task - Qwen3VL based GT Verification

Verifies if user actually gazes at the object in 10-second clips using Qwen3VL
"""

import os
import base64
import cv2
import time
from tqdm import tqdm
from .utils import parse_time_to_seconds, get_qwen_model, get_qwen_processor


def get_gaze_visualization_path(original_video_path, base_dir=None):
    """Convert original video path to gaze_visualization video path"""
    video_name = os.path.basename(original_video_path).replace('.mp4', '')
    
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'final_data')
    
    # Try EGTEA path
    egtea_path = os.path.join(base_dir, 'egtea', 'metadata', video_name, f'{video_name}_gaze_visualization.mp4')
    if os.path.exists(egtea_path):
        return egtea_path
    
    # Try EgoExo path
    egoexo_path = os.path.join(base_dir, 'egoexo', 'metadata', video_name, f'{video_name}_gaze_visualization.mp4')
    if os.path.exists(egoexo_path):
        return egoexo_path
    
    # Try HoloAssist path
    holo_path = os.path.join(base_dir, 'holoassist', 'metadata', video_name, f'{video_name}_gaze_visualization.mp4')
    if os.path.exists(holo_path):
        return holo_path
    
    return original_video_path


def extract_frames_from_clip(video_path, start_sec, duration=10, num_frames=8):
    """Extract frames from a specific clip"""
    if not os.path.exists(video_path):
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration) * fps)
    end_frame = min(end_frame, total_frames)
    
    if start_frame >= total_frames:
        cap.release()
        return []
    
    # Sample frames uniformly
    frame_indices = []
    if end_frame > start_frame:
        step = max(1, (end_frame - start_frame) // num_frames)
        frame_indices = list(range(start_frame, end_frame, step))[:num_frames]
    
    frames_base64 = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (512, 512))
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(frame_base64)
    
    cap.release()
    return frames_base64


def verify_gaze_qwen3vl(frames_base64, object_name, realtime_str):
    """Ask Qwen3VL if the user gazes at the object in the given frames"""
    if not frames_base64:
        return -1
    
    model = get_qwen_model()
    processor = get_qwen_processor()
    
    text_prompt = f"""You are analyzing a 10-second egocentric video clip with gaze visualization ending at timestamp {realtime_str}.

The video shows:
- Green dot: Current gaze point (where the user is looking)
- Objects in the scene

Question: Does the user gaze at the "{object_name}" in this clip (the 10 seconds leading up to {realtime_str})?

Task Context: "Monitor my gaze and alert me when I gaze on the <{object_name}>"

Rules:
- Answer "Yes" ONLY if the green gaze point is on/near the {object_name} in the frames
- The {object_name} should be visible AND the green gaze point should be directed at it
- Answer "No" if:
  - The {object_name} is NOT visible in the frames
  - The green gaze point is NOT directed at the {object_name}
  - The user is looking at other objects
- Consider all frames provided from this 10-second segment

Answer format: Start with either "Yes" or "No", then provide brief reasoning."""
    
    content = [{"type": "text", "text": text_prompt}]
    
    # Add frames
    for frame_b64 in frames_base64[:8]:
        content.append({
            "type": "image",
            "image": f"data:image/jpeg;base64,{frame_b64}"
        })
    
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
        
        answer = output_text[0].strip()
        answer_lower = answer.lower()
        
        if answer_lower.startswith('yes'):
            return 1
        elif answer_lower.startswith('no'):
            return 0
        else:
            if 'yes' in answer_lower[:50]:
                return 1
            elif 'no' in answer_lower[:50]:
                return 0
            return -1
            
    except Exception as e:
        print(f"  [ERROR] Qwen3VL API call failed: {e}")
        return -1


def filter_future_remind_easy(data, log_file=None):
    """Verify and fix GT for Future Remind Easy tasks using Qwen3VL"""
    stats = {
        'total_questions': 0,
        'total_tests': 0,
        'total_yes': 0,
        'total_no': 0,
        'verified_tests': 0,
        'skipped_tests': 0,
        'gt_changed': 0,
        'type_changes': {
            '0->1': 0,
            '1->0': 0,
            'unchanged': 0
        },
        'initial': len(data),
        'final': len(data)
    }
    
    if log_file:
        log_file.write("Strategy: Only verify type=0 (No) clips\n")
        log_file.write("Reason: type=1 (Yes) already human-verified\n\n")
    
    for item in tqdm(data, desc="Filtering future_remind_easy"):
        question = item['questions'][0]
        original_video_path = item.get('video', '')
        object_name = question.get('target_object', '')
        
        stats['total_questions'] += 1
        
        # Get gaze_visualization video path
        video_path = get_gaze_visualization_path(original_video_path)
        
        if not os.path.exists(video_path):
            if log_file:
                log_file.write(f"[SKIP] Video not found: {video_path}\n")
            stats['skipped_tests'] += len(question.get('test_info', []))
            continue
        
        # Process each test_info (only type=0)
        for test in question.get('test_info', []):
            realtime_str = test['realtime']
            original_type = test['type']
            
            stats['total_tests'] += 1
            
            # Skip if already Yes (type=1) - already human verified
            if original_type == 1:
                stats['total_yes'] += 1
                stats['type_changes']['unchanged'] += 1
                continue
            
            stats['total_no'] += 1
            
            # Extract frames from 10-second clip BEFORE realtime
            realtime_sec = parse_time_to_seconds(realtime_str)
            start_sec = max(0, realtime_sec - 10)
            
            frames = extract_frames_from_clip(video_path, start_sec, duration=10, num_frames=8)
            
            if not frames:
                if log_file:
                    log_file.write(f"[SKIP] Failed to extract frames: {video_path} @ {realtime_str}\n")
                stats['skipped_tests'] += 1
                continue
            
            # Verify with Qwen3VL
            gpt_result = verify_gaze_qwen3vl(frames, object_name, realtime_str)
            
            if gpt_result == -1:
                if log_file:
                    log_file.write(f"[ERROR] Qwen3VL verification failed: {object_name} @ {realtime_str}\n")
                stats['skipped_tests'] += 1
                continue
            
            stats['verified_tests'] += 1
            
            # Compare with original GT
            if gpt_result != original_type:
                stats['gt_changed'] += 1
                
                if original_type == 0 and gpt_result == 1:
                    stats['type_changes']['0->1'] += 1
                    change_type = "0->1 (GT: No, GPT: Yes)"
                else:
                    stats['type_changes']['1->0'] += 1
                    change_type = "1->0 (GT: Yes, GPT: No)"
                
                if log_file:
                    log_file.write(f"[CHANGED] {object_name} @ {realtime_str}\n")
                    log_file.write(f"  Video: {video_path}\n")
                    log_file.write(f"  Change: {change_type}\n\n")
                
                # Update GT
                test['type'] = gpt_result
            else:
                stats['type_changes']['unchanged'] += 1
            
            # Rate limiting
            time.sleep(0.5)
    
    return data, stats

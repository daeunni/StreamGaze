import tqdm
import os
import time
import json
from utils.data_execution import get_model_response
from benchmark.Benchmark import Benchmark

GAZE_INSTRUCTION = '''In this video, the green dot represents the gaze point (where the person is looking), and the red circle represents the field of view (FOV) area.

'''

PROMPT_TEMPLATE_REMIND = '''You are monitoring a video stream. Based on what you have seen so far in the video, answer the following question.
<video>

Question: {}

If you have already seen the requested object/event, answer "Yes".
If you have not seen it yet, answer "No".

Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
'''

def time_to_seconds(time_str):
    """Convert timestamp like '00:03:10' or '02:22' to seconds"""
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    else:
        return int(parts[0])

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

class StreamingBenchRemind_StreamGaze(Benchmark):
    """
    Benchmark for StreamGaze format data - Proactive/Remind Tasks
    Uses OVO-Bench style evaluation with multiple test points
    """
    def __init__(self, data, video_root, use_gaze_instruction=False):
        self.video_root = video_root
        self.data = data
        self.use_gaze_instruction = use_gaze_instruction

    def eval(self, data, model, output_path):
        """
        Evaluate model on StreamGaze format data for Proactive/Remind tasks
        
        Expected format:
        {
            "response_time": "[00:08 - 02:42]",
            "questions": [
                {
                    "question": "Monitor my gaze and alert me when I gaze on the <plate>.",
                    "time_stamp": "00:08",
                    "first_appearance": "02:22",
                    "target_object": "plate",
                    "test_info": [
                        {
                            "realtime": "02:02",
                            "type": 0,  # 0: before appearance, 1: after appearance
                            "input_video_clip": [0, 122]
                        },
                        ...
                    ]
                }
            ],
            "video_path": "video.mp4"
        }
        """
        for subset in tqdm.tqdm(data, desc="Evaluating Proactive/Remind Tasks"):
            video_path = subset.get("video_path", "")
            video_name = os.path.basename(video_path)
            
            # Get the actual video file path
            if os.path.exists(video_path):
                file = video_path
            else:
                file = os.path.join(self.video_root, video_name)
            
            if not os.path.exists(file):
                print(f"⚠️  Video not found: {file}")
                continue
            
            for question in subset["questions"]:
                # Get test_info for OVO-Bench style evaluation
                test_info = question.get("test_info", [])
                if not test_info:
                    print(f"Warning: No test_info found for {question.get('target_object', 'unknown')}, skipping...")
                    continue
                
                timestamp = question.get("time_stamp", "0:00")
                first_appearance = question.get("first_appearance", "0:00")
                target_object = question.get("target_object", "unknown")
                
                # Apply prompt template for Yes/No response with optional gaze instruction
                gaze_prefix = GAZE_INSTRUCTION if self.use_gaze_instruction else ""
                inp = gaze_prefix + PROMPT_TEMPLATE_REMIND.format(question['question'])
                
                print(f"\n  Video: {video_name}, Object: {target_object}, First appearance: {first_appearance}")
                
                # Loop through each evaluation time point (OVO-Bench style)
                for i, test in enumerate(test_info):
                    # Skip if already evaluated
                    if 'response' in test:
                        continue
                    
                    realtime = test['realtime']
                    eval_type = test['type']  # 0: before appearance, 1: after appearance
                    
                    # Video segment: always from 0 to realtime (OVO-Bench style)
                    start_time = 0
                    end_time = time_to_seconds(realtime)
                    
                    print(f"    [{i+1}/{len(test_info)}] Realtime: {realtime} (0s → {end_time}s), Expected: {'Yes' if eval_type == 1 else 'No'}")
                    
                    time_s = time.time()
                    
                    # Extract salience_map_path if available
                    salience_map_path = test.get('salience_map_path', None)
                    
                    # Get model response (single Yes/No response)
                    response, response_time_sec = get_model_response(
                        model, 
                        file, 
                        inp, 
                        start_time,  # Always from 0
                        end_time,    # Up to realtime
                        max(0, start_time - 1),  # Frame before start
                        False,  # not multimodal conversation
                        False,  # NOT proactive (single response)
                        salience_map_path  # Salience map image path
                    )
                    
                    time_e = time.time()
                    timecost = time_e - time_s
                    
                    # Store response in test_info
                    test['response'] = response
                    test['response_time_sec'] = response_time_sec
                    test['cost'] = timecost
                    
                    print(f"       Response: {response}, Time: {response_time_sec:.2f}s")
                
                # Save incrementally after each question
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=4)
        
        print(f"✅ Evaluation complete! Results saved to {output_path}")


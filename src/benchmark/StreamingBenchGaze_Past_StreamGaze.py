from tqdm import tqdm
import os
import json
from utils.data_execution import get_model_response
from benchmark.Benchmark import Benchmark

def parse_timestamp(timestamp_str):
    """Convert MM:SS timestamp string to seconds"""
    parts = timestamp_str.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    else:
        return float(timestamp_str)

GAZE_INSTRUCTION = '''In this video, the green dot represents the gaze point (where the person is looking), and the red circle represents the field of view (FOV) area.

'''

PROMPT_TEMPLATE = '''
Question: {}
Options:
{}
{}
{}
{}
Answer the question with only the letter (A, B, C, or D) of the correct option.'''

PROMPT_TEMPLATE_WITHOUT_OPTIONS = '''
Question: {}
Analyze the video and provide the answer to the question.
'''

class StreamingBenchGaze_Past_StreamGaze(Benchmark):
    """
    Benchmark for StreamGaze format data - Past Tasks
    Uses full video from start to question timestamp
    """
    def __init__(self, data, video_root, use_gaze_instruction=False):
        self.video_root = video_root
        self.data = data
        self.use_gaze_instruction = use_gaze_instruction

    def eval(self, data, model, output_path):
        """
        Evaluate model on StreamGaze format data for Past tasks
        
        Past tasks use the full video from 0 to question timestamp
        
        Expected format:
        {
            "response_time": "[02:22 - 13:20]",
            "questions": [
                {
                    "question": "...",
                    "time_stamp": "13:20",
                    "answer": "C",
                    "options": ["A. ...", "B. ...", ...]
                }
            ],
            "video_path": "video.mp4"
        }
        """
        results = []
        
        for entry in tqdm(data, desc="Evaluating Past Tasks"):
            video_path = entry.get("video_path", "")
            video_name = os.path.basename(video_path)
            
            # Get the actual video file path
            if os.path.exists(video_path):
                file = video_path
            else:
                file = os.path.join(self.video_root, video_name)
            
            if not os.path.exists(file):
                print(f"⚠️  Video not found: {file}")
                results.append(entry)
                continue
            
            # Process each question in the entry
            questions = entry.get("questions", [])
            if not questions:
                print(f"⚠️  No questions found in entry")
                results.append(entry)
                continue
            
            # Create result entry
            result_entry = entry.copy()
            result_entry["model_predictions"] = []
            
            for question_data in questions:
                question_text = question_data.get("question", "")
                time_stamp = question_data.get("time_stamp", "0:00")
                options = question_data.get("options", [])
                
                # Parse timestamp to seconds
                question_time = parse_timestamp(time_stamp)
                
                # Build prompt with optional gaze instruction
                gaze_prefix = GAZE_INSTRUCTION if self.use_gaze_instruction else ""
                
                if options:
                    # Ensure options start with A., B., C., D.
                    if not options[0].startswith("A."):
                        formatted_options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
                    else:
                        formatted_options = options
                    
                    # Build prompt with variable number of options
                    if len(formatted_options) == 2:
                        inp = gaze_prefix + f"Question: {question_text}\nOptions:\n{formatted_options[0]}\n{formatted_options[1]}\n\nThe best option is:"
                    elif len(formatted_options) == 4:
                        inp = gaze_prefix + PROMPT_TEMPLATE.format(question_text, *formatted_options)
                        inp += "\n\nThe best option is:"
                    else:
                        options_str = "\n".join(formatted_options)
                        inp = gaze_prefix + f"Question: {question_text}\nOptions:\n{options_str}\n\nThe best option is:"
                else:
                    inp = gaze_prefix + PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(question_text)
                    inp += "\n\nAnswer:"
                
                # Get model response
                # Past tasks: use full video from 0 to question timestamp
                start_time = 0
                end_time = question_time
                
                response, response_time = get_model_response(
                    model,
                    file,
                    inp,
                    start_time=start_time,
                    end_time=end_time,
                    question_time=question_time + 0.1,
                    salience_map_path=None
                )
                
                # Store prediction
                prediction = {
                    "question": question_text,
                    "time_stamp": time_stamp,
                    "model_prediction": response,
                    "model_response_time_sec": response_time,
                    "model_response_time_formatted": f"{int(response_time // 60)}:{int(response_time % 60):02d}"
                }
                result_entry["model_predictions"].append(prediction)
            
            results.append(result_entry)
            
            # Save incrementally
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        
        print(f"✅ Evaluation complete! Results saved to {output_path}")


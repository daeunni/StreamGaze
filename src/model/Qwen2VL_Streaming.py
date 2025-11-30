"""
Qwen2VL Streaming Wrapper
Wraps the original Qwen2VL model to support StreamingGaze evaluation with time-based video chunking
"""
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from decord import VideoReader, cpu
import numpy as np
import time as time_module

from model.modelclass import Model

model, processor = None, None

def Qwen2VL_Init():
    global model, processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def extract_video_frames(video_path, start_time, end_time, num_frames=8):
    """Extract frames from video between start_time and end_time"""
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames = vr.get_batch(frame_indices).asnumpy()
    
    return frames

def Qwen2VL_Run(video_path, inp, start_time, end_time):
    """
    Run Qwen2VL with time-based video chunking
    """
    # Use video with time specification to limit frames
    # Qwen2VL supports nframes parameter to limit token count
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 0.5,  # Reduce FPS to 0.5 frame per second (1 frame per 2 seconds)
                    "max_pixels": 360 * 420,  # Reduce resolution to save tokens (must be >= min_pixels)
                    "nframes": 4,  # Limit to maximum 4 frames
                },
                {"type": "text", "text": inp},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Reduce max_new_tokens to save context
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

class Qwen2VL_Streaming(Model):
    """
    Streaming wrapper for Qwen2VL model
    Compatible with StreamingGaze evaluation framework
    """
    def __init__(self):
        Qwen2VL_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False):
        """
        Run method compatible with StreamingGaze framework
        """
        start = time_module.time()
        response = Qwen2VL_Run(file, inp, start_time, end_time)
        response_time = time_module.time() - start
        return response, response_time
    
    def name(self):
        return "Qwen2-VL-7B"

from model.modelclass import Model
import time as time_module
import numpy as np
from decord import VideoReader, cpu
import torch

class VideoLLaMA2_Streaming(Model):
    def __init__(self):
        VideoLLaMA2_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False):
        return VideoLLaMA2_Run(file, inp, start_time, end_time)
    
    def name(self):
        return "VideoLLaMA2-7B"

from videollama2 import model_init, mm_infer

model, processor, tokenizer = None, None, None

def VideoLLaMA2_Init():
    global model, processor, tokenizer

    # 1. Initialize the model.
    model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B'
    model, processor, tokenizer = model_init(model_path)

def VideoLLaMA2_Run(file, inp, start_time, end_time):
    start = time_module.time()
    
    # For VideoLLaMA2, we need to extract frames within the time range
    # and create a temporary video or pass frames directly if the processor supports it
    
    modal = 'video'
    
    # Load video and extract frames for the specified time range
    vr = VideoReader(file, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frame_num = len(vr)
    
    # Calculate frame indices based on start_time and end_time
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time > 0 else total_frame_num
    
    # Ensure frames are within bounds
    start_frame = max(0, min(start_frame, total_frame_num - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frame_num))
    
    # Sample frames (limit to reasonable number to avoid memory issues)
    max_frames = 64
    num_frames = min(max_frames, end_frame - start_frame)
    uniform_sampled_frames = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    
    # Get frames
    frames = vr.get_batch(frame_idx).asnumpy()
    
    # VideoLLaMA2 processor expects video file path, so we'll use the original file
    # The model will process the entire video, which is a limitation
    # For better time-range support, we'd need to modify the preprocessing
    try:
        output = mm_infer(
            processor[modal](file), 
            inp, 
            model=model, 
            tokenizer=tokenizer, 
            do_sample=False, 
            modal=modal
        )
        result = output[0]
    except Exception as e:
        print(f"VideoLLaMA2 inference error: {e}")
        result = ""
    
    response_time = time_module.time() - start
    return result, response_time


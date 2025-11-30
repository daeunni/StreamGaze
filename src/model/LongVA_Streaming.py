from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import time as time_module

# fix seed
torch.manual_seed(0)

tokenizer, model, image_processor, max_frames_num, gen_kwargs = None, None, None, None, None

from model.modelclass import Model

class LongVA_Streaming(Model):
    def __init__(self):
        LongVA_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False):
        return LongVA_Run(file, inp, start_time, end_time)
    
    def name(self):
        return "LongVA-7B"

def LongVA_Init():
    global tokenizer, model, image_processor, max_frames_num, gen_kwargs

    model_path = "lmms-lab/LongVA-7B"

    # Reduce max_frames for streaming to avoid memory issues
    max_frames_num = 64  # Reduced from 128
    gen_kwargs = {"do_sample": False, "temperature": 0.0, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 512}
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto")

def LongVA_Run(file, inp, start_time, end_time):
    start = time_module.time()
    
    video_path = file
    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{}<|im_end|>\n<|im_start|>assistant\n".format(inp)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    
    # Load video with time range
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frame_num = len(vr)
    
    # Calculate frame indices based on start_time and end_time
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time > 0 else total_frame_num
    
    # Ensure frames are within bounds
    start_frame = max(0, min(start_frame, total_frame_num - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frame_num))
    
    # Sample frames uniformly within the time range
    num_frames = min(max_frames_num, end_frame - start_frame)
    uniform_sampled_frames = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor], modalities=["video"], **gen_kwargs)
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    response_time = time_module.time() - start
    return outputs, response_time


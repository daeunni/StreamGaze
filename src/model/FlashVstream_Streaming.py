"""
FlashVstream Streaming Wrapper
Wraps the original FlashVstream model to support StreamingGaze evaluation with time-based video chunking
"""
import torch
from decord import VideoReader, cpu
from .flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .flash_vstream.conversation import conv_templates, SeparatorStyle
from .flash_vstream.model.builder import load_pretrained_model
from .flash_vstream.utils import disable_torch_init
from .flash_vstream.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import time as time_module
import numpy as np

from model.modelclass import Model

tokenizer, model, image_processor, context_len = None, None, None, None

def FlashVstream_Init():
    global tokenizer, model, image_processor, context_len
    model_path = "IVGSZ/Flash-VStream-7b"
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, device="cuda", device_map="auto"
    )
    print("FlashVstream Model initialized.")

def load_video_with_time(video_path, start_time, end_time, fps_sample=1):
    """
    Load video frames between start_time and end_time
    Args:
        video_path: path to video
        start_time: start time in seconds
        end_time: end time in seconds
        fps_sample: sample every N frames based on video fps
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    
    # Convert time to frame indices
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    
    # Sample frames at specified fps
    frame_step = max(1, int(fps / fps_sample))
    frame_idx = list(range(start_frame, end_frame, frame_step))
    
    if not frame_idx:
        frame_idx = [start_frame]
    
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def FlashVstream_Run(video_path, inp, start_time, end_time):
    """
    Run FlashVstream with time-based video chunking
    """
    # Load video frames within time range
    video = load_video_with_time(video_path, start_time, end_time, fps_sample=1)
    video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
    video = [video]

    qs = inp
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)  # Fixed: use qs instead of inp to include IMAGE_TOKEN
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video,
            do_sample=True,
            temperature=0.002,
            max_new_tokens=1024,
            use_cache=False,  # Changed from True to False to fix cache_position compatibility
            stopping_criteria=[stopping_criteria]
        )
        
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    print(f'FlashVstream: {outputs}')
    return outputs

class FlashVstream_Streaming(Model):
    """
    Streaming wrapper for FlashVstream model
    Compatible with StreamingGaze evaluation framework
    """
    def __init__(self):
        FlashVstream_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        """
        Run method compatible with StreamingGaze framework
        """
        start = time_module.time()
        response = FlashVstream_Run(file, inp, start_time, end_time)
        response_time = time_module.time() - start
        return response, response_time
    
    def name(self):
        return "Flash-VStream"

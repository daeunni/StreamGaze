import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
import time as time_module
import numpy as np

from model.modelclass import Model

class MiniCPMV_Streaming(Model):
    def __init__(self, model_path='openbmb/MiniCPM-V-2_6'):
        """
        Initialize the model by loading the pretrained MiniCPM-V model and tokenizer.
        """
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            attn_implementation='sdpa', 
            torch_dtype=torch.bfloat16
        )
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.MAX_NUM_FRAMES = 64  # Maximum number of frames to process

    def encode_video(self, video_path, start_time=0, end_time=0):
        """
        Encode the video frames from the video path with time range support.
        """
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frame_num = len(vr)
        
        # Calculate frame indices based on start_time and end_time
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time > 0 else total_frame_num
        
        # Ensure frames are within bounds
        start_frame = max(0, min(start_frame, total_frame_num - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frame_num))
        
        # Sample frames with FPS
        sample_fps = round(fps / 1)  # FPS
        frame_idx = [i for i in range(start_frame, end_frame, sample_fps)]
        
        if len(frame_idx) > self.MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, self.MAX_NUM_FRAMES)
        
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        
        return frames

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False):
        """
        Given the file (video file path) and input prompt (inp), run the model and return the response.
        Adapted for StreamingGaze with time range support.
        """
        start = time_module.time()
        
        frames = self.encode_video(file, start_time, end_time)
        msgs = [
            {'role': 'user', 'content': frames + [inp]},
        ]

        # Set decode parameters for video
        params = {
            "use_image_id": False,
            "max_slice_nums": 1  # Adjust if CUDA OOM and video resolution > 448x448
        }

        # Generate the response using the model
        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **params
        )
        
        response_time = time_module.time() - start
        return answer, response_time

    def name(self):
        """
        Return the name of the model
        """
        return "MiniCPM-V-2.6"


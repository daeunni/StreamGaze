"""
VideollmOnline Streaming Wrapper
Wraps the original VideollmOnline model to support StreamingGaze evaluation with time-based video chunking
"""
import os
import tempfile
import torch
import transformers
import time as time_module
import subprocess
from dataclasses import dataclass

logger = transformers.logging.get_logger('liveinfer')

from model.modelclass import Model

# Add videollm-online to path
import sys
videollm_online_path = os.environ.get('VIDEOLLM_ONLINE_PATH', 'path/to/videollm-online')
if videollm_online_path not in sys.path:
    sys.path.insert(0, videollm_online_path)

try:
    from models import build_model_and_tokenizer, fast_greedy_generate
    from models.arguments_live import LiveTrainingArguments
    from data.utils import ffmpeg_once
    LIVEINFER_AVAILABLE = True
except Exception as e:
    LIVEINFER_AVAILABLE = False
    logger.warning(f"VideollmOnline modules not available: {e}. VideollmOnline_Streaming will not work.")

class VideollmOnline_Streaming(Model):
    """
    Streaming wrapper for VideollmOnline model
    Compatible with StreamingGaze evaluation framework
    """
    def __init__(self):
        """
        Initialize the model directly without using LiveInfer class
        """
        super().__init__()
        if not LIVEINFER_AVAILABLE:
            raise ImportError("VideollmOnline modules not available. Cannot initialize VideollmOnline_Streaming.")
        
        # Create args manually instead of parsing from command line
        # Use local model path if set via environment variable, otherwise use HuggingFace
        model_checkpoint = os.environ.get(
            'VIDEOLLM_ONLINE_MODEL_PATH',
            'chenjoya/videollm-online-8b-v1plus'
        )
        args = LiveTrainingArguments(
            output_dir='outputs/debug',
            resume_from_checkpoint=model_checkpoint,
            frame_fps=2,
            frame_resolution=384,
            system_prompt="A multimodal AI assistant is helping users with some activities. Below is their conversation, interleaved with the list of video frames received by the assistant."
        )
        
        # Build model and tokenizer directly
        from dataclasses import asdict
        self.model, self.tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=True, **asdict(args))
        self.model.to('cuda')
        
        # Setup visual parameters (from LiveInfer.__init__)
        self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        self.frame_interval = 1 / self.frame_fps
        self.frame_resolution = self.model.config.frame_resolution
        self.frame_num_tokens = self.model.config.frame_num_tokens if self.model.config.frame_num_tokens is not None else 256  # default value
        self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens
        self.frame_token_interval_id = self.model.config.frame_token_interval_id if self.model.config.frame_token_interval_id is not None else 0
        self.frame_placeholder_ids = torch.tensor([self.model.config.v_placeholder_id] if isinstance(self.model.config.v_placeholder_id, int) else self.model.config.v_placeholder_id).repeat(self.frame_num_tokens).reshape(1,-1)
        
        # Setup generation parameters
        self.system_prompt = args.system_prompt
        self.inplace_output_ids = torch.zeros(1, 100, device='cuda', dtype=torch.long)
        self.frame_token_interval_threshold = 0.725
        self.eos_token_id = self.model.config.eos_token_id
        self._start_ids = self.tokenizer.apply_chat_template([{'role': 'system', 'content': self.system_prompt}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        self._added_stream_prompt_ids = self.tokenizer.apply_chat_template([{}], add_stream_prompt=True, return_tensors='pt').to('cuda')
        self._added_stream_generation_ids = self.tokenizer.apply_chat_template([{}], add_stream_generation_prompt=True, return_tensors='pt').to('cuda')
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset state for new video"""
        import collections
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.last_frame_idx = -1
        self.video_tensor = None
        self.last_ids = torch.tensor([[]], device='cuda', dtype=torch.long)
        self.past_key_values = None
    
    def load_video(self, video_path):
        """Load video into memory"""
        from torchvision.io import read_video
        self.video_tensor = read_video(video_path, pts_unit='sec', output_format='TCHW')[0].to('cuda')
        self.num_video_frames = self.video_tensor.size(0)
        self.video_duration = self.video_tensor.size(0) / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS')
    
    def input_query_stream(self, query, history=None, video_time=None):
        """Add query to queue"""
        if video_time is None:
            self.query_queue.append((self.video_time, query))
        else:
            self.query_queue.append((video_time, query))
    
    def input_video_stream(self, video_time):
        """Process video frames up to video_time"""
        frame_idx = int(video_time * self.frame_fps)
        
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            frames_embeds = self.model.visual_embed(self.video_tensor[ranger]).split(self.frame_num_tokens)
            self.frame_embeds_queue.extend([(r / self.frame_fps, frame_embeds) for r, frame_embeds in zip(ranger, frames_embeds)])
        
        self.last_frame_idx = frame_idx
        self.video_time = video_time
    
    def _call_for_streaming(self):
        """Process frame embeddings and check if response needed"""
        import collections
        
        while self.frame_embeds_queue:
            # Check if query should be answered before next frame
            if self.query_queue and self.frame_embeds_queue[0][0] > self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            
            video_time, frame_embeds = self.frame_embeds_queue.popleft()
            
            # Initialize if no past key values
            if not self.past_key_values:
                self.last_ids = self._start_ids
            elif self.last_ids == self.eos_token_id:
                self.last_ids = torch.cat([self.last_ids, self._added_stream_prompt_ids], dim=1)
            
            # Combine previous text and current frame embedding
            inputs_embeds = torch.cat([
                self.model.get_input_embeddings()(self.last_ids).view(1, -1, self.hidden_size),
                frame_embeds.view(1, -1, self.hidden_size),
            ], dim=1)
            
            outputs = self.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.past_key_values)
            self.past_key_values = outputs.past_key_values
            
            # Check if query should be answered at current time
            if self.query_queue and video_time >= self.query_queue[0][0]:
                video_time, query = self.query_queue.popleft()
                return video_time, query
            
            # Check if model wants to respond proactively
            next_score = outputs.logits[:,-1:].softmax(dim=-1)
            if next_score[:,:,self.frame_token_interval_id] < self.frame_token_interval_threshold:
                next_score[:,:,self.frame_token_interval_id].zero_()
            
            self.last_ids = next_score.argmax(dim=-1)
            
            if self.last_ids != self.frame_token_interval_id:
                return video_time, None
        
        return None, None
    
    def _call_for_response(self, video_time, query):
        """Generate response for query at video_time"""
        if query is not None:
            self.last_ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': query}], add_stream_query_prompt=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
            # Debug: print encoded query
            encoded_query = self.tokenizer.decode(self.last_ids[0], skip_special_tokens=False)
            print(f'DEBUG - Encoded query: {encoded_query}', flush=True)
        else:
            self.last_ids = self._added_stream_generation_ids
        
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        output_ids, self.past_key_values = fast_greedy_generate(model=self.model, inputs_embeds=inputs_embeds, past_key_values=self.past_key_values, eos_token_id=self.eos_token_id, inplace_output_ids=self.inplace_output_ids)
        self.last_ids = output_ids[:, -1:]
        
        if query:
            query = f'(Video Time = {video_time}s) User: {query}'
        
        decoded_response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f'DEBUG - Decoded response: {decoded_response}', flush=True)
        response = f'(Video Time = {video_time}s) Assistant:{decoded_response}'
        return query, response
    
    def __call__(self):
        """Main inference loop"""
        import collections
        
        while not self.frame_embeds_queue:
            continue
        
        video_time, query = self._call_for_streaming()
        response = None
        
        if video_time is not None:
            query, response = self._call_for_response(video_time, query)
        
        return query, response

    def chunk_video(self, video_path, start_time, end_time):
        """
        Create a temporary video chunk from start_time to end_time using decord and cv2
        """
        import cv2
        import decord
        from decord import VideoReader
        
        # Create temporary output file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file_path = temp_file.name
        temp_file.close()
        
        try:
            # Read video using decord
            vr = VideoReader(video_path, ctx=decord.cpu(0))
            fps = vr.get_avg_fps()
            
            # Calculate frame indices
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            total_frames = len(vr)
            
            # Clip to valid range
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            
            # Extract frames
            frames = vr.get_batch(range(start_frame, end_frame)).asnumpy()
            
            if len(frames) == 0:
                raise ValueError(f"No frames extracted from {video_path} between {start_time}s and {end_time}s")
            
            # Get video properties
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_file_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in frames:
                # Convert RGB to BGR for cv2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
        except Exception as e:
            logger.error(f"Video chunking error: {str(e)}")
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise
        
        return temp_file_path

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        """
        Run method compatible with StreamingGaze framework
        
        Args:
            file: video file path
            inp: input prompt
            start_time: start time in seconds
            end_time: end time in seconds
            question_time: question timestamp
            omni: multimodal flag
            proactive: proactive response flag
        
        Returns:
            tuple: (response, response_time)
        """
        start = time_module.time()
        
        # Create video chunk for the time range
        chunk_path = None
        try:
            chunk_path = self.chunk_video(file, start_time, end_time)
            response = self.videollmOnline_Run(chunk_path, inp, question_time - start_time)  # Adjust timestamp relative to chunk
        finally:
            # Clean up temporary file
            if chunk_path and os.path.exists(chunk_path):
                os.remove(chunk_path)
        
        response_time = time_module.time() - start
        return response, response_time

    def videollmOnline_Run(self, file, inp, timestamp):
        """
        Run VideollmOnline inference on a video chunk
        """
        import cv2
        import decord
        from decord import VideoReader
        
        self.reset()
        
        # Preprocess video using decord and cv2
        name, ext = os.path.splitext(file)
        name = name.split('/')[-1]
        ffmpeg_video_path = os.path.join('./cache', name + f'_{self.frame_fps}fps_{self.frame_resolution}' + ext)
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        
        # Process video with decord and cv2
        try:
            vr = VideoReader(file, ctx=decord.cpu(0))
            original_fps = vr.get_avg_fps()
            total_frames = len(vr)
            
            # Calculate target frame indices at desired fps
            duration = total_frames / original_fps
            target_frame_count = int(duration * self.frame_fps)
            frame_indices = [int(i * original_fps / self.frame_fps) for i in range(target_frame_count)]
            frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
            
            # Extract and resize frames
            frames = vr.get_batch(frame_indices).asnumpy()
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(ffmpeg_video_path, fourcc, self.frame_fps, (self.frame_resolution, self.frame_resolution))
            
            for frame in frames:
                # Resize and pad frame
                h, w = frame.shape[:2]
                if h > w:
                    new_h = self.frame_resolution
                    new_w = int(w * new_h / h)
                else:
                    new_w = self.frame_resolution
                    new_h = int(h * new_w / w)
                
                resized = cv2.resize(frame, (new_w, new_h))
                
                # Pad to square
                pad_h = (self.frame_resolution - new_h) // 2
                pad_w = (self.frame_resolution - new_w) // 2
                padded = cv2.copyMakeBorder(resized, pad_h, self.frame_resolution - new_h - pad_h,
                                          pad_w, self.frame_resolution - new_w - pad_w,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
                
                # Convert RGB to BGR for cv2
                frame_bgr = cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            logger.warning(f'{file} -> {ffmpeg_video_path}, {self.frame_fps} FPS, {self.frame_resolution} Resolution')
            
        except Exception as e:
            logger.error(f"Video preprocessing error: {str(e)}")
            if os.path.exists(ffmpeg_video_path):
                os.remove(ffmpeg_video_path)
            raise

        # Load video and add query
        self.load_video(ffmpeg_video_path)
        print(f'Question: {inp} | Timestamp: {timestamp}s', flush=True)
        self.input_query_stream(inp, video_time=timestamp)

        # Process frames until we get a response
        for i in range(self.num_video_frames):
            self.input_video_stream(i / self.frame_fps)
            query, response = self()

            if response:
                print(f'VideollmOnline Answer: {response}', flush=True)
                logger.warning(f'VideollmOnline: {response}')
                # Clean up temporary ffmpeg file
                if os.path.exists(ffmpeg_video_path):
                    os.remove(ffmpeg_video_path)
                return response
        
        # Clean up temporary ffmpeg file
        if os.path.exists(ffmpeg_video_path):
            os.remove(ffmpeg_video_path)
        return ""
    
    @staticmethod
    def name():
        """
        Return the name of the model
        """
        return "VideollmOnline"

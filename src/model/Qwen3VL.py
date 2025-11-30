import torch
import time
import os
from PIL import Image
from decord import VideoReader, cpu
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from model.modelclass import Model


class Qwen3VL(Model):
    def __init__(self):
        self.Qwen3VL_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False):
        return self.Qwen3VL_Run(file, inp, start_time, end_time, question_time, omni, proactive)
    
    def name(self):
        return "Qwen3VL"

    def Qwen3VL_Init(self):
        print("🔧 Loading Qwen3-VL-8B-Instruct model...")
        
        # Try to load with flash_attention_2, fallback to default if not available
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            print("✅ Using Flash Attention 2")
        except Exception as e:
            print(f"⚠️ Flash Attention 2 not available, using default attention: {e}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        self.model.eval()
        
        print("✅ Qwen3-VL model loaded successfully!")

    def extract_frames(self, video_path, start_time, end_time, max_frames=16):
        """Extract frames from video between start_time and end_time"""
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            
            # Calculate frame indices
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Ensure frames are within bounds
            start_frame = max(0, min(start_frame, len(vr) - 1))
            end_frame = max(start_frame + 1, min(end_frame, len(vr)))
            
            # Sample frames uniformly
            total_frames = end_frame - start_frame
            if total_frames <= max_frames:
                frame_indices = list(range(start_frame, end_frame))
            else:
                # Uniformly sample max_frames
                step = total_frames / max_frames
                frame_indices = [int(start_frame + i * step) for i in range(max_frames)]
            
            # Extract frames
            frames = vr.get_batch(frame_indices).asnumpy()
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            return pil_frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []

    def Qwen3VL_Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False):
        try:
            T_start = time.time()
            
            # Extract frames from video
            frames = self.extract_frames(file, start_time, end_time, max_frames=16)
            
            if len(frames) == 0:
                print(f"⚠️ No frames extracted from video: {file}")
                return " ", 0
            
            print(f"📹 Extracted {len(frames)} frames from {start_time:.1f}s to {end_time:.1f}s")
            
            # Prepare messages with multiple images
            content = []
            for frame in frames:
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": inp})
            
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            
            # Prepare inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    repetition_penalty=1.0,
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            T_end = time.time()
            response_time = T_end - T_start
            
            print(f"💬 Response: {output_text}")
            print(f"⏰ Response time: {response_time:.3f}s")
            
            return output_text, response_time
            
        except Exception as e:
            print(f"❌ Error in Qwen3VL inference: {e}")
            import traceback
            traceback.print_exc()
            return " ", 0


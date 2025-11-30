import torch
import time
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from model.modelclass import Model


class Qwen25VL(Model):
    def __init__(self):
        self.Qwen25VL_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        return self.Qwen25VL_Run(file, inp, start_time, end_time, question_time, omni, proactive, salience_map_path)
    
    def name(self):
        return "Qwen2.5-VL"

    def Qwen25VL_Init(self):
        # Get model size from environment variable (default: 7B)
        model_size = os.environ.get('QWEN_MODEL_SIZE', '7B')
        model_path = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"
        
        print(f"🔧 Loading Qwen2.5-VL-{model_size}-Instruct model...")
        
        # Try to load with flash_attention_2, fallback to default if not available
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            print("✅ Using Flash Attention 2")
        except Exception as e:
            print(f"⚠️ Flash Attention 2 not available, using default attention: {e}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        
        print(f"✅ Qwen2.5-VL-{model_size}-Instruct model loaded successfully!")

    def create_video_segment(self, video_path, start_time, end_time):
        """
        Create a temporary video segment for the specified time range
        Qwen2.5-VL natively supports video input with fps control
        """
        import subprocess
        import tempfile
        
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        
        try:
            # Use ffmpeg to extract the video segment
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(end_time - start_time),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                temp_path
            ]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return temp_path
        except Exception as e:
            print(f"⚠️ Error creating video segment: {e}, falling back to full video")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    def Qwen25VL_Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        try:
            T_start = time.time()
            
            # Calculate duration and determine appropriate fps
            duration = end_time - start_time
            
            # Dynamic FPS based on duration (Qwen2.5-VL supports this!)
            if duration <= 30:
                fps = 1.0  # 30 frames for short clips
            elif duration <= 60:
                fps = 0.5  # 30 frames for 60s clips
            elif duration <= 300:
                fps = 0.2  # 60 frames for 5min clips
            else:
                fps = 0.1  # Long videos
            
            # Create video segment
            temp_video = self.create_video_segment(file, start_time, end_time)
            video_to_use = temp_video if temp_video else file
            
            print(f"📹 Processing video from {start_time:.1f}s to {end_time:.1f}s (duration: {duration:.1f}s, fps: {fps})")
            
            # Build content list with video and optional salience map
            content = [
                {
                    "type": "video",
                    "video": f"file://{video_to_use}",
                    "fps": fps,
                }
            ]
            
            # Add salience map if provided
            if salience_map_path and os.path.exists(salience_map_path):
                print(f"🖼️  Adding salience map: {os.path.basename(salience_map_path)}")
                content.append({
                    "type": "image",
                    "image": f"file://{salience_map_path}",
                })
            
            # Add text prompt
            content.append({"type": "text", "text": inp})
            
            # Prepare messages with video and optional image input
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.8,
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
            
            # Clean up temporary file
            if temp_video and os.path.exists(temp_video):
                os.remove(temp_video)
            
            T_end = time.time()
            response_time = T_end - T_start
            
            print(f"💬 Response: {output_text}")
            print(f"⏰ Response time: {response_time:.3f}s")
            
            return output_text, response_time
            
        except Exception as e:
            print(f"❌ Error in Qwen2.5-VL inference: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error
            if 'temp_video' in locals() and temp_video and os.path.exists(temp_video):
                os.remove(temp_video)
            
            return " ", 0


import time
import os
import base64
import json
import boto3
from botocore.exceptions import ClientError
from model.modelclass import Model
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import io


class ClaudeSonnet4(Model):
    def __init__(self):
        self.ClaudeSonnet4_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        return self.ClaudeSonnet4_Run(file, inp, start_time, end_time, question_time, omni, proactive, salience_map_path)
    
    def name(self):
        return "Claude-Sonnet-4"

    def ClaudeSonnet4_Init(self):
        """Initialize AWS Bedrock client for Claude Sonnet 4"""
        print("🔧 Initializing Claude Sonnet 4 with AWS Bedrock...")
        
        self.client = boto3.client('bedrock-runtime', region_name='us-east-2')
        self.model_id = 'us.anthropic.claude-sonnet-4-20250514-v1:0'
        
        print("✅ Claude Sonnet 4 client initialized successfully!")

    def extract_frames(self, video_path, start_time, end_time, max_frames=32):
        """Extract frames from video segment using decord"""
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            total_frames = end_frame - start_frame
            
            # Sample frames uniformly
            if total_frames > max_frames:
                indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)
            else:
                indices = list(range(start_frame, end_frame))
            
            frames = vr.get_batch(indices).asnumpy()
            
            print(f"📹 Extracted {len(frames)} frames from {start_time:.1f}s to {end_time:.1f}s")
            return frames
            
        except Exception as e:
            print(f"❌ Error extracting frames: {e}")
            return None

    def encode_image_base64(self, image_array):
        """Convert numpy array to base64 encoded image"""
        try:
            # Convert from RGB to PIL Image
            img = Image.fromarray(image_array.astype('uint8'), 'RGB')
            
            # Resize if too large (max 5MB for Claude)
            max_size = 1568  # Conservative size for Claude
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            
            # Encode to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_base64
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            return None

    def ClaudeSonnet4_Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        """Run Claude Sonnet 4 inference on video segment"""
        try:
            T_start = time.time()
            
            duration = end_time - start_time
            
            # Always extract 16 frames regardless of duration
            max_frames = 16
            
            print(f"📹 Processing video from {start_time:.1f}s to {end_time:.1f}s (duration: {duration:.1f}s, frames: {max_frames})")
            
            # Extract frames
            frames = self.extract_frames(file, start_time, end_time, max_frames)
            if frames is None or len(frames) == 0:
                print("❌ Failed to extract frames")
                return " ", 0
            
            # Build content list for Claude
            content = []
            
            # Add frames as base64 images
            for i, frame in enumerate(frames):
                img_base64 = self.encode_image_base64(frame)
                if img_base64:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64
                        }
                    })
            
            # Add salience map if provided
            if salience_map_path and os.path.exists(salience_map_path):
                print(f"🖼️  Adding salience map: {os.path.basename(salience_map_path)}")
                try:
                    with open(salience_map_path, 'rb') as f:
                        salience_base64 = base64.b64encode(f.read()).decode('utf-8')
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": salience_base64
                        }
                    })
                except Exception as e:
                    print(f"⚠️ Failed to load salience map: {e}")
            
            # Add text prompt
            content.append({
                "type": "text",
                "text": inp
            })
            
            # Prepare request body for Bedrock
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 2048,
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": 0.7,
                "top_p": 0.8
            }
            
            # Call AWS Bedrock API with retry logic
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(body)
                    )
                    break
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code == 'ThrottlingException':
                        wait_time = (2 ** attempt) + (attempt * 0.5)  # Exponential backoff
                        if attempt < max_retries - 1:
                            print(f"⚠️ Throttling (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"❌ Throttling exceeded after {max_retries} attempts")
                            raise
                    else:
                        wait_time = 2 ** attempt
                        if attempt < max_retries - 1:
                            print(f"⚠️ API error (attempt {attempt + 1}/{max_retries}): {e}, waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"❌ API error after {max_retries} attempts")
                            raise
            
            # Parse response
            response_body = json.loads(response['body'].read())
            output_text = response_body['content'][0]['text'].strip()
            
            # Add small delay between requests to avoid throttling
            time.sleep(0.5)
            
            T_end = time.time()
            response_time = T_end - T_start
            
            print(f"💬 Response: {output_text}")
            print(f"⏰ Response time: {response_time:.3f}s")
            
            return output_text, response_time
            
        except Exception as e:
            print(f"❌ Error in Claude Sonnet 4 inference: {e}")
            import traceback
            traceback.print_exc()
            return " ", 0



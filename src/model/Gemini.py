import time
import os
import base64
import sys
from model.modelclass import Model
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import io

try:
    from google import genai
    from google.genai import errors as genai_errors
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️ google-genai not available. Please install: pip install google-genai")


class Gemini(Model):
    def __init__(self):
        self.Gemini_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        return self.Gemini_Run(file, inp, start_time, end_time, question_time, omni, proactive, salience_map_path)
    
    def name(self):
        return "Gemini-2.0-Flash"

    def Gemini_Init(self):
        """Initialize Gemini client with GCP Vertex AI"""
        print("🔧 Initializing Gemini 2.0 Flash with GCP Vertex AI...")
        
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai package not available. Please install: pip install google-genai")
        
        YOUR_PROJECT_ID = 'your id'
        YOUR_LOCATION = 'your location'
        
        try:
            self.client = genai.Client(
                vertexai=True, 
                project=YOUR_PROJECT_ID, 
                location=YOUR_LOCATION,
            )
            self.model = "gemini-2.0-flash-exp"
            print("✅ Gemini client initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to setup Gemini client: {e}")
            raise

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
            
            # Resize if too large (max 2048 for Gemini)
            max_size = 2048
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

    def Gemini_Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        """Run Gemini inference on video segment"""
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
            
            # Build content list for Gemini
            contents = []
            
            # Add text prompt first
            contents.append(inp)
            
            # Add frames as base64 images
            for i, frame in enumerate(frames):
                img_base64 = self.encode_image_base64(frame)
                if img_base64:
                    contents.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64
                        }
                    })
            
            # Add salience map if provided
            if salience_map_path and os.path.exists(salience_map_path):
                print(f"🖼️  Adding salience map: {os.path.basename(salience_map_path)}")
                try:
                    with open(salience_map_path, 'rb') as f:
                        salience_base64 = base64.b64encode(f.read()).decode('utf-8')
                    contents.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": salience_base64
                        }
                    })
                except Exception as e:
                    print(f"⚠️ Failed to load salience map: {e}")
            
            # Call Gemini API with retry logic
            max_retries = 5
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config={
                            'temperature': 0.7,
                            'top_p': 0.8,
                            'max_output_tokens': 1536,
                        }
                    )
                    break
                except genai_errors.ClientError as e:
                    # Rate limit or quota errors
                    wait_time = (2 ** attempt) + (attempt * 0.5)  # Exponential backoff
                    if attempt < max_retries - 1:
                        print(f"⚠️ API error (attempt {attempt + 1}/{max_retries}): {e}, waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ API error after {max_retries} attempts: {e}")
                        raise
                except Exception as e:
                    wait_time = 2 ** attempt
                    if attempt < max_retries - 1:
                        print(f"⚠️ Error (attempt {attempt + 1}/{max_retries}): {e}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Error after {max_retries} attempts: {e}")
                        raise
            
            if response is None:
                print("❌ No response from Gemini API")
                return " ", 0
            
            output_text = response.text.strip()
            
            # Add delay between requests to avoid rate limiting (Vertex AI quota: 60 RPM)
            time.sleep(3.0)  # Wait 3 seconds between requests for safety margin
            
            T_end = time.time()
            response_time = T_end - T_start
            
            print(f"💬 Response: {output_text}")
            print(f"⏰ Response time: {response_time:.3f}s")
            
            return output_text, response_time
            
        except Exception as e:
            print(f"❌ Error in Gemini inference: {e}")
            import traceback
            traceback.print_exc()
            return " ", 0


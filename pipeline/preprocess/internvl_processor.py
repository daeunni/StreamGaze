import torch
import json
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from typing import List, Dict, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio

class InternVLProcessor:
    def __init__(self, model_path="OpenGVLab/InternVL3_5-38B"):
        """Initialize InternVL model"""
        print(f"🚀 Loading InternVL model: {model_path}")
        
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
        print(f"✅ InternVL model loaded successfully!")
        print(f"📍 Model device map: {self.model.hf_device_map}")
        
    def build_transform(self, input_size=448):
        """Build image transformation pipeline"""
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def load_image(self, image_file, input_size=448):
        """Load and preprocess image"""
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        else:
            image = image_file.convert('RGB')
        
        transform = self.build_transform(input_size=input_size)
        pixel_values = transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(torch.bfloat16)
        return pixel_values

    def extract_frames_from_video_clip(self, video_path: str, start_time: float, end_time: float, max_frames=8):
        """Extract frames from video clip between start_time and end_time"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        frames = []
        frame_indices = np.linspace(start_frame, end_frame, min(max_frames, end_frame - start_frame + 1), dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames

    def get_pixel_values(self, frames):
        """Convert frames to pixel values for model input"""
        pixel_values_list = []
        
        for frame in frames:
            pixel_values = self.load_image(frame, input_size=448)
            pixel_values_list.append(pixel_values)
        
        # Concatenate all frames for batch processing
        if len(pixel_values_list) > 1:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = pixel_values_list[0]
        
        # Move to appropriate device
        if hasattr(self.model, 'device'):
            pixel_values = pixel_values.to(self.model.device, dtype=torch.bfloat16)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
        
        return pixel_values

    def analyze_video_clip_with_gaze(self, video_path: str, gaze_x: float, gaze_y: float, 
                                   start_time: float, end_time: float) -> Dict:
        """Analyze video clip with gaze point using InternVL"""
        try:
            # Extract frames from video clip
            frames = self.extract_frames_from_video_clip(video_path, start_time, end_time)
            
            if not frames:
                return {"error": "No frames extracted from video clip"}
            
            pixel_values = self.get_pixel_values(frames)
            
            # Create prompt for analysis
            prompt = f"""
        Analyze this video clip and return a JSON with this exact structure:

        {{
        "scene_caption": "Brief 2-3 sentence description of the overall scene and what's happening",
        "gaze_object": {{
            "object_identity": "name of object at coordinate ({gaze_x}, {gaze_y})",
            "detailed_caption": "Natural 3-5 sentence description of this object's appearance and characteristics"
        }},
        "other_objects": [
            {{
            "object_identity": "object_name",
            "detailed_caption": "Two sentences describing the object's appearance, attributes, and spatial position relative to other objects"
            }}
        ]
        }}

        Return only valid JSON. Be concise and direct.
        """
            
            # Generate response
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    generation_config={
                        'max_new_tokens': 1024,
                        'temperature': 0.3,
                        'do_sample': True,
                        'top_p': 0.9
                    }
                )
            
            # Clean up response
            cleaned_result = response.strip()
            if "```json" in cleaned_result:
                cleaned_result = cleaned_result.split("```json")[1]
            if "```" in cleaned_result:
                cleaned_result = cleaned_result.split("```")[0]
            cleaned_result = cleaned_result.strip()
            
            # Parse JSON
            try:
                result = json.loads(cleaned_result)
                print(result)
                return result
            except json.JSONDecodeError as e:
                return {"error": f"JSON parsing failed: {e}", "raw_response": cleaned_result}
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

# Global processor instance
_processor = None

def get_processor():
    """Get or create global processor instance"""
    global _processor
    if _processor is None:
        _processor = InternVLProcessor()
    return _processor

def extract_objects_and_scene_from_video_clip_internvl(
    video_path: str,
    gaze_x: float,
    gaze_y: float,
    start_time: float,
    end_time: float,
    temperature: float = 0.3,
    **kwargs
) -> Dict:
    """
    Extract objects and scene from video clip using InternVL
    Compatible with Gemini function signature
    """
    processor = get_processor()
    return processor.analyze_video_clip_with_gaze(video_path, gaze_x, gaze_y, start_time, end_time)

def extract_objects_and_scene_from_video_clip_internvl_threaded(
    requests_data: List[Dict],
    max_workers: int = 1,
    delay_between_requests: float = 1.0,
    temperature: float = 0.3,
    **kwargs
) -> List[Dict]:
    """
    Multi-threaded processing of video clips using InternVL
    Compatible with Gemini threaded function signature
    
    Args:
        requests_data: List of dicts with keys: video_path, gaze_x, gaze_y, start_time, end_time, request_id
        max_workers: Number of parallel workers (recommended: 1 for GPU memory)
        delay_between_requests: Delay between requests in seconds
        temperature: Generation temperature (unused but kept for compatibility)
    
    Returns:
        List of analysis results in same order as input
    """
    processor = get_processor()
    results = [None] * len(requests_data)
    
    def process_single_request(request_data, index):
        """Process single request"""
        try:
            # Add delay to avoid overwhelming GPU
            if index > 0:
                time.sleep(delay_between_requests)
            
            result = processor.analyze_video_clip_with_gaze(
                video_path=request_data['video_path'],
                gaze_x=request_data['gaze_x'],
                gaze_y=request_data['gaze_y'],
                start_time=request_data['start_time'],
                end_time=request_data['end_time']
            )
            
            # Add request_id to result for tracking
            result['request_id'] = request_data.get('request_id', f'request_{index}')
            
            return index, result
            
        except Exception as e:
            error_result = {
                "error": f"Processing failed: {str(e)}",
                "request_id": request_data.get('request_id', f'request_{index}')
            }
            return index, error_result
    
    # Process requests with threading
    print(f"🚀 Processing {len(requests_data)} requests with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_request, req_data, idx): idx 
            for idx, req_data in enumerate(requests_data)
        }
        
        # Collect results with progress bar
        with tqdm(total=len(requests_data), desc="Processing fixations", unit="fixation") as pbar:
            for future in as_completed(future_to_index):
                try:
                    index, result = future.result()
                    results[index] = result
                    pbar.update(1)
                except Exception as e:
                    index = future_to_index[future]
                    results[index] = {
                        "error": f"Thread execution failed: {str(e)}",
                        "request_id": f"request_{index}"
                    }
                    pbar.update(1)
    
    print(f"✅ Completed processing {len(requests_data)} requests")
    
    # Count successes and errors
    successes = sum(1 for r in results if r and "error" not in r)
    errors = len(results) - successes
    print(f"📊 Results: {successes} successful, {errors} errors")
    
    return results

def extract_objects_and_scene_from_video_clip_internvl_v2_sequential(
    requests_data: List[Dict],
    fov_radius: int = 100,
    save_images: bool = False,
    show_images: bool = False,
    object_pool: List[str] = None,
    temperature: float = 0.3,
    output_dir: str = None,
    dataset_type: str = None,
    **kwargs
) -> List[Dict]:
    """
    Sequential processing of video clips using InternVL v2 (NO THREADING)
    Much safer for large models like 38B to avoid OOM issues
    
    Args:
        requests_data: List of dicts with keys: video_path, gaze_x, gaze_y, start_time, end_time, request_id
        fov_radius: Field of view radius for cropping/masking
        save_images: Whether to save visualization images
        show_images: Whether to show visualization images
        object_pool: List of known objects for consistency
        temperature: Generation temperature (unused but kept for compatibility)
    
    Returns:
        List of analysis results in same order as input
    """
    # Use single processor instance (no threading)
    processor = get_processor_v2()
    results = []
    
    print(f"🚀 Processing {len(requests_data)} requests sequentially (no threading)...")
    print("⚠️ This is slower but much safer for the 38B model")
    
    # Process each request one by one
    for idx, request_data in enumerate(tqdm(requests_data, desc="Processing fixations", unit="fixation")):
        try:
            result = processor.analyze_video_clip_two_stage(
                video_path=request_data['video_path'],
                gaze_x=request_data['gaze_x'],
                gaze_y=request_data['gaze_y'],
                start_time=request_data['start_time'],
                end_time=request_data['end_time'],
                fov_radius=fov_radius,
                save_images=save_images,
                show_images=show_images,
                object_pool=object_pool,
                output_dir=output_dir,
                fixation_id=request_data.get('request_id', f'request_{idx}'),
                action_caption=request_data.get('action_caption', None),
                dataset_type=dataset_type
            )
            
            # Add request_id to result for tracking
            result['request_id'] = request_data.get('request_id', f'request_{idx}')
            results.append(result)
            
            # Memory cleanup after each request
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        except Exception as e:
            error_result = {
                "error": f"Processing failed: {str(e)}",
                "request_id": request_data.get('request_id', f'request_{idx}')
            }
            results.append(error_result)
            
            # Cleanup on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    print(f"✅ Completed processing {len(requests_data)} requests")
    
    # Count successes and errors
    successes = sum(1 for r in results if r and "error" not in r)
    errors = len(results) - successes
    print(f"📊 Results: {successes} successful, {errors} errors")
    
    return results

# Utility functions for compatibility
def save_frequency_analysis(counter, output_path, video_name):
    """Save frequency analysis to file (compatible with existing function)"""
    with open(output_path, 'w') as f:
        f.write(f"OBJECT FREQUENCY ANALYSIS - {video_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("TOP 20 MOST FREQUENT OTHER OBJECTS:\n")
        f.write("-" * 40 + "\n")
        
        for obj, count in counter.most_common(20):
            f.write(f"{obj}: {count}\n")
        
        f.write(f"\nTOTAL UNIQUE OBJECTS: {len(counter)}\n")
        f.write(f"TOTAL OBJECT MENTIONS: {sum(counter.values())}\n")

def count_total_fixations(base_dir):
    """Count total fixations (compatible with existing function)"""
    tasks = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    total_fixations = 0
    video_fixations = {}
    
    for video_name in tasks:
        try:
            import pandas as pd
            csv_path = os.path.join(base_dir, video_name, f'{video_name}_fixation_dataset.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                count = len(df)
                video_fixations[video_name] = count
                total_fixations += count
        except Exception as e:
            print(f"Warning: Could not count fixations for {video_name}: {e}")
            video_fixations[video_name] = 0
    
    return total_fixations, video_fixations


class InternVLProcessor_v2:
    def __init__(self, model_path="OpenGVLab/InternVL3_5-38B", gpu_id=None):
        """Initialize InternVL model for two-stage video analysis"""
        print(f"🚀 Loading InternVL model v2: {model_path}")
        
        # Check GPU availability and assign specific GPU
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"🔥 Available GPUs: {num_gpus}")
            
            if gpu_id is not None:
                # Use specific GPU
                device_map = {"": gpu_id}
                print(f"📍 Using GPU {gpu_id}")
            else:
                # Use auto mapping (original behavior)
                device_map = "auto"
                print(f"📍 Using auto device mapping across {num_gpus} GPUs")
        else:
            print("⚠️ No GPU available, using CPU")
            device_map = "cpu"
        
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
        # Store GPU ID for processor reuse
        self.gpu_id = gpu_id
        
        print(f"✅ InternVL model v2 loaded successfully!")
        print(f"📍 Model device map: {self.model.hf_device_map}")
        
        
    def build_transform(self, input_size=448):
        """Build image transformation pipeline"""
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def extract_frames_from_video_clip(self, video_path: str, start_time: float, end_time: float, fps=1):
        """Extract frames from video clip between start_time and end_time"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame numbers
            start_frame = int(start_time * video_fps)
            end_frame = int(end_time * video_fps)
            
            # Extract frames
            frames = []
            frame_interval = max(1, int(video_fps / fps))  # Extract 1 frame per second by default
            
            for frame_num in range(start_frame, min(end_frame, total_frames), frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return []

    def crop_gaze_fov(self, frames, gaze_x, gaze_y, fov_radius):
        """Crop frames to gaze field of view area"""
        cropped_frames = []
        
        for frame in frames:
            # Get frame dimensions
            w, h = frame.size
            
            # Convert normalized coordinates to pixel coordinates
            center_x = int(gaze_x * w)
            center_y = int(gaze_y * h)
            
            # Calculate crop boundaries
            left = max(0, center_x - fov_radius)
            right = min(w, center_x + fov_radius)
            top = max(0, center_y - fov_radius)
            bottom = min(h, center_y + fov_radius)
            
            # Crop the frame
            cropped = frame.crop((left, top, right, bottom))
            cropped_frames.append(cropped)
        
        return cropped_frames

    def save_cropped_video_gif(self, cropped_frames, output_path, gaze_x, gaze_y, fps=2):
        """Save cropped frames as GIF with gaze point visualization"""
        try:
            # Create frames with gaze point visualization
            gif_frames = []
            for frame in cropped_frames:
                # Convert PIL to numpy array
                frame_array = np.array(frame)
                h, w = frame_array.shape[:2]
                
                # Convert normalized coordinates to cropped frame coordinates
                center_x = int(gaze_x * w)
                center_y = int(gaze_y * h)
                
                # Draw gaze point (bright green circle)
                cv2.circle(frame_array, (center_x, center_y), 15, (0, 255, 0), 3)  # Green circle
                cv2.circle(frame_array, (center_x, center_y), 3, (0, 255, 0), -1)  # Green dot
                
                gif_frames.append(frame_array)
            
            # Save as GIF
            imageio.mimsave(output_path, gif_frames, duration=1.0/fps, loop=0)
            print(f"✅ Cropped video GIF saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving cropped video GIF: {str(e)}")

    def save_masked_video_gif(self, masked_frames, output_path, fps=2):
        """Save masked frames as GIF"""
        try:
            # Convert PIL frames to numpy arrays
            gif_frames = []
            for frame in masked_frames:
                frame_array = np.array(frame)
                gif_frames.append(frame_array)
            
            # Save as GIF
            imageio.mimsave(output_path, gif_frames, duration=1.0/fps, loop=0)
            print(f"✅ Masked video GIF saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving masked video GIF: {str(e)}")

    def mask_gaze_fov(self, frames, gaze_x, gaze_y, fov_radius):
        """Mask the gaze field of view area in frames"""
        masked_frames = []
        
        for frame in frames:
            # Convert PIL to numpy array
            frame_array = np.array(frame)
            h, w = frame_array.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            center_x = int(gaze_x * w)
            center_y = int(gaze_y * h)
            
            # Create mask
            mask = np.ones((h, w), dtype=np.uint8) * 255
            cv2.circle(mask, (center_x, center_y), fov_radius, 0, -1)
            
            # Apply mask to all channels
            if len(frame_array.shape) == 3:
                mask_3d = np.stack([mask] * frame_array.shape[2], axis=2)
                masked_array = frame_array * (mask_3d / 255.0)
            else:
                masked_array = frame_array * (mask / 255.0)
            
            # Convert back to PIL Image
            masked_frame = Image.fromarray(masked_array.astype(np.uint8))
            masked_frames.append(masked_frame)
        
        return masked_frames

    def get_pixel_values(self, frames):
        """Convert frames to pixel values for model input"""
        pixel_values_list = []
        
        for frame in frames:
            transform = self.build_transform(input_size=448)
            pixel_values = transform(frame).unsqueeze(0)
            pixel_values = pixel_values.to(torch.bfloat16)
            pixel_values_list.append(pixel_values)
        
        # Concatenate all frames for batch processing
        if len(pixel_values_list) > 1:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = pixel_values_list[0]
        
        # Move to appropriate device - use the same device as the model
        if torch.cuda.is_available():
            # Find which GPU the model is on
            model_device = next(self.model.parameters()).device
            pixel_values = pixel_values.to(model_device, dtype=torch.bfloat16)
            print(f"📍 Moving pixel values to device: {model_device}")
        else:
            pixel_values = pixel_values.to('cpu', dtype=torch.bfloat16)
        
        return pixel_values

    def analyze_gaze_fov_cropped(self, video_path, gaze_x, gaze_y, start_time, end_time, fov_radius=100, object_pool=None, save_images=True, show_images=True, output_dir=None, fixation_id=None, action_caption=None, dataset_type=None):
        """Step 1: Analyze cropped gaze FOV area with gaze point visualization"""
        try:
            # Extract frames from video clip
            frames = self.extract_frames_from_video_clip(video_path, start_time, end_time)

            # import pdb;pdb.set_trace()
            
            if not frames:
                return {"error": "No frames extracted from video clip"}
            
            # Crop frames to gaze FOV area
            cropped_frames = self.crop_gaze_fov(frames, gaze_x, gaze_y, fov_radius)
            
            # Save cropped video GIF if output directory is provided
            if output_dir and fixation_id is not None:
                os.makedirs(output_dir, exist_ok=True)
                cropped_gif_path = os.path.join(output_dir, f'cropped_video_fixation_{fixation_id}.gif')
                self.save_cropped_video_gif(cropped_frames, cropped_gif_path, gaze_x, gaze_y, fps=2)
            
            # Visualize cropped frames with gaze point
            if show_images or save_images:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle(f'Cropped Gaze FOV Images (Center: {gaze_x:.3f}, {gaze_y:.3f})', fontsize=14)
                
                for i, frame in enumerate(cropped_frames[:8]):  # Show up to 8 frames
                    row = i // 4
                    col = i % 4
                    axes[row, col].imshow(frame)
                    
                    # Add gaze point visualization (bright green large dot)
                    # Convert normalized coordinates to cropped frame coordinates
                    w, h = frame.size
                    center_x = int(gaze_x * w)  # This will be the center of the cropped area
                    center_y = int(gaze_y * h)  # This will be the center of the cropped area
                    
                    # Draw a bright green circle for the gaze point
                    circle = Circle((center_x, center_y), 15, color='lime', alpha=0.8, linewidth=3)
                    axes[row, col].add_patch(circle)
                    axes[row, col].plot(center_x, center_y, 'o', color='lime', markersize=20, markeredgecolor='darkgreen', markeredgewidth=2)
                    
                    axes[row, col].set_title(f'Frame {i+1}')
                    axes[row, col].axis('off')
                
                # Hide unused subplots
                for i in range(len(cropped_frames), 8):
                    row = i // 4
                    col = i % 4
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                
                if save_images:
                    os.makedirs('gaze_analysis_images', exist_ok=True)
                    plt.savefig(f'gaze_analysis_images/cropped_fov_{start_time:.1f}s-{end_time:.1f}s.png', dpi=150, bbox_inches='tight')
                    print(f"Saved cropped FOV images to: gaze_analysis_images/cropped_fov_{start_time:.1f}s-{end_time:.1f}s.png")
                
                if show_images:
                    plt.show()
                else:
                    plt.close()
            
            # Get pixel values for InternVL
            pixel_values = self.get_pixel_values(cropped_frames)
            
            # Create object pool instruction
            object_pool_instruction = ""
            if object_pool:
                object_pool_str = ", ".join(object_pool)
                object_pool_instruction = f"""
IMPORTANT: If you see any of these previously identified objects in the image, please use their exact names from this list: {object_pool_str}
If you see objects not in this list, you may add new object names, but prioritize consistency with the existing object pool.
"""
            
            # Create action context instruction (skip for egoexo_kitchen)
            action_context = ""
            if action_caption and dataset_type != 'egoexo_kitchen':
                # Parse action_caption if it's a JSON list (EgoExoLearn format)
                try:
                    action_list = json.loads(action_caption) if isinstance(action_caption, str) and action_caption.startswith('[') else [action_caption]
                    if action_list:
                        action_text = "\n".join([f"- {act}" for act in action_list])
                        action_context = f"""
CONTEXT - Actions being performed during this time period:
{action_text}

Use this context to help identify relevant objects that are being used or interacted with in these actions.
"""
                except:
                    if action_caption and action_caption != 'unknown':
                        action_context = f"""
CONTEXT - Action being performed: {action_caption}

Use this context to help identify relevant objects that are being used or interacted with.
"""
            
            # Create cooking ingredient focus instruction for egoexo_kitchen
            cooking_focus = ""
            if dataset_type == 'egoexo_kitchen':
                cooking_focus = f"""
🍳 SPECIAL FOCUS FOR COOKING SCENE:
- When identifying objects, PRIORITIZE the food ingredients and cooking materials over containers
- If you see food inside a pot/bowl/pan, identify the FOOD ITSELF (e.g., "potato", "onion", "meat", "rice", "vegetables")
- Only identify containers (pot, bowl, pan) if they are empty or if the container itself is the main focus
- For exact_gaze_object: if the gaze is pointing at food in a container, name the FOOD, not the container
- Examples: 
  * Instead of "pot" → identify "soup", "stew", "boiling water", "pasta", etc.
  * Instead of "bowl" → identify "salad", "flour", "dough", "chopped vegetables", etc.
  * Instead of "pan" → identify "fried egg", "stir-fried vegetables", "grilled meat", etc.
- This is a COOKING scene, so focus on the culinary content!
"""
            
            # Create prompt for analysis with new format
            prompt = f"""
{action_context}
{cooking_focus}
Analyze this cropped video clip focused on the gaze area and return a JSON with this exact structure:

{{
  "scene_caption": "Brief 2-3 sentence description of the overall scene and what's happening",
  "exact_gaze_object": {{
    "object_identity": "single object name that the gaze point is exactly pointing at",
    "detailed_caption": "Natural 3-5 sentence description of this specific object's appearance and attributes"
  }},
  "other_objects_in_cropped_area": [
    {{
      "object_identity": "object_name",
      "detailed_caption": "Two sentences describing the object's appearance and position relative to the gaze point"
    }}
  ]
}}

{object_pool_instruction}

IMPORTANT FOR OBJECT_IDENTITY:
- For INANIMATE OBJECTS: Use ONLY simple, single-word names (e.g., "pot", "stove", "microwave")
  * NO descriptive phrases like "pot on stove", "microwave mounted"
  * NO location descriptions like "pot in kitchen"
  * NO attribute descriptions like "white pot", "large microwave"

- For PEOPLE: Use descriptive identifiers that include gender and clothing
  * Format: "man_wearing_[color]_[garment]" or "woman_wearing_[color]_[garment]"
  * Examples: "man_wearing_black_top", "woman_wearing_blue_shirt", "man_wearing_red_jacket"
  * This helps track the same person across different video segments
  * If the person matches someone from the object pool, use the EXACT same identifier

- For exact_gaze_object: identify the ONE object/person that the gaze point is directly looking at
- For other_objects_in_cropped_area: list all other objects/people visible in the cropped area

Return only valid JSON. Be concise and direct.
"""

            # import pdb;pdb.set_trace()
            
            # Generate response
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    generation_config={
                        'max_new_tokens': 1024,
                        'temperature': 0.3,
                        'do_sample': True,
                        'top_p': 0.9
                    }
                )
            
            # Clean up response
            cleaned_result = response.strip()
            if "```json" in cleaned_result:
                cleaned_result = cleaned_result.split("```json")[1]
            if "```" in cleaned_result:
                cleaned_result = cleaned_result.split("```")[0]
            cleaned_result = cleaned_result.strip()
            
            # Parse JSON
            try:
                result = json.loads(cleaned_result)
                print("Step 1 - Gaze FOV Analysis Result:", result)
                return result
            except json.JSONDecodeError as e:
                return {"error": f"JSON parsing failed: {e}", "raw_response": cleaned_result}
                
        except Exception as e:
            return {"error": f"Gaze FOV analysis failed: {str(e)}"}

    def analyze_outside_fov_masked(self, video_path, gaze_x, gaze_y, start_time, end_time, fov_radius=100, object_pool=None, gaze_objects=None, output_dir=None, fixation_id=None):
        """Step 2: Analyze masked video for outside objects"""
        try:
            # Extract frames from video clip
            frames = self.extract_frames_from_video_clip(video_path, start_time, end_time)
            
            if not frames:
                return {"error": "No frames extracted from video clip"}
            
            # Mask gaze FOV area
            masked_frames = self.mask_gaze_fov(frames, gaze_x, gaze_y, fov_radius)
            
            # Save masked video GIF if output directory is provided
            if output_dir and fixation_id is not None:
                os.makedirs(output_dir, exist_ok=True)
                masked_gif_path = os.path.join(output_dir, f'masked_video_fixation_{fixation_id}.gif')
                self.save_masked_video_gif(masked_frames, masked_gif_path, fps=2)
            
            # Get pixel values for InternVL
            pixel_values = self.get_pixel_values(masked_frames)
            
            # Create object pool instruction
            object_pool_instruction = ""
            if object_pool:
                object_pool_str = ", ".join(object_pool)
                object_pool_instruction = f"""
IMPORTANT: If you see any of these previously identified objects in the image, please use their exact names from this list: {object_pool_str}
If you see objects not in this list, you may add new object names, but prioritize consistency with the existing object pool.
"""
            
            # Create gaze objects exclusion instruction
            gaze_exclusion_instruction = ""
            if gaze_objects:
                gaze_objects_str = ", ".join(gaze_objects)
                gaze_exclusion_instruction = f"""
IMPORTANT: Do NOT include these objects that are already identified in the gaze area: {gaze_objects_str}
Focus only on objects that are clearly outside the masked center area.
"""
            
            # Create prompt for analysis
            prompt = f"""
Analyze this video clip (with the center gaze area masked out) and return a JSON with this exact structure:

{{
  "other_objects": [
    {{
      "object_identity": "object_name",
      "detailed_caption": "Two sentences describing the object's appearance, attributes, and spatial position relative to other objects"
    }}
  ]
}}

{object_pool_instruction}

{gaze_exclusion_instruction}

IMPORTANT FOR OBJECT_IDENTITY:
- For INANIMATE OBJECTS: Use ONLY simple, single-word names (e.g., "refrigerator", "cabinet", "counter")
  * NO descriptive phrases like "refrigerator in corner", "cabinet mounted"
  * NO location descriptions like "refrigerator on right"
  * NO attribute descriptions like "white refrigerator"

- For PEOPLE: Use descriptive identifiers that include gender and clothing
  * Format: "man_wearing_[color]_[garment]" or "woman_wearing_[color]_[garment]"
  * Examples: "man_wearing_black_top", "woman_wearing_blue_shirt", "man_wearing_red_jacket"
  * This helps track the same person across different video segments
  * If the person matches someone from the object pool, use the EXACT same identifier

Focus on objects/people outside the masked center area. Return only valid JSON. Be concise and direct.
"""
            
            # Generate response
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    generation_config={
                        'max_new_tokens': 1024,
                        'temperature': 0.3,
                        'do_sample': True,
                        'top_p': 0.9
                    }
                )
            
            # Clean up response
            cleaned_result = response.strip()
            if "```json" in cleaned_result:
                cleaned_result = cleaned_result.split("```json")[1]
            if "```" in cleaned_result:
                cleaned_result = cleaned_result.split("```")[0]
            cleaned_result = cleaned_result.strip()
            
            # Parse JSON
            try:
                result = json.loads(cleaned_result)
                print("Step 2 - Outside FOV Analysis Result:", result)
                
                
                return result
            except json.JSONDecodeError as e:
                return {"error": f"JSON parsing failed: {e}", "raw_response": cleaned_result}
                
        except Exception as e:
            return {"error": f"Outside FOV analysis failed: {str(e)}"}

    def visualize_original_frames_with_gaze(self, video_path, gaze_x, gaze_y, start_time, end_time, fov_radius=100, save_images=True, show_images=True):
        """Visualize original frames with gaze point and FOV circle"""
        try:
            # Extract frames from video clip
            frames = self.extract_frames_from_video_clip(video_path, start_time, end_time)
            
            if not frames:
                return {"error": "No frames extracted from video clip"}
            
            # Visualize original frames with gaze point
            if show_images or save_images:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle(f'Original Frames with Gaze Point (Center: {gaze_x:.3f}, {gaze_y:.3f})', fontsize=14)
                
                for i, frame in enumerate(frames[:8]):  # Show up to 8 frames
                    row = i // 4
                    col = i % 4
                    axes[row, col].imshow(frame)
                    
                    # Convert normalized coordinates to pixel coordinates
                    h, w = frame.size[1], frame.size[0]  # PIL Image size is (width, height)
                    center_x = int(gaze_x * w)
                    center_y = int(gaze_y * h)
                    
                    # Draw gaze point and FOV circle
                    circle = Circle((center_x, center_y), fov_radius, fill=False, color='red', linewidth=2)
                    axes[row, col].add_patch(circle)
                    axes[row, col].plot(center_x, center_y, 'ro', markersize=8)
                    
                    axes[row, col].set_title(f'Frame {i+1}')
                    axes[row, col].axis('off')
                
                # Hide unused subplots
                for i in range(len(frames), 8):
                    row = i // 4
                    col = i % 4
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                
                if save_images:
                    os.makedirs('gaze_analysis_images', exist_ok=True)
                    plt.savefig(f'gaze_analysis_images/original_with_gaze_{start_time:.1f}s-{end_time:.1f}s.png', dpi=150, bbox_inches='tight')
                    print(f"Saved original frames with gaze to: gaze_analysis_images/original_with_gaze_{start_time:.1f}s-{end_time:.1f}s.png")
                
                if show_images:
                    plt.show()
                else:
                    plt.close()
                    
        except Exception as e:
            print(f"Error visualizing original frames: {str(e)}")

    def analyze_video_clip_two_stage(self, video_path, gaze_x, gaze_y, start_time, end_time, fov_radius=100, save_images=True, show_images=True, object_pool=None, output_dir=None, fixation_id=None, action_caption=None, dataset_type=None):
        """Two-stage analysis with InternVL and new return format"""
        try:
            print(f"🎯 Starting two-stage analysis for gaze point ({gaze_x}, {gaze_y})")
            
            # Visualize original frames first
            print("📸 Visualizing original frames with gaze point...")
            self.visualize_original_frames_with_gaze(video_path, gaze_x, gaze_y, start_time, end_time, fov_radius, save_images, show_images)
            
            # Step 1: Analyze cropped gaze FOV area
            print("Step 1: Analyzing cropped gaze FOV area...")
            gaze_result = self.analyze_gaze_fov_cropped(video_path, gaze_x, gaze_y, start_time, end_time, fov_radius, object_pool, save_images, show_images, output_dir, fixation_id, action_caption, dataset_type)
            
            if "error" in gaze_result:
                return {"error": f"Step 1 failed: {gaze_result['error']}"}
            
            # Extract gaze area objects for exclusion in step 2
            gaze_objects = []
            if "exact_gaze_object" in gaze_result and "object_identity" in gaze_result["exact_gaze_object"]:
                gaze_objects.append(gaze_result["exact_gaze_object"]["object_identity"].strip())
            
            if "other_objects_in_cropped_area" in gaze_result:
                for obj in gaze_result["other_objects_in_cropped_area"]:
                    if "object_identity" in obj:
                        gaze_objects.append(obj["object_identity"].strip())
            
            # Step 2: Analyze masked video for outside objects
            print("Step 2: Analyzing masked video for outside objects...")
            outside_result = self.analyze_outside_fov_masked(video_path, gaze_x, gaze_y, start_time, end_time, fov_radius, object_pool, gaze_objects, output_dir, fixation_id)
            
            if "error" in outside_result:
                return {"error": f"Step 2 failed: {outside_result['error']}"}
            
            # Combine results with new format
            combined_result = {
                "scene_caption": gaze_result.get("scene_caption", ""),
                "exact_gaze_object": gaze_result.get("exact_gaze_object", {}),
                "other_objects_in_cropped_area": gaze_result.get("other_objects_in_cropped_area", []),
                "other_objects_outside_fov": outside_result.get("other_objects", [])
            }
            
            print("✅ Two-stage analysis completed successfully")
            return combined_result
            
        except Exception as e:
            return {"error": f"Two-stage analysis failed: {str(e)}"}

    def extract_object_names_from_result(self, result):
        """Extract all object names from analysis result (updated for new format)"""
        object_names = set()
        
        # Extract from exact_gaze_object (new format)
        if "exact_gaze_object" in result and "object_identity" in result["exact_gaze_object"]:
            object_names.add(result["exact_gaze_object"]["object_identity"].strip())
        
        # Extract from other_objects_in_cropped_area (new format)
        if "other_objects_in_cropped_area" in result:
            for obj in result["other_objects_in_cropped_area"]:
                if "object_identity" in obj:
                    object_names.add(obj["object_identity"].strip())
        
        # Extract from other_objects_outside_fov (new format)
        if "other_objects_outside_fov" in result:
            for obj in result["other_objects_outside_fov"]:
                if "object_identity" in obj:
                    object_names.add(obj["object_identity"].strip())
        
        # Legacy format support (for backward compatibility)
        if "gaze_object" in result and "object_identity" in result["gaze_object"]:
            gaze_objects = result["gaze_object"]["object_identity"].split(", ")
            object_names.update([obj.strip() for obj in gaze_objects if obj.strip()])
        
        if "other_objects" in result:
            for obj in result["other_objects"]:
                if "object_identity" in obj:
                    object_names.add(obj["object_identity"].strip())
        
        return list(object_names)

    def deduplicate_object_pool(self, object_pool):
        """Remove duplicate object names from the pool (simple deduplication)"""
        if not object_pool:
            return set()
        
        # Simple deduplication - just remove exact duplicates
        cleaned_pool = set()
        for obj in object_pool:
            if obj and obj.strip():
                cleaned_pool.add(obj.strip().lower())
        
        return cleaned_pool

    def analyze_gaze_fov_cropped_with_visualization_v2(self, video_path, gaze_x, gaze_y, start_time, end_time, fov_radius=100, save_images=True, show_images=True, object_pool=None):
        """Step 1: Analyze cropped gaze FOV area with image visualization and gaze point marking (compatible with notebook)"""
        return self.analyze_gaze_fov_cropped(video_path, gaze_x, gaze_y, start_time, end_time, fov_radius, object_pool, save_images, show_images)

    def analyze_video_clip_two_stage_with_visualization_v2(self, video_path, gaze_x, gaze_y, start_time, end_time, fov_radius=100, save_images=True, show_images=True, object_pool=None):
        """Two-stage analysis with image visualization and new return format (compatible with notebook)"""
        return self.analyze_video_clip_two_stage(video_path, gaze_x, gaze_y, start_time, end_time, fov_radius, save_images, show_images, object_pool)


# Global processor instance for v2
_processor_v2 = None

def get_processor_v2(gpu_id=None):
    """Get or create global processor v2 instance"""
    global _processor_v2
    if _processor_v2 is None:
        _processor_v2 = InternVLProcessor_v2(gpu_id=gpu_id)
    return _processor_v2

def extract_objects_and_scene_from_video_clip_internvl_v2(
    video_path: str,
    gaze_x: float,
    gaze_y: float,
    start_time: float,
    end_time: float,
    fov_radius: int = 100,
    save_images: bool = False,
    show_images: bool = False,
    object_pool: List[str] = None,
    temperature: float = 0.3,
    **kwargs
) -> Dict:
    """
    Extract objects and scene from video clip using InternVL v2 with two-stage analysis
    Compatible with existing function signature
    """
    processor = get_processor_v2()
    return processor.analyze_video_clip_two_stage(
        video_path=video_path,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        start_time=start_time,
        end_time=end_time,
        fov_radius=fov_radius,
        save_images=save_images,
        show_images=show_images,
        object_pool=object_pool
    )

def analyze_gaze_fov_cropped_with_visualization_v2(client, video_path, gaze_x, gaze_y, start_time, end_time, fov_radius=100, save_images=True, show_images=True, object_pool=None):
    """Wrapper function compatible with notebook interface"""
    processor = get_processor_v2()
    return processor.analyze_gaze_fov_cropped_with_visualization_v2(
        video_path=video_path,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        start_time=start_time,
        end_time=end_time,
        fov_radius=fov_radius,
        save_images=save_images,
        show_images=show_images,
        object_pool=object_pool
    )

def analyze_video_clip_two_stage_with_visualization_v2(client, video_path, gaze_x, gaze_y, start_time, end_time, fov_radius=100, save_images=True, show_images=True, object_pool=None):
    """Wrapper function compatible with notebook interface"""
    processor = get_processor_v2()
    return processor.analyze_video_clip_two_stage_with_visualization_v2(
        video_path=video_path,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        start_time=start_time,
        end_time=end_time,
        fov_radius=fov_radius,
        save_images=save_images,
        show_images=show_images,
        object_pool=object_pool
    )

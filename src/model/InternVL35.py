import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import time
import os
import subprocess
import tempfile
from model.modelclass import Model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class InternVL35(Model):
    def __init__(self):
        self.InternVL35_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        return self.InternVL35_Run(file, inp, start_time, end_time, question_time, omni, proactive, salience_map_path)
    
    def name(self):
        return "InternVL3.5"

    def InternVL35_Init(self):
        # Get model size from environment variable (default: 8B)
        model_size = os.environ.get('INTERNVL_MODEL_SIZE', '8B')
        
        print(f"🔧 Loading InternVL3.5-{model_size} model...")
        
        path = f'OpenGVLab/InternVL3_5-{model_size}'
        
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        
        print(f"✅ InternVL3.5-{model_size} model loaded successfully!")

    def create_video_segment(self, video_path, start_time, end_time):
        """
        Create a temporary video segment for the specified time range
        """
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

    def InternVL35_Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False, salience_map_path=None):
        try:
            T_start = time.time()
            
            # Calculate duration and determine number of segments
            duration = end_time - start_time
            
            # Adaptive num_segments based on duration
            if duration <= 30:
                num_segments = 8
            elif duration <= 60:
                num_segments = 16
            elif duration <= 300:
                num_segments = 32
            else:
                num_segments = 64
            
            print(f"📹 Processing video from {start_time:.1f}s to {end_time:.1f}s (duration: {duration:.1f}s, segments: {num_segments})")
            
            # Create video segment
            temp_video = self.create_video_segment(file, start_time, end_time)
            video_to_use = temp_video if temp_video else file
            
            # Load video frames
            if temp_video:
                # For segmented video, no bound needed
                pixel_values_video, num_patches_list_video = load_video(
                    video_to_use, 
                    bound=None, 
                    input_size=448, 
                    max_num=1, 
                    num_segments=num_segments
                )
            else:
                # For full video, use bound
                pixel_values_video, num_patches_list_video = load_video(
                    video_to_use, 
                    bound=[start_time, end_time], 
                    input_size=448, 
                    max_num=1, 
                    num_segments=num_segments
                )
            
            pixel_values_video = pixel_values_video.to(torch.bfloat16).cuda()
            
            # Prepare video prefix
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list_video))])
            
            # Add salience map if provided
            num_patches_list = num_patches_list_video.copy()
            if salience_map_path and os.path.exists(salience_map_path):
                print(f"🖼️  Adding salience map: {os.path.basename(salience_map_path)}")
                
                # Load salience map image
                pixel_values_salience = load_image(salience_map_path, input_size=448, max_num=12)
                pixel_values_salience = pixel_values_salience.to(torch.bfloat16).cuda()
                
                # Concatenate video frames and salience map
                pixel_values = torch.cat((pixel_values_video, pixel_values_salience), dim=0)
                num_patches_list.append(pixel_values_salience.shape[0])
                
                # Add salience map to prefix
                question = video_prefix + f'Salience Map: <image>\n{inp}'
            else:
                pixel_values = pixel_values_video
                question = video_prefix + inp
            
            # Generation config
            generation_config = dict(
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.8
            )
            
            # Get model response
            response = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                question, 
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )
            
            # Clean up temporary file
            if temp_video and os.path.exists(temp_video):
                os.remove(temp_video)
            
            T_end = time.time()
            response_time = T_end - T_start
            
            print(f"💬 Response: {response}")
            print(f"⏰ Response time: {response_time:.3f}s")
            
            return response, response_time
            
        except Exception as e:
            print(f"❌ Error in InternVL3.5 inference: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error
            if 'temp_video' in locals() and temp_video and os.path.exists(temp_video):
                os.remove(temp_video)
            
            return " ", 0


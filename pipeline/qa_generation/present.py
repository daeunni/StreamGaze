import pandas as pd
import numpy as np
import json, random
from pprint import pprint
import os
from datetime import datetime, timedelta
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen3 model setup
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# load the tokenizer and the model
print("Loading Qwen3 model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("Qwen3 model loaded successfully!")

def safe_json_parse(gpt_output): 
    """Strip code fences then parse JSON.""" 
    s = gpt_output.strip() 
    if s.startswith(""): 
        s = s.strip("`")
        # drop leading 'json' token if present
        if s[:4].lower().startswith("json"):
            s = s[4:].strip()
    try:
        return json.loads(s)
    except Exception:
        # fallback: try to locate the first JSON array/object
        return []

def _label_options(options_plain, correct_idx):
    labels = ["A","B","C","D","E","F","G","H"]
    labeled, answer_letter = [], None
    for i, opt in enumerate(options_plain):
        lab = labels[i]
        labeled.append(f"{lab}. {opt}")
        if i == correct_idx:
            answer_letter = lab
    return labeled, answer_letter


ATTRIBUTE_QUESTION_TEMPLATES = {
    "color": "What color is this object?",
    "material": "What material is this object made of?",
    "shape": "What shape does this object have?",
    "size": "What is the size of this object?",
    "state": "What is the state of this object?",
    "texture": "What is the texture of this object?",
    "usage": "What is this object typically used for?"
}


def build_gpt_prompt_with_templates(object_identity, detailed_caption):
    templates_str = json.dumps(ATTRIBUTE_QUESTION_TEMPLATES, indent=2)
    return f"""
You are a helpful assistant generating **one Object Attribute MCQ** grounded in an observed object.

Object identity: "{object_identity}"
Detailed caption: "{detailed_caption}"

We already provide fixed question templates for each attribute type:
{templates_str}

Your task:
1. Decide which attribute type is most natural for this object.
   Choose one from the keys of the dictionary above.

2. Select the corresponding question wording from the dictionary.
   DO NOT create your own question text.

3. From the detailed caption, extract the correct answer for that attribute.
   (Answer should be one word or short phrase, e.g., "red", "metal", "round")

4. Generate three plausible distractors that are visually reasonable but different from the correct answer.

Return strictly in this JSON format:
{{
  "attribute_type": "...",
  "question": "...",   // copy the template string for the chosen attribute
  "answer": "...",
  "distractors": ["...", "...", "..."]
}}
Only return the JSON object. No explanation.
"""

def call_qwen3_generate_mcq_with_template(object_identity, detailed_caption):
    prompt = build_gpt_prompt_with_templates(object_identity, detailed_caption)
    
    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384,
        temperature=0.3,
        do_sample=True
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return safe_json_parse(output)

def generate_present_object_attribute_MCQ_template(
    fixation_df: pd.DataFrame,
    video_path: str,
    video_category: str = "EGTEA",
    seed: int = 42
) -> List[Dict]:

    rng = random.Random(seed)
    all_tasks: List[Dict] = []

    for _, row in tqdm(fixation_df.iterrows(), total=len(fixation_df), desc="Attribute-MCQ"):
        if pd.isna(row.get("representative_object")):
            continue
        try:
            rep_obj = safe_json_parse_from_csv(row["representative_object"])
        except Exception:
            continue
        if not rep_obj:
            continue

        obj = (rep_obj.get("object_identity") or "").strip()
        caption = (rep_obj.get("detailed_caption") or "").strip()
        if not obj or not caption:
            continue

        start_sec = row.get("episode_start_time")
        end_sec = row.get("episode_end_time")
        if pd.isna(start_sec) or pd.isna(end_sec):
            continue
        start_str = _mmss_or_hhmmss(float(start_sec))
        end_str = _mmss_or_hhmmss(float(end_sec))
        time_range = f"[{start_str} - {end_str}]"

        # Get attribute + correct/wrong answers from Qwen3
        mcq_info = call_qwen3_generate_mcq_with_template(obj, caption)
        if not mcq_info:
            continue

        question_text = mcq_info.get("question", "").strip()
        correct = mcq_info.get("answer", "").strip()
        attr_type = mcq_info.get("attribute_type", "").strip().lower()
        distractors = mcq_info.get("distractors", [])

        if not question_text or not correct or not attr_type or not distractors:
            continue

        # Final options
        options_plain = [correct] + distractors
        rng.shuffle(options_plain)
        try:
            correct_idx = [o.lower() for o in options_plain].index(correct.lower())
        except ValueError:
            correct_idx = 0

        labeled_options, answer_letter = _label_options(options_plain, correct_idx)

        task = {
            "response_time": time_range,
            "questions": [{
                "task_type": "Object_Attribute",
                "question": question_text,
                "time_stamp": start_str,
                "answer": answer_letter,
                "options": labeled_options,
                "required_ability": "object_grounding",
                "attribute_type": attr_type,
                "object_name": obj
            }],
            "video_categories": video_category,
            "video_path": video_path,
        }
        all_tasks.append(task)

    return all_tasks


def safe_json_parse_from_csv(json_str):
    """
    Safely parse Python dictionary/list strings read from CSV to JSON
    """
    if pd.isna(json_str) or not json_str:
        return None
        
    try:
        # Try JSON first
        return json.loads(json_str)
    except:
        try:
            # Try JSON with single quotes replaced by double quotes
            if isinstance(json_str, str):
                json_str_fixed = json_str.replace("'", '"')
                # Remove newline characters and clean whitespace
                json_str_fixed = json_str_fixed.replace('\n', ' ').replace('\r', ' ')
                import re
                json_str_fixed = re.sub(r'\s+', ' ', json_str_fixed).strip()
                return json.loads(json_str_fixed)
        except:
            try:
                # Try ast.literal_eval for Python literals
                import ast
                return ast.literal_eval(json_str)
            except:
                print(f"⚠️ JSON 파싱 실패: {json_str[:100]}...")
                return None

# Extract unique objects from the three fields
def extract_unique_objects_from_fields(fixation_df):
    """
    Extract unique objects from representative_object, other_objects_in_cropped_area, 
    and other_objects_outside_fov fields
    """
    representative_objects = set()
    other_objects_in_cropped = set()
    other_objects_outside_fov = set()
    
    for _, row in fixation_df.iterrows():
        # Extract from representative_object
        if pd.notna(row.get("representative_object")):
            try:
                rep_obj = safe_json_parse_from_csv(row["representative_object"])
                if rep_obj and rep_obj.get("object_identity"):
                    representative_objects.add(rep_obj["object_identity"].strip())
            except Exception as e:
                print(f"Error parsing representative_object: {e}")
        
        # Extract from other_objects_in_cropped_area
        if pd.notna(row.get("other_objects_in_cropped_area")):
            try:
                other_objects = safe_json_parse_from_csv(row["other_objects_in_cropped_area"])
                if isinstance(other_objects, list):
                    for obj in other_objects:
                        if obj.get("object_identity"):
                            other_objects_in_cropped.add(obj["object_identity"].strip())
            except Exception as e:
                print(f"Error parsing other_objects_in_cropped_area: {e}")
        
        # Extract from other_objects_outside_fov
        if pd.notna(row.get("other_objects_outside_fov")):
            try:
                outside_objects = safe_json_parse_from_csv(row["other_objects_outside_fov"])
                if isinstance(outside_objects, list):
                    for obj in outside_objects:
                        if obj.get("object_identity"):
                            other_objects_outside_fov.add(obj["object_identity"].strip())
            except Exception as e:
                print(f"Error parsing other_objects_outside_fov: {e}")
    
    return {
        "representative_objects": sorted(list(representative_objects)),
        "other_objects_in_cropped_area": sorted(list(other_objects_in_cropped)),
        "other_objects_outside_fov": sorted(list(other_objects_outside_fov))
    }

# Helper Functions for Past Seen Recall

def seconds_to_timestamp_int(s):
    """Convert seconds to HH:MM:SS format"""
    s = max(0, int(round(float(s))))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

# Extract representative object names and create a new column
def extract_representative_object_names(fixation_df):
    """
    Extract only object_identity from representative_object to create a new column
    """
    object_names = []
    
    for _, row in fixation_df.iterrows():
        if pd.notna(row.get("representative_object")):
            try:
                rep_obj = safe_json_parse_from_csv(row["representative_object"])
                if rep_obj and rep_obj.get("object_identity"):
                    object_names.append(rep_obj["object_identity"].strip())
                else:
                    object_names.append(None)
            except Exception as e:
                print(f"Error parsing representative_object: {e}")
                object_names.append(None)
        else:
            object_names.append(None)
    
    return object_names

# Filter rows to keep only the longest duration for each representative_object_name
def filter_duplicate_objects_by_duration(fixation_df):
    """
    Keep only the row with the longest duration among rows with the same representative_object_name
    """
    print("Filtering duplicate objects by duration...")
    
    # Filter only rows with valid duration
    valid_rows = fixation_df[fixation_df['duration'].notna()].copy()
    print(f"Rows with valid duration: {len(valid_rows)}")
    
    # Group by representative_object_name
    object_groups = {}
    for idx, row in valid_rows.iterrows():
        obj_name = row.get('representative_object_name')
        if pd.notna(obj_name):
            if obj_name not in object_groups:
                object_groups[obj_name] = []
            object_groups[obj_name].append((idx, row))
    
    print(f"Unique objects: {len(object_groups)}")
    
    # Check duplicate objects
    duplicates = {obj: rows for obj, rows in object_groups.items() if len(rows) > 1}
    if duplicates:
        print(f"Objects with duplicates: {len(duplicates)}")
        for obj, rows in duplicates.items():
            print(f"  {obj}: {len(rows)} rows")
    
    # Select the row with the longest duration for each object
    selected_indices = []
    for object_name, rows in object_groups.items():
        if len(rows) == 1:
            # Add as-is if no duplicates
            selected_indices.append(rows[0][0])
        else:
            # Select the one with the longest duration if duplicates exist
            longest_row = max(rows, key=lambda x: x[1]['duration'])
            selected_indices.append(longest_row[0])
            
            # Print selected row's duration info
            selected_duration = longest_row[1]['duration']
            print(f"  {object_name}: Selected row with duration {selected_duration}")
    
    # Create new dataframe with selected rows
    filtered_df = fixation_df.loc[selected_indices].copy()
    
    print(f"Original rows: {len(fixation_df)}")
    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Removed: {len(fixation_df) - len(filtered_df)} duplicate rows")
    
    return filtered_df

# Updated Object Identification MCQ Generation Functions for New Metadata Structure

FALLBACK_OBJECTS = [
    "spoon","fork","knife","cup","mug","plate","bowl","pan","pot","kettle",
    "microwave","refrigerator","toaster","cutting board","sink","towel","jar","bottle"
]

def _sec_to_hhmmss(seconds: float) -> str:
    total = int(round(float(seconds)))
    hh, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def _mmss_or_hhmmss(seconds: float) -> str:
    hhmmss = _sec_to_hhmmss(seconds)
    hh, mm, ss = hhmmss.split(":")
    return f"{mm}:{ss}" if hh == "00" else hhmmss

def _dedup_ci_order(items):
    """Remove case-insensitive duplicates while preserving first occurrence order"""
    seen = set()
    out = []
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out

def _label_options_id(options_plain, correct_idx):
    labels = ["A","B","C","D","E","F","G","H"]
    labeled, answer_letter = [], None
    for i, opt in enumerate(options_plain):
        lab = labels[i]
        labeled.append(f"{lab}. {opt}")
        if i == correct_idx:
            answer_letter = lab
    return labeled, answer_letter

def _collect_all_objects_from_updated_metadata(fixation_df: pd.DataFrame) -> list:
    """Collect all unique objects from the updated metadata structure"""
    all_objects = set()
    
    for _, row in fixation_df.iterrows():
        # representative_object (single dict)
        if pd.notna(row.get("representative_object")):
            try:
                rep_obj = safe_json_parse_from_csv(row["representative_object"])
                if rep_obj and rep_obj.get("object_identity"):
                    all_objects.add(rep_obj["object_identity"].strip())
            except Exception:
                pass
        
        # other_objects_in_cropped_area (list of dicts)
        if pd.notna(row.get("other_objects_in_cropped_area")):
            try:
                cropped_objects = safe_json_parse_from_csv(row["other_objects_in_cropped_area"])
                if isinstance(cropped_objects, list):
                    for obj in cropped_objects:
                        if obj.get("object_identity"):
                            all_objects.add(obj["object_identity"].strip())
            except Exception:
                pass
        
        # other_objects_outside_fov (list of dicts)
        if pd.notna(row.get("other_objects_outside_fov")):
            try:
                outside_objects = safe_json_parse_from_csv(row["other_objects_outside_fov"])
                if isinstance(outside_objects, list):
                    for obj in outside_objects:
                        if obj.get("object_identity"):
                            all_objects.add(obj["object_identity"].strip())
            except Exception:
                pass
    
    return sorted(list(all_objects))

def _collect_cropped_area_objects(fixation_df: pd.DataFrame) -> set:
    """Collect all objects that appear in other_objects_in_cropped_area"""
    cropped_objects = set()
    
    for _, row in fixation_df.iterrows():
        if pd.notna(row.get("other_objects_in_cropped_area")):
            try:
                cropped_objects_list = safe_json_parse_from_csv(row["other_objects_in_cropped_area"])
                if isinstance(cropped_objects_list, list):
                    for obj in cropped_objects_list:
                        if obj.get("object_identity"):
                            cropped_objects.add(obj["object_identity"].strip().lower())
            except Exception:
                pass
    
    return cropped_objects



# Object Identification MCQ Generation Functions for New Metadata Structure

def generate_present_object_identify_MCQ_difficulty_updated(
    fixation_df: pd.DataFrame,
    video_path: str,
    video_category: str = "EGTEA",
    num_options: int = 4,
    seed: int = 42,
    difficulty: str = "easy"  # "easy" or "hard"
    ) -> list:

    rng = random.Random(seed)
    tasks: list = []

    # Collect all objects from the video for easy difficulty
    all_video_objects = _collect_all_objects_from_updated_metadata(fixation_df)
    all_video_objects_lower = [obj.lower() for obj in all_video_objects]
    
    # Collect objects that appear in cropped area (to avoid them in easy mode)
    cropped_area_objects = _collect_cropped_area_objects(fixation_df)

    for _, row in tqdm(fixation_df.iterrows(), total=len(fixation_df), desc=f"Identify-MCQ-{difficulty.title()}"):
        # Use representative_object instead of gaze_object
        if pd.isna(row.get("representative_object")):
            continue
        try:
            rep_obj = safe_json_parse_from_csv(row["representative_object"])
        except Exception:
            continue
        if not rep_obj:
            continue

        obj = (rep_obj.get("object_identity") or "").strip()
        caption = (rep_obj.get("detailed_caption") or "").strip()
        if not obj or not caption:
            continue

        start_sec = row.get("episode_start_time")
        end_sec = row.get("episode_end_time")
        if pd.isna(start_sec) or pd.isna(end_sec):
            continue
        start_str = _mmss_or_hhmmss(float(start_sec))
        end_str = _mmss_or_hhmmss(float(end_sec))
        time_range = f"[{start_str} - {end_str}]"

        # Question and correct answer
        question_text = "What is this?"
        correct = obj

        # Distractor selection based on difficulty
        if difficulty == "easy":
            # Easy: Use all video objects but exclude cropped area objects
            available_objects = [o for o in all_video_objects_lower 
                               if o != obj.lower() and o not in cropped_area_objects]
            rng.shuffle(available_objects)
            distractors = available_objects
            
        elif difficulty == "hard":
            # Hard: Use only objects from other_objects_outside_fov of current scene
            scene_distractors = []
            outside_fov_objects = []
            cropped_area_objects_in_scene = []
            
            # Get outside_fov objects
            if pd.notna(row.get("other_objects_outside_fov")):
                try:
                    outside_objects = safe_json_parse_from_csv(row["other_objects_outside_fov"])
                    if isinstance(outside_objects, list):
                        for o in outside_objects:
                            oid = (o.get("object_identity") or "").strip()
                            if oid:
                                outside_fov_objects.append(oid)
                                if oid.lower() != obj.lower():
                                    scene_distractors.append(oid)
                except Exception:
                    pass
            
            # Get cropped_area objects in current scene for comparison
            if pd.notna(row.get("other_objects_in_cropped_area")):
                try:
                    cropped_objects = safe_json_parse_from_csv(row["other_objects_in_cropped_area"])
                    if isinstance(cropped_objects, list):
                        for o in cropped_objects:
                            oid = (o.get("object_identity") or "").strip()
                            if oid:
                                cropped_area_objects_in_scene.append(oid)
                except Exception:
                    pass
            
            # Remove duplicates and shuffle
            scene_distractors = _dedup_ci_order([d.lower() for d in scene_distractors])
            rng.shuffle(scene_distractors)
            distractors = scene_distractors
            
            # Hard mode: If not enough outside_fov objects, skip this task
            if len(distractors) < (num_options - 1):
                print(f"⚠️ Hard mode: Not enough outside_fov objects for '{obj}' (need {num_options-1}, got {len(distractors)}).")
                print(f"   Representative object: {obj}")
                print(f"   Outside FOV objects: {outside_fov_objects}")
                print(f"   Cropped area objects: {cropped_area_objects_in_scene}")
                print(f"   Available distractors: {distractors}")
                print(f"   Overlap check: {set([o.lower() for o in outside_fov_objects]) & set([o.lower() for o in cropped_area_objects_in_scene])}")
                print("   Skipping task.")
                continue

        # Fill with fallback objects if needed (only for easy mode)
        if difficulty == "easy" and len(distractors) < (num_options - 1):
            fb = [x for x in FALLBACK_OBJECTS
                  if x.lower() != obj.lower() and x.lower() not in {d.lower() for d in distractors}]
            rng.shuffle(fb)
            distractors += fb

        distractors = _dedup_ci_order(distractors)[:(num_options - 1)]

        # Final safety net
        while len(distractors) < (num_options - 1):
            cand = rng.choice(FALLBACK_OBJECTS)
            if cand.lower() != obj.lower() and cand.lower() not in {d.lower() for d in distractors}:
                distractors.append(cand)

        # Final option composition and labeling
        options_plain = [correct] + distractors
        rng.shuffle(options_plain)
        correct_idx = options_plain.index(correct)
        labeled_options, answer_letter = _label_options_id(options_plain, correct_idx)

        tasks.append({
            "response_time": time_range,
            "questions": [{
                "task_type": "Object_Identification",
                "question": question_text,
                "time_stamp": start_str,
                "answer": answer_letter,
                "options": labeled_options,
                "required_ability": "object_grounding",
                "object": obj, 
            }],
            "video_categories": video_category,
            "video": video_path,
        })

    return tasks

def Present_object_identity_attribute(fixation_df: pd.DataFrame, video_path: str):
    '''
    Main function 
    '''
    identification_tasks_easy = generate_present_object_identify_MCQ_difficulty_updated(fixation_df, video_path, difficulty="easy")
    identification_tasks_hard = generate_present_object_identify_MCQ_difficulty_updated(fixation_df, video_path, difficulty="hard")
    attribute_tasks = generate_present_object_attribute_MCQ_template(fixation_df, video_path)
    return identification_tasks_easy, identification_tasks_hard, attribute_tasks
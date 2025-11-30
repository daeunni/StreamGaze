# StreamGaze QA Generation Pipeline

## 🎯 Overview

This pipeline processes raw gaze data from egocentric videos and generates high-quality question-answer pairs for temporal reasoning tasks (past, present, future).

## 📋 Contents

- [🎯 Overview](#-overview)
- [📦 Setup](#-setup)
  - [Supported Datasets](#supported-datasets)
  - [Environment Setup](#environment-setup)
- [📋 Pipeline Steps](#-pipeline-steps)
- [💻 Usage](#-usage)
  - [Option 1: Run Full Pipeline](#option-1-run-full-pipeline)
  - [Option 2: Run Individual Steps](#option-2-run-individual-steps)
- [📊 Output Structure](#-output-structure)

---

## 📦 Setup

### Supported Datasets

- **EGTEA-Gaze+**: https://cbs.ic.gatech.edu/fpv/
- **HoloAssist**: https://holoassist.github.io/
  - Requires 3D to 2D gaze projection using `step0_gaze_projection.py`
  - Based on [official code](https://github.com/taeinkwon/PyHoloAssist)
- **EgoExoLearn**: https://github.com/OpenGVLab/EgoExoLearn
  - Use preprocessed gaze data: [Download](https://drive.google.com/file/d/1W3blKBEe_h_aUcaJdw4ohROLp-apfE39/view)
> We provided human-verified meta data (processed step0-step2) for EGTEA/HoloAssist/EgoExoLearn in the `dataset/metadata` folder.

- **Ego4D-Gaze**: https://ego4d-data.org/docs/data/gaze/  

**If you want to add your own gaze dataset, we require:**
- Egocentric videos
- 2D frame-wise gaze coordinates on the image plane
- Camera parameters (if projection is needed)

Place datasets in: `raw_gaze_dataset/{dataset}/`

**Expected Directory Structure:**

```
raw_gaze_dataset/
├── egtea/
│   ├── videos/
│   │   ├── OP01-R01-PastaSalad.mp4
│   │   └── ...
│   ├── gaze_data/
│   │   ├── OP01-R01-PastaSalad.txt
│   │   └── ...
│   └── raw_annotations/
│       ├── action_labels.csv
│       └── cls_label_index.csv
│
├── ego4d/
│   ├── ego4d.json
│   ├── v2/
│   │   ├── gaze_videos/
│   │   └── frame_gaze_labels/
│   └── Gaze_ego4dgaze/
│
├── holoassist/
│   └── full/
│       ├── data-annnotation-trainval-v1_1.json
│       ├── R0027-12-GoPro/
│       └── ...
│
└── egoexolearn/
    ├── annotations/
    └── full/
```

### Environment Setup

```bash
conda create -n streamgaze_pipeline python=3.10
conda activate streamgaze_pipeline
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

---


## 📋 Pipeline Steps

| Step | Script | Model | Description |
|------|--------|-------|-------------|
| **0** | `step0_gaze_projection.py` | - | Gaze coordinate projection (HoloAssist only) |
| **1** | `step1_extract_fixation.py` | I-VT | Fixation extraction from gaze data |
| **1.5** | `step1.5_filtering_fixation.py` | - | Quality filtering and fixation merging |
| **2** | `step2_egtea_gaze_object_internvl.py` | InternVL-3.5 (38B) | Object identification at fixation points |
| **2.5** | `step2.5_sequence_filtering.py` | - | Sequence filtering and metadata merging |
| **3** | `step3_qa_gen.py` | - | QA pair generation for all task types |
| **4** | `step4_run_all_filtering.py` | Qwen3VL (30B) | QA validation and quality filtering |

---

## 💻 Usage

### Option 1: Run Full Pipeline

```bash
# Process entire pipeline for a dataset
bash pipeline.sh --dataset egtea

# Skip gaze visualizations (faster)
bash pipeline.sh --dataset egtea --skip-viz

# Use specific GPU
bash pipeline.sh --dataset egtea --gpu 0
```

### Option 2: Run Individual Steps

```bash
# Step 1: Extract fixations
python step1_extract_fixation.py --dataset egtea --fps 30 --no-viz

# Step 1.5: Filter fixations
python step1.5_filtering_fixation.py --dataset egtea

# Step 2: Map objects to fixations (requires GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python step2_egtea_gaze_object_internvl.py --dataset egtea

# Step 2.5: Filter sequences and merge metadata
python step2.5_sequence_filtering.py --dataset egtea

# Step 3: Generate QA pairs
python step3_qa_gen.py --dataset egtea

# Step 3 (with human-verified metadata)
python step3_qa_gen.py --dataset egtea --use-human-verified

# Step 4: Filter QA pairs
CUDA_VISIBLE_DEVICES=0 python step4_run_all_filtering.py \
  --input_dir final_data/egtea/qa_raw \
  --output_dir final_data/egtea/qa_filtered \
  --log_dir final_data/egtea/logs/filtering
```

---


## 📊 Output Structure

```
final_data/{dataset}/
├── metadata/                           # Steps 1-2 output
│   ├── {video_name}/
│   │   ├── {video_name}_fixation_dataset.csv
│   │   ├── {video_name}_fixation_filtered.csv
│   │   ├── {video_name}_fixation_merged_filtered_v2.csv
│   │   ├── {video_name}_fixation_with_internvl_v2_scene.csv
│   │   └── {video_name}_gaze_visualization.mp4
│   └── ...
│
├── total_metadata.csv                  # Step 2.5 output (merged)
│
├── qa_raw/                            # Step 3 output
│   ├── {dataset}_present_ident_tasks.json
│   ├── {dataset}_present_attr_tasks.json
│   ├── {dataset}_past_scene_reconstruction_tasks.json
│   ├── {dataset}_past_transition_pattern_tasks.json
│   ├── {dataset}_future_action_tasks.json
│   ├── {dataset}_future_remind_easy_tasks.json
│   └── task_summary.json
│
├── qa_filtered/                       # Step 4 output
│   ├── {dataset}_present_ident_tasks.json
│   ├── {dataset}_present_attr_tasks.json
│   ├── filtering_summary.json
│   └── logs/
│       └── *.log
│
└── logs/                              # Execution logs
    ├── step1_extract_fixation.log
    ├── step2_gaze_object_internvl.log
    ├── step3_qa_gen.log
    └── ...
```


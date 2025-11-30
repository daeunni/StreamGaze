# StreamGaze Model Integration Guide

This guide walks you through how to add a new model to the StreamGaze benchmark for evaluation.

## Overview

To properly evaluate a model on StreamGaze, you need to:
1. Create a model wrapper class that implements the `modelclass` interface
2. Register your model in `src/eval.py`
3. Create a bash script in `scripts/` to run your model

---

## Directory Structure

```
StreamGaze/
├── src/
│   ├── eval.py                      # Main evaluation script
│   ├── model/
│   │   ├── modelclass.py           # Base model interface
│   │   ├── GPT4o.py                # Example: GPT-4o implementation
│   │   ├── ViSpeak.py              # Example: ViSpeak implementation
│   │   ├── Qwen25VL.py             # Example: Qwen2.5-VL implementation
│   │   └── YourModel.py            # ← Your model goes here
│   ├── benchmark/
│   │   ├── StreamingBenchGaze_StreamGaze.py         # Present tasks
│   │   ├── StreamingBenchGaze_Past_StreamGaze.py    # Past tasks
│   │   └── StreamingBenchRemind_StreamGaze.py       # Proactive tasks
│   └── utils/
├── scripts/
│   ├── gpt4o.sh                    # Example: GPT-4o script
│   ├── vispeak.sh                  # Example: ViSpeak script
│   └── yourmodel.sh                # ← Your script goes here
├── dataset/
│   ├── qa/                         # QA task files (.json)
│   └── videos/
│       ├── original_video/         # Original videos
│       └── gaze_viz_video/         # Videos with gaze visualization
├── results/
│   └── YourModel/                  # Results will be saved here
└── evaluate_results.py             # Automatic evaluation script
```

---

## Step 1: Create Model Wrapper

### 1.1 Create Your Model File

```bash
cd /path/to/StreamGaze
touch src/model/YourModel.py
```

**Recommended:** Use `src/model/ViSpeak.py` or `src/model/Qwen25VL.py` as reference implementations.

### 1.2 Implement the `modelclass` Interface

All models must subclass `model.modelclass.Model` and implement three methods:

```python
from model.modelclass import Model

class YourModel(Model):
    def __init__(self):
        """
        Initialize your model
        - Load model weights
        - Load tokenizer/processor
        - Set device
        """
        pass
    
    def Run(self, file, inp, start_time=None, end_time=None, question_time=None, 
            omni=False, proactive=False, salience_map_path=None):
        """
        Run inference on a video clip
        
        Args:
            file (str): Path to the video file
                       - With --use_gaze_instruction: gaze_viz_video/xxx.mp4 (with overlay)
                       - Without: original_video/xxx.mp4 (no overlay)
            inp (str): Input prompt/question
                      - With --use_gaze_instruction: includes gaze explanation
                      - Without: just the question
            start_time (float): Start time of the video clip (seconds)
            end_time (float): End time of the video clip (seconds)
            question_time (float): Time when the question is asked (seconds)
            omni (bool): Whether this is an omni-temporal task
            proactive (bool): Whether this is a proactive task (requires timing)
            salience_map_path (str): Path to salience map for attention-based models
                                    (optional visual prompting, not commonly used)
        
        Returns:
            str: Model's response
        """
        # Your inference logic here
        return "Your model's response"
    
    def name(self):
        """
        Return the name of your model
        
        Returns:
            str: Model name (used for logging and results)
        """
        return "YourModel"
```

### 1.3 Example Implementation

```python
from model.modelclass import Model
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

class YourModel(Model):
    def __init__(self):
        print("🔧 Loading YourModel...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            "your/model/path",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("your/model/path")
        self.model.eval()
        print("✅ YourModel loaded successfully!")
    
    def Run(self, file, inp, start_time=None, end_time=None, question_time=None,
            omni=False, proactive=False, salience_map_path=None):
        # 1. Load and process video
        video_frames = self.load_video(file, start_time, end_time)
        
        # 2. Prepare inputs
        inputs = self.processor(
            text=inp,
            videos=video_frames,
            return_tensors="pt"
        ).to(self.model.device)
        
        # 3. Generate response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        
        # 4. Decode and return
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def name(self):
        return "YourModel"
    
    def load_video(self, file, start_time, end_time):
        # Your video loading logic
        # Return: List of video frames or tensor
        pass
```

---

## Step 2: Register Model in `eval.py`

Add your model to `src/eval.py`:

```python
####### MODEL ############

model = Model()

# ... existing models ...

elif args.model_name == "YourModel":
    from model.YourModel import YourModel
    model = YourModel()

######################
```

Location: Around line 30-105 in `src/eval.py`

---

## Step 3: Create Evaluation Script

Create a bash script in `scripts/` to run your model:

```bash
cd scripts
touch yourmodel.sh
chmod +x yourmodel.sh
```

### 3.1 Basic Script Template

```bash
#!/bin/bash

# ===== VISUAL PROMPTING SUPPORT =====
# Parse command line arguments for gaze instruction
USE_GAZE_VIZ=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_gaze_instruction)
            USE_GAZE_VIZ=true  # Enable gaze visualization
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--use_gaze_instruction]"
            exit 1
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

# ===== MODEL CONFIGURATION =====
MODEL_NAME="YourModel"
# ===============================

WORKDIR="$ROOT_DIR/results/$MODEL_NAME"

# ===== VIDEO & RESULTS PATH SELECTION =====
# Automatically switch between original and gaze-visualization videos
if [ "$USE_GAZE_VIZ" = "true" ]; then
    LOGDIR="$WORKDIR/logs_viz"
    RESULTS_DIR="$WORKDIR/results_viz"
    VIDEO_ROOT="$ROOT_DIR/dataset/videos/gaze_viz_video"  # With green dot + red circle
    echo "🎯 Using GAZE VISUALIZATION videos"
else
    LOGDIR="$WORKDIR/logs"
    RESULTS_DIR="$WORKDIR/results"
    VIDEO_ROOT="$ROOT_DIR/dataset/videos/original_video"  # Original videos
    echo "📹 Using ORIGINAL videos (no gaze overlay)"
fi

mkdir -p "$LOGDIR"
mkdir -p "$RESULTS_DIR"

QA_DIR="$ROOT_DIR/dataset/qa"
GAZE_VIZ_VIDEO_ROOT="$ROOT_DIR/dataset/videos/gaze_viz_video"

echo "🔧 Using model: $MODEL_NAME"
echo "📁 Results will be saved to: $RESULTS_DIR"

# Define tasks
PAST_TASKS=(
    "past_scene_recall.json"
    "past_object_transition_prediction.json"
    "past_gaze_sequence_matching.json"
    "past_non_fixated_object_identification.json"
)

PRESENT_TASKS=(
    "present_object_identification_easy.json"
    "present_object_identification_hard.json"
    "present_object_attribute_recognition.json"
)

FUTURE_TASKS=(
    "present_future_action_prediction.json"
)

REMIND_TASKS=(
    "proactive_gaze_triggered_alert.json"
    "proactive_object_appearance_alert.json"
)

PIDS=()

# Run Past Tasks
echo "=== Past Tasks ==="
for TASK_FILE in "${PAST_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    echo "  Launching: $TASK_NAME"
    
    # ===== GAZE INSTRUCTION ARGUMENTS =====
    # --use_gaze_instruction: Adds gaze explanation to prompts
    # --gaze_viz_video_root: Path to videos with gaze overlay
    # These are conditionally added based on USE_GAZE_VIZ flag
    python src/eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchGaze_Past_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        ${USE_GAZE_VIZ:+--use_gaze_instruction} \
        ${USE_GAZE_VIZ:+--gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT"} \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    
    PIDS+=($!)
done

# Run Present Tasks
echo "=== Present Tasks ==="
for TASK_FILE in "${PRESENT_TASKS[@]}" "${FUTURE_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    echo "  Launching: $TASK_NAME"
    
    python src/eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchGaze_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        ${USE_GAZE_VIZ:+--use_gaze_instruction} \
        ${USE_GAZE_VIZ:+--gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT"} \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    
    PIDS+=($!)
done

# Run Proactive (Remind) Tasks
echo "=== Proactive Tasks ==="
for TASK_FILE in "${REMIND_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    echo "  Launching: $TASK_NAME"
    
    python src/eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchRemind_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        ${USE_GAZE_VIZ:+--use_gaze_instruction} \
        ${USE_GAZE_VIZ:+--gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT"} \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    
    PIDS+=($!)
done

# Wait for all tasks
echo ""
echo "========================================="
echo "Waiting for all tasks to complete..."
echo "========================================="

FAILED=0
ALL_TASKS=("${PAST_TASKS[@]}" "${PRESENT_TASKS[@]}" "${FUTURE_TASKS[@]}" "${REMIND_TASKS[@]}")

for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    STATUS=$?
    TASK_FILE="${ALL_TASKS[$i]}"
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    
    if [ $STATUS -eq 0 ]; then
        echo "  ✓ Completed: $TASK_NAME"
    else
        echo "  ✗ Failed: $TASK_NAME"
        FAILED=1
    fi
done

echo ""
echo "========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All evaluations completed successfully!"
else
    echo "✗ Some evaluations failed."
fi
echo "Results saved to: $RESULTS_DIR/"
echo "========================================="

# ===== AUTOMATIC EVALUATION =====
echo ""
echo "========================================="
echo "🔬 Running Automatic Evaluation..."
echo "========================================="

python "$ROOT_DIR/evaluate_results.py" "$RESULTS_DIR" "$MODEL_NAME"

echo ""
echo "========================================="
echo "✅ All tasks completed!"
echo "========================================="
```

---

## Step 4: Run Your Model

### 4.1 Basic Usage

```bash
cd /path/to/StreamGaze/scripts

# Run with original videos
bash yourmodel.sh

# Run with gaze visualization
bash yourmodel.sh --use_gaze_instruction
```

### 4.2 What Happens

1. **Task Execution**: All 10 tasks run in parallel
   - 4 Past tasks (memory/recall)
   - 4 Present tasks (perception/reasoning)
   - 2 Proactive tasks (anticipation/alerting)

2. **Results Saved**: Output JSON files saved to `results/YourModel/results/`
   - Format: `{task_name}_output.json`

3. **Automatic Evaluation**: Metrics calculated automatically
   - **MC Tasks**: Accuracy (A/B/C/D)
   - **Proactive Tasks**: Precision, Recall, F1

4. **Summary**: Detailed metrics printed and saved to `evaluation_summary.json`

---

## Visual Prompting & Gaze Instruction

StreamGaze supports **visual prompting** through gaze visualization, which significantly improves model performance by providing explicit visual cues about user attention.

### What is Visual Prompting?

Visual prompting overlays gaze information directly onto the video frames:
- 🟢 **Green dot**: Current gaze point (where the user is looking)
- 🔴 **Red circle**: Field of View (FOV) region (~15° visual angle, perifovea)

This allows models to:
1. **Understand attention** - Know exactly what the user is looking at
2. **Track gaze patterns** - Follow user attention over time
3. **Identify gazed objects** - Accurately determine fixation targets
4. **Anticipate actions** - Predict based on gaze behavior

### Usage

Enable gaze instruction with the `--use_gaze_instruction` flag:

```bash
# Without gaze instruction (original videos)
bash scripts/yourmodel.sh

# With gaze instruction (visualization overlays)
bash scripts/yourmodel.sh --use_gaze_instruction
```

### How It Works

#### 1. Video Selection

```bash
if [ "$USE_GAZE_VIZ" = "true" ]; then
    VIDEO_ROOT="$ROOT_DIR/dataset/videos/gaze_viz_video"
    RESULTS_DIR="$WORKDIR/results_viz"
else
    VIDEO_ROOT="$ROOT_DIR/dataset/videos/original_video"
    RESULTS_DIR="$WORKDIR/results"
fi
```

#### 2. Prompt Augmentation

When `--use_gaze_instruction` is enabled, benchmarks automatically add instruction to prompts:

```python
if use_gaze_instruction:
    gaze_instruction = (
        "\n\n[Gaze Information]\n"
        "In this video, you will see visual indicators:\n"
        "- 🟢 Green dot: Current gaze point (where the user is looking)\n"
        "- 🔴 Red circle: Field of View (FOV) region\n"
        "Use this information to understand user attention and focus."
    )
    prompt = original_prompt + gaze_instruction
```

#### 3. Video Paths

The benchmark automatically uses the correct video path:

```python
# In benchmark class
if self.use_gaze_instruction and args.gaze_viz_video_root:
    video_path = os.path.join(args.gaze_viz_video_root, video_name)
else:
    video_path = os.path.join(args.video_root, video_name)
```

### Implementation in Your Model

Your model **doesn't need special handling** for gaze instruction! The framework handles everything:

```python
class YourModel(Model):
    def Run(self, file, inp, start_time=None, end_time=None, ...):
        # 'file' will automatically be:
        # - original_video/xxx.mp4 (without gaze instruction)
        # - gaze_viz_video/xxx.mp4 (with gaze instruction)
        
        # 'inp' will automatically include gaze instruction text
        # if --use_gaze_instruction is used
        
        # Just process as normal!
        video_frames = self.load_video(file, start_time, end_time)
        response = self.generate(video_frames, inp)
        return response
```

### Directory Structure

```
dataset/
├── videos/
│   ├── original_video/           # Original videos (no overlay)
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── gaze_viz_video/           # Videos with gaze visualization
│       ├── video1.mp4            # Same video + green dot + red circle
│       ├── video2.mp4
│       └── ...
└── qa/
    └── *.json                     # QA data (same for both)

results/
├── YourModel/
│   ├── results/                   # Results without gaze instruction
│   │   └── *_output.json
│   └── results_viz/               # Results with gaze instruction
│       └── *_output.json
```



### Common Questions

**Q: Do I need to modify my model for gaze instruction?**
A: No! The framework handles video selection and prompt augmentation automatically.

**Q: Can I use custom gaze visualization?**
A: Yes, provide your own videos in `dataset/videos/gaze_viz_video/` with the same filenames.

**Q: Does gaze instruction help all tasks?**
A: It helps most, especially:
- ✅ Present tasks (identifying gazed objects)
- ✅ Proactive tasks (detecting when user gazes at target)
- ⚠️ Past tasks (may not help as much, as it shows current gaze)

**Q: How do I know if my model is using gaze instruction correctly?**
A: Check the logs - you should see "Using gaze visualization videos from: ..." message.

---


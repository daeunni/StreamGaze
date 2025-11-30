#!/bin/bash
## --> offline model evaluation 
# Parse command line arguments
USE_GAZE_VIZ=false  # Default: use original videos
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_gaze_instruction)
            USE_GAZE_VIZ=true
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
export PYTHONPATH="$ROOT_DIR"
export DECORD_EOF_RETRY_MAX=40960  # Increase EOF retry limit for problematic videos
# ===== MODEL CONFIGURATION =====
MODEL_NAME="Qwen25VL"  # Using Qwen2.5-VL-32B-Instruct
EVAL_NAME="Qwen2.5-VL-32B-Instruct"
export QWEN_MODEL_SIZE="32B"  # Set model size via environment variable
# ===============================
WORKDIR="$ROOT_DIR/results/Qwen25VL"
if [ "$USE_GAZE_VIZ" = "true" ]; then
    LOGDIR="$WORKDIR/logs_viz"
else
    LOGDIR="$WORKDIR/logs"
fi
if [ "$USE_GAZE_VIZ" = "true" ]; then
    RESULTS_DIR="$WORKDIR/results_viz"
else
    RESULTS_DIR="$WORKDIR/results"
fi
mkdir -p "$LOGDIR"
mkdir -p "$RESULTS_DIR"
echo "🔧 Using model: $MODEL_NAME ($QWEN_MODEL_SIZE)"
# StreamingGaze dataset paths - Using _StreamGaze_eval data
QA_DIR="$ROOT_DIR/dataset/qa"
VIDEO_ROOT="$ROOT_DIR/dataset/videos/original_video" 
GAZE_VIZ_VIDEO_ROOT="$ROOT_DIR/dataset/videos/gaze_viz_video"  # Gaze visualization videos
cd ../src
# Past tasks: Use full video [0, timestamp]
PAST_TASKS=(
    "past_scene_recall.json"
    "past_object_transition_prediction.json"
    "past_gaze_sequence_matching.json"
    "past_non_fixated_object_identification.json"
)
# Present tasks: Use 60-second window [timestamp-60, timestamp]
PRESENT_TASKS=(
    "present_object_attribute_recognition.json"
    "present_object_identification_easy.json"
    "present_object_identification_hard.json"
)
# Future tasks: Use 60-second window [timestamp-60, timestamp]
FUTURE_TASKS=(
    "present_future_action_prediction.json"
)
# Remind tasks: Use OVO-Bench style (0 ~ realtime, multiple test points)
REMIND_TASKS=(
    "proactive_gaze_triggered_alert.json"
    "proactive_object_appearance_alert.json"
)
# Launch all tasks in parallel - 3 GPUs
GPU_ARRAY=(0 2 3)  # Use GPUs 0, 2, 3
GPU_IDX=0
echo ""
echo "=== Past Tasks (Full Video: 0 ~ timestamp) ==="
PIDS=()
for TASK_FILE in "${PAST_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    GPU_ID=${GPU_ARRAY[$GPU_IDX]}
    echo "  [GPU $GPU_ID] Launching: $TASK_NAME"
    CUDA_VISIBLE_DEVICES=$GPU_ID QWEN_MODEL_SIZE=$QWEN_MODEL_SIZE python eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchGaze_Past_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        --use_gaze_instruction \
        --gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT" \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    PIDS+=($!)
    echo "    → PID: $! | Log: $LOGDIR/${TASK_NAME}.log"
    GPU_IDX=$(( (GPU_IDX + 1) % 3 ))
done
echo ""
echo "=== Present Tasks (60s Window: timestamp-60 ~ timestamp) ==="
for TASK_FILE in "${PRESENT_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    GPU_ID=${GPU_ARRAY[$GPU_IDX]}
    echo "  [GPU $GPU_ID] Launching: $TASK_NAME"
    CUDA_VISIBLE_DEVICES=$GPU_ID QWEN_MODEL_SIZE=$QWEN_MODEL_SIZE python eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchGaze_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        --use_gaze_instruction \
        --gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT" \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    PIDS+=($!)
    echo "    → PID: $! | Log: $LOGDIR/${TASK_NAME}.log"
    GPU_IDX=$(( (GPU_IDX + 1) % 3 ))
done
echo ""
echo "=== Future Tasks (60s Window: timestamp-60 ~ timestamp) ==="
for TASK_FILE in "${FUTURE_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    GPU_ID=${GPU_ARRAY[$GPU_IDX]}
    echo "  [GPU $GPU_ID] Launching: $TASK_NAME"
    CUDA_VISIBLE_DEVICES=$GPU_ID QWEN_MODEL_SIZE=$QWEN_MODEL_SIZE python eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchGaze_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        --use_gaze_instruction \
        --gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT" \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    PIDS+=($!)
    echo "    → PID: $! | Log: $LOGDIR/${TASK_NAME}.log"
    GPU_IDX=$(( (GPU_IDX + 1) % 3 ))
done
# Wait for all tasks to complete
echo ""
echo "========================================="
echo "Waiting for all tasks to complete..."
echo "Total tasks: $((${#PAST_TASKS[@]} + ${#PRESENT_TASKS[@]} + ${#FUTURE_TASKS[@]}))"
echo "Monitor logs: tail -f $LOGDIR/*.log"
echo "========================================="
echo ""
for PID in "${PIDS[@]}"; do
    wait $PID
done
# Remind tasks are commented out above, but if needed, they would be processed here
# All tasks completed
echo ""
echo "========================================="
echo "✓ All tasks completed!"
echo "  - Past tasks: ${#PAST_TASKS[@]}"
echo "  - Present tasks: ${#PRESENT_TASKS[@]}"
echo "  - Future tasks: ${#FUTURE_TASKS[@]}"
echo "========================================="
echo "Results saved to: $RESULTS_DIR/"
echo "Logs saved to: $LOGDIR/"
echo "========================================="

# ===== AUTOMATIC EVALUATION =====
echo ""
echo "========================================="
echo "🔬 Running Automatic Evaluation..."
echo "========================================="

# Run evaluation script
python "$ROOT_DIR/evaluate_results.py" "$RESULTS_DIR" "$MODEL_NAME"

echo ""
echo "========================================="
echo "✅ All tasks completed!"
echo "========================================="



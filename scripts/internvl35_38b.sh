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
MODEL_NAME="InternVL35"  # Using InternVL3.5-38B
EVAL_NAME="InternVL3.5-38B"
export INTERNVL_MODEL_SIZE="38B"  # Set model size via environment variable
# ===============================
WORKDIR="$ROOT_DIR/results/InternVL35"
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
echo "🔧 Using model: $MODEL_NAME ($INTERNVL_MODEL_SIZE)"
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
# Sequential execution: Process 4 tasks at a time with 4 GPUs
GPU_ARRAY=(0 1 2 3)  # Use GPUs 0, 1, 2, 3
ALL_TASKS=("${PAST_TASKS[@]}" "${PRESENT_TASKS[@]}" "${FUTURE_TASKS[@]}" "${REMIND_TASKS[@]}")
TASK_TYPES=()
for task in "${PAST_TASKS[@]}"; do TASK_TYPES+=("StreamingBenchGaze_Past_StreamGaze"); done
for task in "${PRESENT_TASKS[@]}"; do TASK_TYPES+=("StreamingBenchGaze_StreamGaze"); done
for task in "${FUTURE_TASKS[@]}"; do TASK_TYPES+=("StreamingBenchGaze_StreamGaze"); done
for task in "${REMIND_TASKS[@]}"; do TASK_TYPES+=("StreamingBenchRemind_StreamGaze"); done
TOTAL_TASKS=${#ALL_TASKS[@]}
BATCH_SIZE=4
echo ""
echo "========================================="
echo "Sequential Execution: Processing 4 tasks at a time"
echo "Total tasks: $TOTAL_TASKS"
echo "Using GPUs: ${GPU_ARRAY[@]}"
echo "========================================="
for ((i=0; i<$TOTAL_TASKS; i+=BATCH_SIZE)); do
    PIDS=()
    BATCH_NUM=$((i/BATCH_SIZE + 1))
    echo ""
    echo "=== Batch $BATCH_NUM: Tasks $((i+1))-$((i+BATCH_SIZE > TOTAL_TASKS ? TOTAL_TASKS : i+BATCH_SIZE)) ==="
    for ((j=0; j<BATCH_SIZE && i+j<TOTAL_TASKS; j++)); do
        TASK_FILE="${ALL_TASKS[$((i+j))]}"
        TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
        BENCHMARK_NAME="${TASK_TYPES[$((i+j))]}"
        GPU_ID=${GPU_ARRAY[$j]}
        echo "  [GPU $GPU_ID] Launching: $TASK_NAME"
        CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
            --model_name $MODEL_NAME \
            --benchmark_name $BENCHMARK_NAME \
            --data_file "$QA_DIR/$TASK_FILE" \
            --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
            --video_root "$VIDEO_ROOT" \
        --use_gaze_instruction \
        --gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT" \
            > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
        PIDS+=($!)
        echo "    → PID: $! | Log: $LOGDIR/${TASK_NAME}.log"
    done
    # Wait for this batch to complete
    echo "  Waiting for batch $BATCH_NUM to complete..."
    for pid in "${PIDS[@]}"; do
        wait $pid
    done
    echo "  ✓ Batch $BATCH_NUM completed"
done
# Summary
echo ""
echo "========================================="
echo "✓ All evaluations completed!"
echo "  - Past tasks: ${#PAST_TASKS[@]}"
echo "  - Present tasks: ${#PRESENT_TASKS[@]}"
echo "  - Future tasks: ${#FUTURE_TASKS[@]}"
echo "  - Remind tasks: ${#REMIND_TASKS[@]}"
echo "  - Total tasks: $TOTAL_TASKS"
echo "Results saved to: $RESULTS_DIR/"
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



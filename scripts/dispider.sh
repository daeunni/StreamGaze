#!/bin/bash
# Dispider evaluation script for StreamingGaze
# GPU memory constraint: Max 2 processes per GPU to avoid OOM
# Activate Dispider conda environment
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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dispider
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR"
export DECORD_EOF_RETRY_MAX=40960  # Increase EOF retry limit for problematic videos
# Optional: Set HF_HOME and TRANSFORMERS_CACHE if needed
# export HF_HOME="path/to/cache"
# export TRANSFORMERS_CACHE="path/to/cache"
# ===== MODEL CONFIGURATION =====
MODEL_NAME="Dispider"
EVAL_NAME="Dispider"
# ===============================
WORKDIR="$ROOT_DIR/results/Dispider"
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
echo "🔧 Using model: $MODEL_NAME"
# StreamingGaze dataset paths - Using _StreamGaze_eval data
QA_DIR="$ROOT_DIR/dataset/qa"
VIDEO_ROOT="$ROOT_DIR/dataset/videos/original_video"
GAZE_VIZ_VIDEO_ROOT="$ROOT_DIR/dataset/videos/gaze_viz_video"  # Gaze visualization videos
cd $ROOT_DIR/StreamingBench/src
# Define all tasks
declare -A TASKS
TASKS["past_scene_reconstruction"]="StreamingBenchGaze_Past_StreamGaze:vispeak_past_scene_reconstruction.json"
TASKS["past_transition_pattern"]="StreamingBenchGaze_Past_StreamGaze:vispeak_past_transition_pattern.json"
TASKS["past_next_after_group"]="StreamingBenchGaze_Past_StreamGaze:vispeak_past_next_after_group.json"
TASKS["past_never_gazed"]="StreamingBenchGaze_Past_StreamGaze:vispeak_past_never_gazed.json"
TASKS["present_attr"]="StreamingBenchGaze_StreamGaze:vispeak_present_attr.json"
TASKS["present_ident"]="StreamingBenchGaze_StreamGaze:vispeak_present_ident.json"
TASKS["present_ident_hard"]="StreamingBenchGaze_StreamGaze:vispeak_present_ident_hard.json"
TASKS["future_action"]="StreamingBenchGaze_StreamGaze:vispeak_future_action.json"
TASKS["future_remind_easy"]="StreamingBenchRemind_StreamGaze:vispeak_future_remind_easy.json"
TASKS["future_remind_hard"]="StreamingBenchRemind_StreamGaze:vispeak_future_remind_hard.json"
# Task order
TASK_ORDER=(
    "past_scene_reconstruction"
    "past_transition_pattern"
    "past_next_after_group"
    "past_never_gazed"
    "present_attr"
    "present_ident"
    "present_ident_hard"
    "future_action"
    "future_remind_easy"
    "future_remind_hard"
)
# GPU allocation (4 GPUs, max 2 processes each = 8 concurrent)
GPU_ARRAY=(4 5 6 7)
echo ""
echo "========================================="
echo "🚀 Starting Dispider Evaluation"
echo "Total tasks: ${#TASK_ORDER[@]}"
echo "GPUs: ${GPU_ARRAY[@]} (max 2 processes per GPU)"
echo "========================================="
# BATCH 1: First 8 tasks (GPU 4, 5, 6, 7 with 2 each)
echo ""
echo "=== BATCH 1: First 8 tasks (2 per GPU) ==="
PIDS_BATCH1=()
TASKS_BATCH1=()
for i in {0..7}; do
    TASK_NAME="${TASK_ORDER[$i]}"
    TASK_INFO="${TASKS[$TASK_NAME]}"
    BENCHMARK=$(echo "$TASK_INFO" | cut -d':' -f1)
    DATA_FILE=$(echo "$TASK_INFO" | cut -d':' -f2)
    GPU_ID=${GPU_ARRAY[$((i % 4))]}
    echo "  [GPU $GPU_ID] Launching: $TASK_NAME"
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name $BENCHMARK \
        --data_file "$QA_DIR/$DATA_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        --use_gaze_instruction \
        --gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT" \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    PIDS_BATCH1+=($!)
    TASKS_BATCH1+=("$TASK_NAME")
    echo "    → PID: $! | Log: $LOGDIR/${TASK_NAME}.log"
done
echo ""
echo "Waiting for Batch 1 to complete..."
FAILED_BATCH1=0
for i in "${!PIDS_BATCH1[@]}"; do
    wait ${PIDS_BATCH1[$i]}
    STATUS=$?
    TASK_NAME="${TASKS_BATCH1[$i]}"
    if [ $STATUS -eq 0 ]; then
        echo "  ✓ Completed: $TASK_NAME"
    else
        echo "  ✗ Failed: $TASK_NAME (exit code: $STATUS)"
        echo "    → Check log: $LOGDIR/${TASK_NAME}.log"
        FAILED_BATCH1=1
    fi
done
# BATCH 2: Last 2 tasks
echo ""
echo "=== BATCH 2: Last 2 tasks ==="
PIDS_BATCH2=()
TASKS_BATCH2=()
for i in {8..9}; do
    TASK_NAME="${TASK_ORDER[$i]}"
    TASK_INFO="${TASKS[$TASK_NAME]}"
    BENCHMARK=$(echo "$TASK_INFO" | cut -d':' -f1)
    DATA_FILE=$(echo "$TASK_INFO" | cut -d':' -f2)
    GPU_ID=${GPU_ARRAY[$((i % 4))]}
    echo "  [GPU $GPU_ID] Launching: $TASK_NAME"
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name $BENCHMARK \
        --data_file "$QA_DIR/$DATA_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        --use_gaze_instruction \
        --gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT" \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    PIDS_BATCH2+=($!)
    TASKS_BATCH2+=("$TASK_NAME")
    echo "    → PID: $! | Log: $LOGDIR/${TASK_NAME}.log"
done
echo ""
echo "Waiting for Batch 2 to complete..."
FAILED_BATCH2=0
for i in "${!PIDS_BATCH2[@]}"; do
    wait ${PIDS_BATCH2[$i]}
    STATUS=$?
    TASK_NAME="${TASKS_BATCH2[$i]}"
    if [ $STATUS -eq 0 ]; then
        echo "  ✓ Completed: $TASK_NAME"
    else
        echo "  ✗ Failed: $TASK_NAME (exit code: $STATUS)"
        echo "    → Check log: $LOGDIR/${TASK_NAME}.log"
        FAILED_BATCH2=1
    fi
done
# Final summary
echo ""
echo "========================================="
if [ $FAILED_BATCH1 -eq 0 ] && [ $FAILED_BATCH2 -eq 0 ]; then
    echo "✓ All Dispider evaluations completed successfully!"
    echo "  Total tasks: ${#TASK_ORDER[@]}"
else
    echo "✗ Some evaluations failed. Check logs above."
    [ $FAILED_BATCH1 -ne 0 ] && echo "  - Batch 1 had failures"
    [ $FAILED_BATCH2 -ne 0 ] && echo "  - Batch 2 had failures"
fi
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



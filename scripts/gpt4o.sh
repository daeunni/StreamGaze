#!/bin/bash

## --> GPT-4o model evaluation via Azure OpenAI
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
MODEL_NAME="GPT4o"  # Using GPT-4o via Azure OpenAI
EVAL_NAME="GPT4o"
# ===============================

WORKDIR="$ROOT_DIR/results/GPT4o"
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

# Launch tasks with concurrency limit (2 at a time to avoid rate limiting)
PIDS=()
MAX_PARALLEL=2

echo ""
echo "=== Past Tasks (Full Video: 0 ~ timestamp) ==="
for TASK_FILE in "${PAST_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    
    # Wait if we already have MAX_PARALLEL tasks running
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 2
    done
    
    echo "  Launching: $TASK_NAME"
    
    python eval.py \
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
done

echo ""
echo "=== Present Tasks (60s Window: timestamp-60 ~ timestamp) ==="
for TASK_FILE in "${PRESENT_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    
    # Wait if we already have MAX_PARALLEL tasks running
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 2
    done
    
    echo "  Launching: $TASK_NAME"
    
    python eval.py \
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
done

echo ""
echo "=== Future Tasks (60s Window: timestamp-60 ~ timestamp) ==="
for TASK_FILE in "${FUTURE_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    
    # Wait if we already have MAX_PARALLEL tasks running
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 2
    done
    
    echo "  Launching: $TASK_NAME"
    
    python eval.py \
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
done

echo ""
echo "=== Remind Tasks (OVO-Bench style: 0 ~ realtime, multiple test points) ==="
for TASK_FILE in "${REMIND_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    
    # Wait if we already have MAX_PARALLEL tasks running
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 2
    done
    
    echo "  Launching: $TASK_NAME"
    
    python eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchRemind_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        --use_gaze_instruction \
        --gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT" \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    
    PIDS+=($!)
    echo "    → PID: $! | Log: $LOGDIR/${TASK_NAME}.log"
done

# Wait for all tasks to complete
echo ""
echo "========================================="
echo "Waiting for all tasks to complete..."
echo "Total tasks: $((${#PAST_TASKS[@]} + ${#PRESENT_TASKS[@]} + ${#FUTURE_TASKS[@]} + ${#REMIND_TASKS[@]}))"
echo "Monitor logs: tail -f $LOGDIR/*.log"
echo "========================================="
echo ""

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
        echo "  ✗ Failed: $TASK_NAME (exit code: $STATUS)"
        echo "    → Check log: $LOGDIR/${TASK_NAME}.log"
        FAILED=1
    fi
done

echo ""
echo "========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All evaluations completed successfully!"
    echo "  - Past tasks: ${#PAST_TASKS[@]}"
    echo "  - Present tasks: ${#PRESENT_TASKS[@]}"
    echo "  - Future tasks: ${#FUTURE_TASKS[@]}"
    echo "  - Remind tasks: ${#REMIND_TASKS[@]}"
else
    echo "✗ Some evaluations failed. Check logs above."
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




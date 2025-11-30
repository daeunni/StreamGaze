#!/bin/bash

################################################################################
# StreamGaze Pipeline - Complete Execution Script
################################################################################
# This script runs the entire StreamGaze QA generation pipeline from raw data
# to filtered QA JSON files.
#
# Pipeline Steps:
#   Step 0: Gaze Projection (convert raw gaze to video coordinates)
#   Step 1: Extract Fixation (identify fixation points from gaze data)
#   Step 1.5: Filter Fixations (remove invalid fixations)
#   Step 2: Gaze-Object Mapping (identify objects at fixation points)
#   Step 2.5: Sequence Filtering (filter object sequences)
#   Step 3: QA Generation (generate question-answer pairs)
#   Step 4: QA Filtering (apply task-specific filters with Qwen3VL)
#
# Usage:
#   bash pipeline.sh [options]
#
# Options:
#   --dataset DATASET     Dataset to process (egtea|ego4d|egoexo|holoassist)
#   --start-step N        Start from step N (default: 0)
#   --end-step N          End at step N (default: 4)
#   --skip-viz            Skip visualization generation (faster)
#   --gpu GPU_ID          GPU to use (default: 0)
#
# Examples:
#   bash pipeline.sh --dataset egtea                    # Full pipeline for EGTEA
#   bash pipeline.sh --dataset egtea --start-step 3     # Only QA generation onwards
#   bash pipeline.sh --dataset egtea --skip-viz         # Skip visualizations
################################################################################

set -e  # Exit on error

# Default configuration
DATASET="egtea"
START_STEP=0
END_STEP=4
SKIP_VIZ=""
GPU_ID=0
FPS=30

# Pipeline directory (relative to script location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PIPELINE_DIR="$SCRIPT_DIR"
cd "$PIPELINE_DIR"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --start-step)
            START_STEP="$2"
            shift 2
            ;;
        --end-step)
            END_STEP="$2"
            shift 2
            ;;
        --skip-viz)
            SKIP_VIZ="--no-viz"
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --help)
            head -n 35 "$0" | tail -n +3 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set dataset-specific FPS
case $DATASET in
    holoassist)
        FPS=24.46
        ;;
    *)
        FPS=30
        ;;
esac

# Output directories (relative to pipeline directory)
OUTPUT_BASE="${PIPELINE_DIR}/final_data"
METADATA_DIR="${OUTPUT_BASE}/${DATASET}/metadata"
QA_RAW_DIR="${OUTPUT_BASE}/${DATASET}/qa_raw"
QA_FILTERED_DIR="${OUTPUT_BASE}/${DATASET}/qa_filtered"
LOGS_DIR="${OUTPUT_BASE}/${DATASET}/logs"

# Create directories
mkdir -p "$METADATA_DIR"
mkdir -p "$QA_RAW_DIR"
mkdir -p "$QA_FILTERED_DIR"
mkdir -p "$LOGS_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          StreamGaze QA Generation Pipeline                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Dataset:        $DATASET"
echo "  Start Step:     $START_STEP"
echo "  End Step:       $END_STEP"
echo "  FPS:            $FPS"
echo "  Skip Viz:       ${SKIP_VIZ:-No}"
echo "  GPU:            $GPU_ID"
echo "  Pipeline Dir:   $PIPELINE_DIR"
echo "  Output Dir:     $OUTPUT_BASE/$DATASET"
echo "  Logs Dir:       $LOGS_DIR"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Export GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

################################################################################
# Step 0: Gaze Projection (HoloAssist only)
################################################################################
if [ $START_STEP -le 0 ] && [ $END_STEP -ge 0 ]; then
    if [ "$DATASET" = "holoassist" ]; then
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║  Step 0: Gaze Projection (HoloAssist)                         ║"
        echo "╚════════════════════════════════════════════════════════════════╝"
        echo "Converting raw 3D gaze data to 2D video coordinates..."
        echo ""
        
        LOG_FILE="$LOGS_DIR/step0_gaze_projection.log"
        HOLOASSIST_BASE_DIR="${PIPELINE_DIR}/raw_gaze_dataset/holoassist/full"
        
        python step0_gaze_projection.py \
            --base_dir "$HOLOASSIST_BASE_DIR" \
            --no-video \
            2>&1 | tee "$LOG_FILE"
        
        echo ""
        echo "✓ Step 0 completed!"
        echo "  Log: $LOG_FILE"
        echo ""
    else
        echo "ℹ️  Step 0: Skipped (not required for $DATASET)"
        echo ""
    fi
fi

################################################################################
# Step 1: Extract Fixation
################################################################################
if [ $START_STEP -le 1 ] && [ $END_STEP -ge 1 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Step 1: Extract Fixation                                      ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo "Identifying fixation points from gaze data..."
    echo ""
    
    LOG_FILE="$LOGS_DIR/step1_extract_fixation.log"
    
    python step1_extract_fixation.py \
        --dataset "$DATASET" \
        --fps "$FPS" \
        $SKIP_VIZ \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Step 1 completed!"
    echo "  Output: $METADATA_DIR"
    echo "  Log: $LOG_FILE"
    echo ""
fi

################################################################################
# Step 1.5: Filter Fixations
################################################################################
if [ $START_STEP -le 1 ] && [ $END_STEP -ge 1 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Step 1.5: Filter Fixations                                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo "Filtering invalid fixations..."
    echo ""
    
    LOG_FILE="$LOGS_DIR/step1.5_filtering_fixation.log"
    
    python step1.5_filtering_fixation.py \
        --dataset "$DATASET" \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Step 1.5 completed!"
    echo "  Log: $LOG_FILE"
    echo ""
fi

################################################################################
# Step 2: Gaze-Object Mapping
################################################################################
if [ $START_STEP -le 2 ] && [ $END_STEP -ge 2 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Step 2: Gaze-Object Mapping                                   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo "Identifying objects at fixation points using InternVL..."
    echo ""
    
    LOG_FILE="$LOGS_DIR/step2_gaze_object_internvl.log"
    
    if [ "$DATASET" = "egtea" ]; then
        python step2_egtea_gaze_object_internvl.py \
            2>&1 | tee "$LOG_FILE"
    else
        echo "⚠️  Step 2 script for $DATASET not implemented yet"
        echo "    Using step2_egtea_gaze_object_internvl.py as template"
        python step2_egtea_gaze_object_internvl.py \
            --dataset "$DATASET" \
            2>&1 | tee "$LOG_FILE"
    fi
    
    echo ""
    echo "✓ Step 2 completed!"
    echo "  Log: $LOG_FILE"
    echo ""
fi

################################################################################
# Step 2.5: Sequence Filtering
################################################################################
if [ $START_STEP -le 2 ] && [ $END_STEP -ge 2 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Step 2.5: Sequence Filtering                                  ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo "Filtering object sequences for quality..."
    echo ""
    
    LOG_FILE="$LOGS_DIR/step2.5_sequence_filtering.log"
    
    python step2.5_sequence_filtering.py \
        --dataset "$DATASET" \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Step 2.5 completed!"
    echo "  Log: $LOG_FILE"
    echo ""
fi

################################################################################
# Step 3: QA Generation
################################################################################
if [ $START_STEP -le 3 ] && [ $END_STEP -ge 3 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Step 3: QA Generation                                         ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo "Generating question-answer pairs..."
    echo ""
    
    LOG_FILE="$LOGS_DIR/step3_qa_gen.log"
    
    python step3_qa_gen.py \
        --dataset "$DATASET" \
        --output_dir "$QA_RAW_DIR" \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Step 3 completed!"
    echo "  Output: $QA_RAW_DIR"
    echo "  Log: $LOG_FILE"
    echo ""
fi

################################################################################
# Step 4: QA Filtering
################################################################################
if [ $START_STEP -le 4 ] && [ $END_STEP -ge 4 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Step 4: QA Filtering (Qwen3VL-30B)                            ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo "Applying task-specific filters with Qwen3VL-30B..."
    echo ""
    
    LOG_FILE="$LOGS_DIR/step4_run_all_filtering.log"
    
    python step4_run_all_filtering.py \
        --input_dir "$QA_RAW_DIR" \
        --output_dir "$QA_FILTERED_DIR" \
        --log_dir "$LOGS_DIR/filtering" \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "✓ Step 4 completed!"
    echo "  Output: $QA_FILTERED_DIR"
    echo "  Log: $LOG_FILE"
    echo ""
fi

################################################################################
# Pipeline Complete
################################################################################
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Pipeline Complete!                                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  Dataset:           $DATASET"
echo "  Steps Executed:    $START_STEP → $END_STEP"
echo "  Raw QA:            $QA_RAW_DIR"
echo "  Filtered QA:       $QA_FILTERED_DIR"
echo "  Logs:              $LOGS_DIR"
echo ""
echo "Next Steps:"
echo "  1. Review filtering logs in: $LOGS_DIR/filtering/"
echo "  2. Check filtering summary: $QA_FILTERED_DIR/filtering_summary.json"
echo "  3. Validate QA files in: $QA_FILTERED_DIR/"
echo ""
echo "════════════════════════════════════════════════════════════════"


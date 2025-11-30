# StreamGaze QA Filtering Pipeline

This package contains filtering functions for various QA task types generated from egocentric videos with gaze data.

## 📦 Filter Functions

Each filter function has the same signature:

```python
def filter_XXX(data: List[dict], log_file: Optional[TextIO] = None) -> Tuple[List[dict], dict]:
    """
    Args:
        data: List of QA items (JSON format)
        log_file: Optional file handle for logging
    
    Returns:
        filtered_data: List of filtered QA items
        stats: Dictionary with filtering statistics
    """
```

### Available Filters

| Task Type | Filter Function | Description |
|-----------|----------------|-------------|
| **Future Tasks** |
| `future_action` | `filter_future_action` | Filters future action prediction tasks |
| `future_remind_hard` | `filter_future_remind_hard` | Filters hard remind tasks |
| `future_remind_easy` | `filter_future_remind_easy` | Filters easy remind tasks with Qwen3VL verification |
| **Present Tasks** |
| `present_attr` | `filter_present_attr` | Filters object attribute recognition tasks |
| `present_ident` | `filter_present_ident` | Filters object identification tasks |
| **Past Tasks** |
| `past_next_after_group` | `filter_past_next_after_group` | Filters gaze sequence tasks |
| `past_scene_reconstruction` | `filter_past_scene_reconstruction` | Filters scene recall tasks |
| `past_transition_pattern` | `filter_past_transition_pattern` | Filters object transition tasks |

## 🚀 Usage

### Option 1: Run All Filters

```bash
# Basic usage
python run_all_filtering.py \
    --input_dir /path/to/qa_jsons \
    --output_dir /path/to/filtered_output

# With custom log directory
python run_all_filtering.py \
    --input_dir /path/to/qa_jsons \
    --output_dir /path/to/filtered_output \
    --log_dir /path/to/logs

# With file pattern
python run_all_filtering.py \
    --input_dir /path/to/qa_jsons \
    --output_dir /path/to/filtered_output \
    --file_pattern "*_qa.json"
```

### Option 2: Use Individual Filter

```python
import json
from filtering import filter_future_action

# Load QA data
with open('future_action_qa.json', 'r') as f:
    data = json.load(f)

# Apply filter
with open('filtering.log', 'w') as log:
    filtered_data, stats = filter_future_action(data, log)

# Save filtered data
with open('filtered_output.json', 'w') as f:
    json.dump(filtered_data, f, indent=2)

# Print statistics
print(stats)
```

## 📊 Input/Output Format

### Input Format (QA JSON)

```json
[
  {
    "video": "/path/to/video.mp4",
    "video_path": "/path/to/video.mp4",
    "response_time": "[00:30 - 01:30]",
    "questions": [
      {
        "question": "What is the user looking at?",
        "options": ["A. cup", "B. plate", "C. spoon", "D. bowl"],
        "answer": "A",
        ...
      }
    ]
  },
  ...
]
```

### Output Format

Same structure as input, but with filtered items.

### Statistics Output

```python
{
    'initial': 1000,                      # Initial item count
    'filtered_reason1': 150,              # Count filtered by reason 1
    'filtered_reason2': 50,               # Count filtered by reason 2
    'final': 800                          # Final item count
}
```

## 🔧 Filtering Criteria

### Future Action
- Time gap validation (3s < gap ≤ 60s)
- Context-answer overlap check
- Generic context filtering
- Option polishing with Qwen3VL

### Future Remind Easy
- Gaze verification with Qwen3VL
- Frame-level validation

### Future Remind Hard
- Ambiguous object filtering

### Present Attribute
- Zero duration clip filtering
- Human-related object filtering
- Ambiguous attribute type filtering
- Confusing option pair replacement with Qwen3VL

### Present Identification
- Zero duration clip filtering
- Human-related object filtering
- Similar object pair resolution

### Past Next After Group
- Human object filtering
- Ambiguous option replacement with Qwen3VL
- Similar object handling

### Past Scene Reconstruction
- Short clip filtering (< 5s)
- Human object filtering
- Gazed object in options check
- Qwen3VL validation

### Past Transition Pattern
- Consecutive identical group filtering
- Short sequence filtering (< 2 groups)
- Qwen3VL validation

## 📝 Log Files

Each filter generates a log file with:
- Filtered items and reasons
- Detailed filtering decisions
- Statistics summary

Example log:
```
Filtering Log: future_action_qa.json
Task Type: future_action
Timestamp: 2024-11-24 10:30:00
================================================================================

[FILTERED - TIME TOO SHORT] video1.mp4: 2.5s
[FILTERED - GENERIC CONTEXT] video2.mp4
[POLISHED OPTIONS] video3.mp4

================================================================================
Filtering Statistics:
================================================================================
initial: 1000
filtered_time_gap_too_short: 150
filtered_generic_context: 50
polished_options: 30
final: 800
```

## ⚙️ Requirements

- Python 3.8+
- transformers (for Qwen3VL)
- torch
- opencv-python (for video frame extraction)
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🤖 Model Dependencies

Some filters use **Qwen3VL-30B** for validation:
- `filter_future_action` - option polishing
- `filter_future_remind_easy` - gaze verification
- `filter_past_next_after_group` - replacement suggestion
- `filter_past_scene_reconstruction` - validation
- `filter_past_transition_pattern` - validation
- `filter_present_attr` - replacement suggestion

The model is loaded once in `utils.py` and shared across all filters.

## 📂 Output Structure

```
output_dir/
├── logs/
│   ├── future_action_qa.log
│   ├── present_attr_qa.log
│   └── ...
├── future_action_qa.json          # Filtered QA files
├── present_attr_qa.json
├── ...
└── filtering_summary.json         # Overall summary
```

## 💡 Tips

1. **Memory Usage**: Qwen3VL-30B requires significant GPU memory (~30GB)
2. **Speed**: Filters with Qwen3VL validation are slower (API calls)
3. **Logs**: Always check logs to understand what was filtered and why
4. **Customization**: Modify filter criteria in individual filter files


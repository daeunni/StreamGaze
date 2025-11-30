import json
import os
import pandas as pd
import re

def normalize_answer(answer):
    """Extract and normalize answer choice (A, B, C, D) from response."""
    if answer is None:
        return ""
    
    answer_str = str(answer).strip().upper()
    
    # Try to extract just the letter choice (A, B, C, or D)
    # Patterns: "A", "A.", "A. something", "Answer: A", etc.
    match = re.search(r'\b([ABCD])\b', answer_str)
    if match:
        return match.group(1)
    
    return answer_str

def calculate_proactive_accuracy(data):
    """Calculate accuracy for proactive monitoring tasks."""
    total = 0
    correct = 0
    
    for entry in data:
        questions = entry.get('questions', [])
        
        for question in questions:
            test_info = question.get('test_info', [])
            
            for test in test_info:
                test_type = test.get('type')
                response = test.get('response', '')
                
                # For proactive tasks, just check if response contains "yes" or "no"
                response_normalized = str(response).strip().upper()
                
                if test_type is not None:
                    total += 1
                    expected_answer = "YES" if test_type == 1 else "NO"
                    
                    if expected_answer in response_normalized:
                        correct += 1
    
    if total == 0:
        return 0.0, 0, 0
    return (correct / total) * 100, correct, total

def calculate_accuracy(json_file_path):
    """Calculate accuracy for a given result file."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Check if this is a proactive task
        is_proactive = False
        if data and len(data) > 0:
            first_entry = data[0]
            if 'questions' in first_entry and len(first_entry['questions']) > 0:
                if 'test_info' in first_entry['questions'][0]:
                    is_proactive = True
        
        if is_proactive:
            return calculate_proactive_accuracy(data)
        
        total = 0
        correct = 0
        
        for entry in data:
            questions = entry.get('questions', [])
            predictions = entry.get('model_predictions', [])
            
            for q, p in zip(questions, predictions):
                correct_answer = normalize_answer(q.get('answer', ''))
                model_prediction = normalize_answer(p.get('model_prediction', ''))
                
                if correct_answer:
                    total += 1
                    if correct_answer == model_prediction:
                        correct += 1
        
        if total == 0:
            return 0.0, 0, 0
        return (correct / total) * 100, correct, total
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return 0.0, 0, 0

# Define models and base paths
base_models = ['GPT4o', 'InternVL35', 'Qwen25VL', 'ViSpeak']
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# Create model-directory combinations (only if directory exists)
model_configs = []
for model in base_models:
    results_path = os.path.join(base_path, model, "results")
    results_viz_path = os.path.join(base_path, model, "results_viz")
    
    if os.path.exists(results_path):
        model_configs.append((f"{model}_results", model, "results"))
    if os.path.exists(results_viz_path):
        model_configs.append((f"{model}_results_viz", model, "results_viz"))

# Define tasks with readable names
tasks = {
    'past_gaze_sequence_matching': 'Gaze Sequence Matching',
    'past_non_fixated_object_identification': 'Non-fixated Object ID',
    'past_object_transition_prediction': 'Object Transition Prediction',
    'past_scene_recall': 'Scene Recall',
    'present_future_action_prediction': 'Future Action Prediction',
    'present_object_attribute_recognition': 'Object Attribute Recognition',
    'present_object_identification_easy': 'Object ID (Easy)',
    'present_object_identification_hard': 'Object ID (Hard)',
    'proactive_gaze_triggered_alert': 'Gaze-Triggered Alert',
    'proactive_object_appearance_alert': 'Object Appearance Alert'
}

# Collect detailed results
detailed_results = []

for model_name, base_model, result_dir in model_configs:
    for task_key, task_name in tasks.items():
        result_path = os.path.join(base_path, base_model, result_dir, f'{task_key}_output.json')
        
        if os.path.exists(result_path):
            accuracy, correct, total = calculate_accuracy(result_path)
            detailed_results.append({
                'Model': model_name,
                'Task': task_name,
                'Accuracy (%)': f"{accuracy:.2f}",
                'Correct': correct,
                'Total': total
            })
        else:
            detailed_results.append({
                'Model': model_name,
                'Task': task_name,
                'Accuracy (%)': "N/A",
                'Correct': 0,
                'Total': 0
            })

# Create detailed DataFrame
detailed_df = pd.DataFrame(detailed_results)

# Pivot for summary table
summary_data = []
for model_name, base_model, result_dir in model_configs:
    row = {'Model': model_name}
    for task_key, task_name in tasks.items():
        result_path = os.path.join(base_path, base_model, result_dir, f'{task_key}_output.json')
        
        if os.path.exists(result_path):
            accuracy, _, _ = calculate_accuracy(result_path)
            row[task_name] = accuracy
        else:
            row[task_name] = None
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)

# Calculate averages
avg_data = []
for model_name, base_model, result_dir in model_configs:
    accuracies = []
    for task_key in tasks.keys():
        result_path = os.path.join(base_path, base_model, result_dir, f'{task_key}_output.json')
        
        if os.path.exists(result_path):
            accuracy, _, _ = calculate_accuracy(result_path)
            accuracies.append(accuracy)
    
    avg = sum(accuracies) / len(accuracies) if accuracies else 0
    avg_data.append({'Model': model_name, 'Average': avg})

avg_df = pd.DataFrame(avg_data)

# Save to CSV files
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
detailed_df.to_csv(os.path.join(output_dir, 'evaluation_detailed.csv'), index=False)
summary_df.to_csv(os.path.join(output_dir, 'evaluation_summary.csv'), index=False)
avg_df.to_csv(os.path.join(output_dir, 'evaluation_averages.csv'), index=False)

# Create markdown table manually
with open(os.path.join(output_dir, 'evaluation_results.md'), 'w') as f:
    f.write("# StreamGaze Evaluation Results\n\n")
    f.write("## Comparison: results vs results_viz\n\n")
    f.write("**Note**: Answer extraction improved to handle different formats (e.g., 'A', 'A.', 'A. object')\n\n")
    
    # Summary table
    f.write("## Summary Table (Accuracy %)\n\n")
    f.write("| Model | " + " | ".join(tasks.values()) + " |\n")
    f.write("|" + "---|" * (len(tasks) + 1) + "\n")
    for _, row in summary_df.iterrows():
        model = row['Model']
        values = [f"{row[task]:.2f}" if pd.notna(row[task]) else "N/A" for task in tasks.values()]
        f.write(f"| {model} | " + " | ".join(values) + " |\n")
    f.write("\n\n")
    
    # Average accuracy
    f.write("## Average Accuracy by Model\n\n")
    f.write("| Model | Average |\n")
    f.write("|---|---|\n")
    for _, row in avg_df.iterrows():
        f.write(f"| {row['Model']} | {row['Average']:.2f}% |\n")
    f.write("\n\n")
    
    # Detailed results
    f.write("## Detailed Results\n\n")
    f.write("| Model | Task | Accuracy (%) | Correct | Total |\n")
    f.write("|---|---|---|---|---|\n")
    for _, row in detailed_df.iterrows():
        f.write(f"| {row['Model']} | {row['Task']} | {row['Accuracy (%)']} | {row['Correct']} | {row['Total']} |\n")
    f.write("\n")

print("\n" + "="*150)
print("EVALUATION RESULTS - results vs results_viz")
print("="*150)
print("\nSummary Table:")
print(summary_df.to_string(index=False))

print("\n\n" + "="*80)
print("AVERAGE ACCURACY BY MODEL")
print("="*80)
print(avg_df.to_string(index=False))
print("="*80)

print("\n\nResults saved to:")
print(f"  - {os.path.join(output_dir, 'evaluation_detailed.csv')}")
print(f"  - {os.path.join(output_dir, 'evaluation_summary.csv')}")
print(f"  - {os.path.join(output_dir, 'evaluation_averages.csv')}")
print(f"  - {os.path.join(output_dir, 'evaluation_results.md')}")


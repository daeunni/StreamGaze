#!/usr/bin/env python3
"""
Generate HTML files for human annotation of QA tasks
"""

import json
import os
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

def time_to_seconds(time_str):
    """Convert MM:SS to seconds"""
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    return 0

def get_video_name_from_path(video_path):
    """Extract video name from path (e.g., OP01-R01-PastaSalad)"""
    return Path(video_path).stem

def generate_html_template(task_name, questions_data, task_type):
    """Generate HTML for QA annotation"""
    
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QA Human Annotation - {task_name}</title>
    <style>
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f5f5;
    color: #333;
    line-height: 1.6;
}}

.container {{
    max-width: 1600px;
    margin: 0 auto;
    padding: 20px;
}}

header {{
    text-align: center;
    margin-bottom: 30px;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}}

header h1 {{
    font-size: 2.5em;
    margin-bottom: 10px;
}}

header p {{
    font-size: 1.2em;
    opacity: 0.9;
}}

.navigation {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 20px;
    margin-bottom: 30px;
    padding: 15px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}}

.navigation button {{
    padding: 10px 20px;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
}}

.navigation button:hover {{
    background: #5a6fd8;
}}

.navigation button:disabled {{
    background: #ccc;
    cursor: not-allowed;
}}

#questionInfo {{
    font-weight: bold;
    font-size: 18px;
}}

.progress-bar {{
    width: 100%;
    height: 30px;
    background: #e0e0e0;
    border-radius: 15px;
    overflow: hidden;
    margin: 10px 0;
}}

.progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    transition: width 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}}

.content {{
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 30px;
}}

.video-section {{
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    position: sticky;
    top: 20px;
    align-self: flex-start;
}}

.video-section h2 {{
    margin-bottom: 15px;
    color: #667eea;
}}

video {{
    width: 100%;
    border-radius: 8px;
    background: #000;
}}

.video-info {{
    margin-top: 15px;
    padding: 15px;
    background: #f8f9fa;
    border-radius: 8px;
}}

.video-info p {{
    margin: 5px 0;
}}

.question-section {{
    background: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}}

.question-section h2 {{
    margin-bottom: 20px;
    color: #667eea;
    font-size: 1.8em;
}}

.question-text {{
    font-size: 1.3em;
    font-weight: 600;
    margin-bottom: 25px;
    padding: 20px;
    background: #f8f9fa;
    border-left: 5px solid #667eea;
    border-radius: 5px;
}}

.options {{
    display: flex;
    flex-direction: column;
    gap: 15px;
    margin-bottom: 30px;
}}

.option-button {{
    padding: 20px;
    background: white;
    border: 2px solid #ddd;
    border-radius: 10px;
    cursor: pointer;
    font-size: 1.1em;
    text-align: left;
    transition: all 0.3s;
}}

.option-button:hover {{
    background: #f0f0f0;
    border-color: #667eea;
}}

.option-button.selected {{
    background: #667eea;
    color: white;
    border-color: #667eea;
}}

.export-section {{
    margin-top: 30px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
}}

.export-button {{
    width: 100%;
    padding: 15px;
    background: #28a745;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1.2em;
    font-weight: bold;
    transition: background-color 0.3s;
}}

.export-button:hover {{
    background: #218838;
}}

.task-info {{
    margin-bottom: 20px;
    padding: 15px;
    background: #e7f3ff;
    border-radius: 8px;
}}

.task-info strong {{
    color: #667eea;
}}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 QA Human Annotation</h1>
            <p>{task_name}</p>
        </header>

        <div class="navigation">
            <button onclick="previousQuestion()" id="prevBtn">◀ Previous</button>
            <div id="questionInfo"></div>
            <button onclick="nextQuestion()" id="nextBtn">Next ▶</button>
        </div>

        <div class="progress-bar">
            <div class="progress-fill" id="progressFill">0%</div>
        </div>

        <div class="content">
            <div class="video-section">
                <h2>📹 Video</h2>
                <video id="videoPlayer" controls>
                    Your browser does not support the video tag.
                </video>
                <div class="video-info">
                    <p><strong>Video Name:</strong> <span id="videoName"></span></p>
                    <p><strong>Time Range:</strong> <span id="timeRange"></span></p>
                    <p><strong>Timestamp:</strong> <span id="timestamp"></span></p>
                </div>
            </div>

            <div class="question-section">
                <div class="task-info">
                    <p><strong>Task Type:</strong> <span id="taskType"></span></p>
                </div>
                
                <h2>Question</h2>
                <div class="question-text" id="questionText"></div>
                
                <h3 style="margin-bottom: 15px;">Select your answer:</h3>
                <div class="options" id="optionsContainer"></div>

                <div class="export-section">
                    <h3 style="margin-bottom: 15px;">Export Annotations</h3>
                    <p style="margin-bottom: 15px;">Answered: <span id="answeredCount">0</span> / <span id="totalCount">0</span></p>
                    <button class="export-button" onclick="exportToCSV()">💾 Export to CSV</button>
                </div>
            </div>
        </div>
    </div>

    <script>
const questionsData = {json.dumps(questions_data, indent=2, ensure_ascii=False)};

const taskType = "{task_type}";
let currentQuestionIndex = 0;
let annotations = {{}};

// Initialize annotations object
questionsData.forEach((item, idx) => {{
    annotations[idx] = {{
        video_path: item.video_path,
        question: item.question,
        options: item.options,
        correct_answer: item.answer,
        user_answer: null,
        time_stamp: item.time_stamp,
        task_type: item.task_type
    }};
}});

function displayQuestion(index) {{
    const item = questionsData[index];
    currentQuestionIndex = index;
    
    // Update navigation
    document.getElementById('questionInfo').textContent = `Question ${{index + 1}} / ${{questionsData.length}}`;
    document.getElementById('prevBtn').disabled = index === 0;
    document.getElementById('nextBtn').disabled = index === questionsData.length - 1;
    
    // Update progress bar
    const progress = ((index + 1) / questionsData.length) * 100;
    const progressFill = document.getElementById('progressFill');
    progressFill.style.width = progress + '%';
    progressFill.textContent = Math.round(progress) + '%';
    
    // Update task info
    document.getElementById('taskType').textContent = item.task_type;
    
    // Update question
    document.getElementById('questionText').textContent = item.question;
    
    // Update options
    const optionsContainer = document.getElementById('optionsContainer');
    optionsContainer.innerHTML = '';
    item.options.forEach((option, optIdx) => {{
        const button = document.createElement('button');
        button.className = 'option-button';
        button.textContent = option;
        button.onclick = () => selectAnswer(index, option.charAt(0));
        
        // Check if this option was previously selected
        if (annotations[index].user_answer === option.charAt(0)) {{
            button.classList.add('selected');
        }}
        
        optionsContainer.appendChild(button);
    }});
    
    // Update video
    const videoName = getVideoNameFromPath(item.video_path);
    const gazeVideoPath = getGazeVisualizationPath(item.video_path);
    const videoPlayer = document.getElementById('videoPlayer');
    
    document.getElementById('videoName').textContent = videoName;
    document.getElementById('timestamp').textContent = item.time_stamp;
    
    if (taskType === 'past') {{
        // For past tasks: play from 0 to timestamp
        const endTime = timeToSeconds(item.time_stamp);
        document.getElementById('timeRange').textContent = `0:00 - ${{item.time_stamp}}`;
        
        videoPlayer.src = gazeVideoPath + '#t=0,' + endTime;
    }} else {{
        // For other tasks: might need different logic
        document.getElementById('timeRange').textContent = item.time_stamp;
        videoPlayer.src = gazeVideoPath;
    }}
    
    // Update answered count
    updateAnsweredCount();
}}

function getVideoNameFromPath(videoPath) {{
    const parts = videoPath.split('/');
    const filename = parts[parts.length - 1];
    return filename.replace('.mp4', '');
}}

function getGazeVisualizationPath(videoPath) {{
    // Extract video name from path
    const videoName = getVideoNameFromPath(videoPath);
    // Construct relative path from StreamingGaze directory
    // When served via HTTP server from StreamingGaze root
    return `/final_data/egtea/metadata/${{videoName}}/${{videoName}}_gaze_visualization.mp4`;
}}

function timeToSeconds(timeStr) {{
    const parts = timeStr.split(':');
    if (parts.length === 2) {{
        return parseInt(parts[0]) * 60 + parseInt(parts[1]);
    }}
    return 0;
}}

function selectAnswer(questionIndex, answer) {{
    annotations[questionIndex].user_answer = answer;
    
    // Update button styles
    const buttons = document.querySelectorAll('.option-button');
    buttons.forEach(button => {{
        button.classList.remove('selected');
        if (button.textContent.charAt(0) === answer) {{
            button.classList.add('selected');
        }}
    }});
    
    updateAnsweredCount();
}}

function updateAnsweredCount() {{
    const answered = Object.values(annotations).filter(a => a.user_answer !== null).length;
    document.getElementById('answeredCount').textContent = answered;
    document.getElementById('totalCount').textContent = questionsData.length;
}}

function previousQuestion() {{
    if (currentQuestionIndex > 0) {{
        displayQuestion(currentQuestionIndex - 1);
    }}
}}

function nextQuestion() {{
    if (currentQuestionIndex < questionsData.length - 1) {{
        displayQuestion(currentQuestionIndex + 1);
    }}
}}

function exportToCSV() {{
    const answered = Object.values(annotations).filter(a => a.user_answer !== null).length;
    
    if (answered === 0) {{
        alert('Please answer at least one question before exporting.');
        return;
    }}
    
    // Create CSV content
    let csv = 'Question Index,Video Path,Task Type,Question,Correct Answer,User Answer,Is Correct,Time Stamp\\n';
    
    Object.entries(annotations).forEach(([index, data]) => {{
        if (data.user_answer !== null) {{
            const isCorrect = data.user_answer === data.correct_answer;
            const row = [
                index,
                data.video_path,
                data.task_type,
                `"${{data.question.replace(/"/g, '""')}}"`,
                data.correct_answer,
                data.user_answer,
                isCorrect,
                data.time_stamp
            ].join(',');
            csv += row + '\\n';
        }}
    }});
    
    // Download CSV
    const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', '{task_name}_annotations.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    alert(`Exported ${{answered}} annotations to CSV!`);
}}

// Initialize
displayQuestion(0);
    </script>
</body>
</html>
"""
    return html

def generate_past_html(sample_size=50):
    """Generate HTML for past tasks"""
    
    past_tasks = [
        "egtea_past_before_tasks.json",
        "egtea_past_location_tasks.json",
        "egtea_past_sequence_tasks.json",
        "egtea_past_unviewed_tasks.json"
    ]
    
    qa_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_data', 'egtea', 'qa', 'final')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qa_html', 'past')
    
    for task_file in past_tasks:
        task_name = task_file.replace("egtea_", "").replace("_tasks.json", "")
        input_path = os.path.join(qa_dir, task_file)
        output_path = os.path.join(output_dir, f"{task_name}.html")
        
        print(f"Generating HTML for {task_name}...")
        
        # Load QA data
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Flatten data structure for HTML
        questions_data = []
        for item in data:
            # Handle both 'video_path' and 'video' keys
            video_path = item.get('video_path') or item.get('video')
            for question in item.get('questions', []):
                questions_data.append({
                    'video_path': video_path,
                    'question': question['question'],
                    'options': question['options'],
                    'answer': question['answer'],
                    'time_stamp': question['time_stamp'],
                    'task_type': question.get('task_type', 'unknown')
                })
        
        # Randomly sample questions
        total_questions = len(questions_data)
        if total_questions > sample_size:
            questions_data = random.sample(questions_data, sample_size)
            print(f"  ⚠ Sampled {sample_size} out of {total_questions} questions")
        
        # Generate HTML
        html_content = generate_html_template(task_name, questions_data, 'past')
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  ✓ Generated: {output_path}")
        print(f"  ✓ Total questions in HTML: {len(questions_data)}")

if __name__ == "__main__":
    print("="*80)
    print("Generating QA Annotation HTML files (50 samples per task)")
    print("="*80)
    
    generate_past_html(sample_size=50)
    
    print("\n" + "="*80)
    print("✅ All HTML files generated successfully!")
    print("="*80)


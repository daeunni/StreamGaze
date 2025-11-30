#!/usr/bin/env python3
"""
Batch process all EGTEA videos to extract clips and generate multiple HTML files (V2 format)
Uses the v2 template with corrected button logic and missing object addition feature
"""

import os
import pandas as pd
import subprocess
import json
from pathlib import Path
from datetime import datetime
import math
import base64

# Configuration
METADATA_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'final_data', 'egtea', 'metadata')
WORK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'html_output')
CLIPS_DIR = os.path.join(WORK_DIR, "clips")
EPISODES_PER_HTML = 10  # Number of episodes per HTML file

def read_file_content(file_path):
    """Read file content as text"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def get_all_video_sources():
    """Get list of all video source directories"""
    video_sources = []
    
    for item in sorted(os.listdir(METADATA_BASE_DIR)):
        item_path = os.path.join(METADATA_BASE_DIR, item)
        if os.path.isdir(item_path):
            csv_file = os.path.join(item_path, f"{item}_fixation_merged_filtered_v2.csv")
            video_file = os.path.join(item_path, f"{item}_gaze_visualization.mp4")
            
            if os.path.exists(csv_file) and os.path.exists(video_file):
                video_sources.append({
                    'name': item,
                    'csv_path': csv_file,
                    'video_path': video_file
                })
            else:
                print(f"⚠️  Skipping {item}: Missing required files")
    
    return video_sources

def extract_clips_for_video(video_info, start_episode_id):
    """Extract video clips for a single video source"""
    name = video_info['name']
    csv_path = video_info['csv_path']
    video_path = video_info['video_path']
    
    print(f"\n📹 Processing: {name}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Found {len(df)} episodes")
    except Exception as e:
        print(f"   ✗ Error reading CSV: {e}")
        return [], start_episode_id
    
    os.makedirs(CLIPS_DIR, exist_ok=True)
    
    episodes_data = []
    current_episode_id = start_episode_id
    
    for idx, row in df.iterrows():
        try:
            start_time = float(row['episode_start_time'])
            end_time = float(row['episode_end_time'])
            duration = end_time - start_time
            
            clip_filename = f"{name}_episode_{idx:02d}.mp4"
            clip_path = os.path.join(CLIPS_DIR, clip_filename)
            
            if os.path.exists(clip_path):
                print(f"   ✓ Clip {clip_filename} exists")
            else:
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-movflags', '+faststart',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    clip_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"   ✓ Created {clip_filename}")
                else:
                    print(f"   ✗ Error creating {clip_filename}")
                    continue
            
            def safe_json_parse(field_value):
                if pd.isna(field_value) or field_value == '':
                    return None
                try:
                    import ast
                    return ast.literal_eval(str(field_value))
                except:
                    return None
            
            episode = {
                'id': current_episode_id,
                'video_source': name,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(duration),
                'fixation_ids': row['fixation_ids'],
                'clip_filename': clip_filename,
                'representative_object': safe_json_parse(row['representative_object']) or {
                    'object_identity': 'Parse Error',
                    'detailed_caption': 'Could not parse data'
                },
                'other_objects_in_cropped_area': safe_json_parse(row['other_objects_in_cropped_area']) or [],
                'other_objects_outside_fov': safe_json_parse(row['other_objects_outside_fov']) or [],
                'captions': safe_json_parse(row['captions']) or []
            }
            
            episodes_data.append(episode)
            current_episode_id += 1
            
        except Exception as e:
            print(f"   ✗ Error processing row {idx}: {e}")
            continue
    
    return episodes_data, current_episode_id

def generate_single_html_v2(episodes, batch_num, total_batches):
    """Generate a single HTML file for a batch of episodes using V2 format"""
    
    # Read CSS (using existing styles.css)
    css_path = os.path.join(WORK_DIR, "styles.css")
    with open(css_path, 'r', encoding='utf-8') as f:
        css_content = f.read()
    
    # Encode videos to base64
    video_data = {}
    for episode in episodes:
        clip_filename = episode['clip_filename']
        clip_path = os.path.join(CLIPS_DIR, clip_filename)
        
        if os.path.exists(clip_path):
            with open(clip_path, 'rb') as f:
                video_bytes = f.read()
                video_data[clip_filename] = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
    
    total_size_mb = sum(os.path.getsize(os.path.join(CLIPS_DIR, ep['clip_filename'])) 
                       for ep in episodes if os.path.exists(os.path.join(CLIPS_DIR, ep['clip_filename']))) / (1024 * 1024)
    estimated_html_size_mb = total_size_mb * 1.33
    
    # Generate HTML with V2 format
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Verification of Metadata - Batch {batch_num}/{total_batches}</title>
    <style>
{css_content}

/* Enhanced styles for exclusion functionality */
.standalone-info {{
    background: #e8f5e8;
    border: 1px solid #4caf50;
    color: #2e7d32;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
    text-align: center;
    font-size: 0.9em;
}}

.export-section {{
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}}

.export-section h3 {{
    color: #667eea;
    margin-bottom: 15px;
    font-size: 1.2em;
}}

.export-stats {{
    margin-bottom: 15px;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 5px;
    font-weight: bold;
}}

.export-buttons {{
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
}}

.export-btn {{
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
    font-weight: bold;
    transition: all 0.3s;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.csv-btn {{
    background: #28a745;
    color: white;
}}

.csv-btn:hover {{
    background: #218838;
}}

.excel-btn {{
    background: #17a2b8;
    color: white;
}}

.excel-btn:hover {{
    background: #138496;
}}

.object-card-container {{
    position: relative;
}}

.object-card-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
}}

.exclude-btn {{
    padding: 4px 8px;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-size: 11px;
    font-weight: bold;
    transition: all 0.3s;
    min-width: 60px;
}}

.exclude-btn.exclude {{
    background: #dc3545;
    color: white;
}}

.exclude-btn.exclude:hover {{
    background: #c82333;
}}

.exclude-btn.include {{
    background: #28a745;
    color: white;
}}

.exclude-btn.include:hover {{
    background: #218838;
}}

.excluded-object {{
    opacity: 0.6;
    background: #f8f9fa !important;
}}

.excluded-object .object-name {{
    text-decoration: line-through;
    color: #6c757d;
}}

.excluded-object .object-description {{
    text-decoration: line-through;
    color: #6c757d;
}}

/* Missing object input section */
.missing-object-section {{
    margin-top: 20px;
    padding: 15px;
    background: #e7f3ff;
    border-radius: 8px;
    border: 2px dashed #4a90e2;
}}

.missing-object-section h4 {{
    margin-bottom: 10px;
    color: #2c5aa0;
    font-size: 14px;
}}

.missing-object-inputs {{
    display: flex;
    flex-direction: column;
    gap: 10px;
}}

.missing-object-inputs input {{
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 13px;
}}

.missing-object-inputs textarea {{
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 13px;
    min-height: 60px;
    resize: vertical;
}}

.add-missing-btn {{
    padding: 8px 16px;
    background: #4a90e2;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    font-weight: bold;
    transition: all 0.3s;
}}

.add-missing-btn:hover {{
    background: #357abd;
}}

.added-missing-objects {{
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid #b3d9ff;
}}

.added-missing-item {{
    background: white;
    padding: 10px;
    margin-top: 8px;
    border-radius: 5px;
    border-left: 3px solid #4a90e2;
    position: relative;
}}

.added-missing-item .object-name {{
    font-weight: bold;
    color: #2c5aa0;
    margin-bottom: 5px;
}}

.added-missing-item .object-description {{
    font-size: 12px;
    color: #555;
}}

.remove-missing-btn {{
    position: absolute;
    top: 5px;
    right: 5px;
    background: #dc3545;
    color: white;
    border: none;
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 10px;
    cursor: pointer;
}}

.remove-missing-btn:hover {{
    background: #c82333;
}}

.video-source-label {{
    background: #667eea;
    color: white;
    padding: 8px 12px;
    border-radius: 5px;
    font-size: 0.9em;
    margin-bottom: 10px;
    display: inline-block;
}}
    </style>
    
    <!-- SheetJS library for Excel export -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="standalone-info">
            📦 Batch {batch_num} of {total_batches} - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 
            | {len(video_data)} videos embedded | ~{estimated_html_size_mb:.0f}MB total
        </div>
        
        <header>
            <h1>Human Verification of Metadata</h1>
            <p>Batch {batch_num} of {total_batches} - {len(episodes)} episodes</p>
        </header>

        <div class="export-section">
            <h3>📊 Data Export & Quality Control</h3>
            <div class="export-stats" id="exportStats">
                Loading statistics...
            </div>
            <div class="export-buttons">
                <button onclick="exportToCSV()" class="export-btn csv-btn">
                    📄 Export CSV
                </button>
                <button onclick="exportToExcel()" class="export-btn excel-btn">
                    📊 Export Excel
                </button>
                <button onclick="resetAllExclusions()" class="export-btn" style="background: #ffc107; color: #212529;">
                    🔄 Reset All
                </button>
            </div>
        </div>

        <div class="navigation">
            <button id="prevBtn" onclick="previousEpisode()">◀ Previous</button>
            <span id="episodeInfo">Episode 1 of {len(episodes)}</span>
            <button id="nextBtn" onclick="nextEpisode()">Next ▶</button>
        </div>

        <div class="content">
            <div class="video-section">
                <h2>Video Clip</h2>
                <div class="video-source-label" id="videoSourceLabel"></div>
                <video id="videoPlayer" controls>
                    <source id="videoSource" src="" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="video-info">
                    <p><strong>Duration:</strong> <span id="duration"></span> seconds</p>
                    <p><strong>Time Range:</strong> <span id="timeRange"></span></p>
                    <p><strong>Fixation IDs:</strong> <span id="fixationIds"></span></p>
                </div>
            </div>

            <div class="objects-section">
                <div class="representative-object">
                    <h3>Pointing Object (Green dot)</h3>
                    <div id="representativeObject" class="object-card"></div>
                </div>

                <div class="other-objects">
                    <h3>Other Objects in Red Circle Area</h3>
                    <div id="otherObjectsCropped" class="objects-grid"></div>
                </div>

                <div class="outside-objects">
                    <h3>Objects Outside of Red Circle Area</h3>
                    <div id="objectsOutsideFov" class="objects-grid"></div>
                </div>
            </div>

            <div class="captions-section">
                <h3>Scene Captions</h3>
                <div id="captions" class="captions-list"></div>
            </div>
        </div>
    </div>

    <script>
// Global variables
let episodeData = {json.dumps(episodes, indent=4)};
let videoData = {json.dumps(video_data, indent=4)};
let currentEpisodeIndex = 0;
const BATCH_NUM = {batch_num};

// Exclusion data structure
let exclusionData = {{
    metadata: {{
        batchNumber: {batch_num},
        exportVersion: "2.0",
        lastModified: null,
        totalEpisodes: {len(episodes)},
        userNotes: ""
    }},
    episodes: {{}}
}};

// Initialize exclusion data for all episodes
function initializeExclusionData() {{
    episodeData.forEach(episode => {{
        if (!exclusionData.episodes[episode.id]) {{
            exclusionData.episodes[episode.id] = {{
                representative: {{
                    excluded: false,
                    reason: "",
                    timestamp: null,
                    notes: "",
                    objectIdentity: episode.representative_object?.object_identity || "Unknown"
                }},
                cropped: episode.other_objects_in_cropped_area?.map((obj, index) => ({{
                    index: index,
                    objectIdentity: obj.object_identity || "Unknown",
                    excluded: false,
                    reason: "",
                    timestamp: null,
                    notes: ""
                }})) || [],
                outside: episode.other_objects_outside_fov?.map((obj, index) => ({{
                    index: index,
                    objectIdentity: obj.object_identity || "Unknown",
                    excluded: false,
                    reason: "",
                    timestamp: null,
                    notes: ""
                }})) || [],
                missingRepresentative: [],
                missingCropped: [],
                missingOutside: []
            }};
        }}
    }});
    
    loadExclusionData();
    updateExportStats();
}}

// Save exclusion data to localStorage
function saveExclusionData() {{
    exclusionData.metadata.lastModified = new Date().toISOString();
    localStorage.setItem(`egtea_exclusions_batch_${{BATCH_NUM}}_v2`, JSON.stringify(exclusionData));
}}

// Load exclusion data from localStorage
function loadExclusionData() {{
    try {{
        const saved = localStorage.getItem(`egtea_exclusions_batch_${{BATCH_NUM}}_v2`);
        if (saved) {{
            const loadedData = JSON.parse(saved);
            Object.keys(loadedData.episodes || {{}}).forEach(episodeId => {{
                if (exclusionData.episodes[episodeId]) {{
                    exclusionData.episodes[episodeId] = loadedData.episodes[episodeId];
                }}
            }});
            console.log('Loaded exclusion data from localStorage');
        }}
    }} catch (error) {{
        console.error('Error loading exclusion data:', error);
    }}
}}

// Toggle object exclusion
function toggleObjectExclusion(episodeId, type, index) {{
    const episodeExclusions = exclusionData.episodes[episodeId];
    if (!episodeExclusions) return;
    
    let targetObj;
    if (type === 'representative') {{
        targetObj = episodeExclusions.representative;
    }} else {{
        targetObj = episodeExclusions[type][index];
    }}
    
    if (targetObj) {{
        targetObj.excluded = !targetObj.excluded;
        targetObj.timestamp = new Date().toISOString();
        
        if (!targetObj.excluded) {{
            targetObj.reason = "";
            targetObj.notes = "";
        }}
        
        saveExclusionData();
        updateExportStats();
        displayEpisode(currentEpisodeIndex);
    }}
}}

// Add missing object
function addMissingObject(episodeId, section) {{
    const nameInput = document.getElementById(`missing-name-${{section}}-${{episodeId}}`);
    const captionInput = document.getElementById(`missing-caption-${{section}}-${{episodeId}}`);
    
    const name = nameInput.value.trim();
    const caption = captionInput.value.trim();
    
    if (!name || !caption) {{
        alert('Please enter both object name and caption');
        return;
    }}
    
    const episodeExclusions = exclusionData.episodes[episodeId];
    if (!episodeExclusions) return;
    
    const missingKey = `missing${{section.charAt(0).toUpperCase() + section.slice(1)}}`;
    if (!episodeExclusions[missingKey]) {{
        episodeExclusions[missingKey] = [];
    }}
    
    episodeExclusions[missingKey].push({{
        objectIdentity: name,
        detailedCaption: caption,
        timestamp: new Date().toISOString(),
        userAdded: true
    }});
    
    // Clear inputs
    nameInput.value = '';
    captionInput.value = '';
    
    // Save and refresh
    saveExclusionData();
    displayEpisode(currentEpisodeIndex);
    updateExportStats();
}}

// Remove missing object
function removeMissingObject(episodeId, section, index) {{
    const episodeExclusions = exclusionData.episodes[episodeId];
    if (!episodeExclusions) return;
    
    const missingKey = `missing${{section.charAt(0).toUpperCase() + section.slice(1)}}`;
    if (episodeExclusions[missingKey]) {{
        episodeExclusions[missingKey].splice(index, 1);
    }}
    
    saveExclusionData();
    displayEpisode(currentEpisodeIndex);
    updateExportStats();
}}

// Get exclusion state
function getExclusionState(episodeId, type, index) {{
    const episodeExclusions = exclusionData.episodes[episodeId];
    if (!episodeExclusions) return {{ excluded: false, reason: "" }};
    
    if (type === 'representative') {{
        return episodeExclusions.representative;
    }} else {{
        return episodeExclusions[type][index] || {{ excluded: false, reason: "" }};
    }}
}}

// Update export statistics
function updateExportStats() {{
    let totalObjects = 0;
    let excludedObjects = 0;
    
    Object.values(exclusionData.episodes).forEach(episode => {{
        totalObjects++;
        if (episode.representative.excluded) excludedObjects++;
        
        totalObjects += episode.cropped.length;
        excludedObjects += episode.cropped.filter(obj => obj.excluded).length;
        
        totalObjects += episode.outside.length;
        excludedObjects += episode.outside.filter(obj => obj.excluded).length;
    }});
    
    const inclusionRate = totalObjects > 0 ? ((totalObjects - excludedObjects) / totalObjects * 100).toFixed(1) : 0;
    
    document.getElementById('exportStats').innerHTML = `
        <strong>Quality Control Status:</strong> ${{excludedObjects}} objects excluded out of ${{totalObjects}} total 
        | Inclusion Rate: ${{inclusionRate}}%
    `;
}}

// Reset all exclusions
function resetAllExclusions() {{
    if (confirm('Are you sure you want to reset all exclusions? This cannot be undone.')) {{
        Object.values(exclusionData.episodes).forEach(episode => {{
            episode.representative.excluded = false;
            episode.representative.reason = "";
            episode.representative.notes = "";
            
            episode.cropped.forEach(obj => {{
                obj.excluded = false;
                obj.reason = "";
                obj.notes = "";
            }});
            
            episode.outside.forEach(obj => {{
                obj.excluded = false;
                obj.reason = "";
                obj.notes = "";
            }});
            
            episode.missingRepresentative = [];
            episode.missingCropped = [];
            episode.missingOutside = [];
        }});
        
        saveExclusionData();
        updateExportStats();
        displayEpisode(currentEpisodeIndex);
        alert('All exclusions and added objects have been reset.');
    }}
}}

// Generate CSV data
function generateCSVData() {{
    const headers = [
        'Episode_ID', 'Video_Source', 'Object_Type', 'Object_Index', 'Object_Identity', 
        'Object_Description', 'Excluded', 'User_Added', 'Timestamp', 'Episode_Start_Time',
        'Episode_End_Time', 'Episode_Duration', 'Fixation_IDs'
    ];
    
    let csvContent = headers.join(',') + '\\n';
    
    episodeData.forEach(episode => {{
        const episodeExclusions = exclusionData.episodes[episode.id] || {{}};
        
        const repState = episodeExclusions.representative || {{}};
        const repObj = episode.representative_object || {{}};
        csvContent += [
            episode.id,
            `"${{episode.video_source}}"`,
            'representative',
            0,
            `"${{(repObj.object_identity || 'Unknown').replace(/"/g, '""')}}"`,
            `"${{(repObj.detailed_caption || 'No description').replace(/"/g, '""')}}"`,
            repState.excluded || false,
            'false',
            repState.timestamp || '',
            episode.start_time,
            episode.end_time,
            episode.duration,
            `"${{episode.fixation_ids || ''}}"`
        ].join(',') + '\\n';
        
        (episode.other_objects_in_cropped_area || []).forEach((obj, index) => {{
            const objState = (episodeExclusions.cropped || [])[index] || {{}};
            csvContent += [
                episode.id,
                `"${{episode.video_source}}"`,
                'cropped',
                index,
                `"${{(obj.object_identity || 'Unknown').replace(/"/g, '""')}}"`,
                `"${{(obj.detailed_caption || 'No description').replace(/"/g, '""')}}"`,
                objState.excluded || false,
                'false',
                objState.timestamp || '',
                episode.start_time,
                episode.end_time,
                episode.duration,
                `"${{episode.fixation_ids || ''}}"`
            ].join(',') + '\\n';
        }});
        
        (episode.other_objects_outside_fov || []).forEach((obj, index) => {{
            const objState = (episodeExclusions.outside || [])[index] || {{}};
            csvContent += [
                episode.id,
                `"${{episode.video_source}}"`,
                'outside',
                index,
                `"${{(obj.object_identity || 'Unknown').replace(/"/g, '""')}}"`,
                `"${{(obj.detailed_caption || 'No description').replace(/"/g, '""')}}"`,
                objState.excluded || false,
                'false',
                objState.timestamp || '',
                episode.start_time,
                episode.end_time,
                episode.duration,
                `"${{episode.fixation_ids || ''}}"`
            ].join(',') + '\\n';
        }});
        
        // Missing objects (user added)
        (episodeExclusions.missingRepresentative || []).forEach((obj, index) => {{
            csvContent += [
                episode.id,
                `"${{episode.video_source}}"`,
                'missing_representative',
                index,
                `"${{(obj.objectIdentity || 'Unknown').replace(/"/g, '""')}}"`,
                `"${{(obj.detailedCaption || 'No description').replace(/"/g, '""')}}"`,
                'false',
                'true',
                obj.timestamp || '',
                episode.start_time,
                episode.end_time,
                episode.duration,
                `"${{episode.fixation_ids || ''}}"`
            ].join(',') + '\\n';
        }});
        
        (episodeExclusions.missingCropped || []).forEach((obj, index) => {{
            csvContent += [
                episode.id,
                `"${{episode.video_source}}"`,
                'missing_cropped',
                index,
                `"${{(obj.objectIdentity || 'Unknown').replace(/"/g, '""')}}"`,
                `"${{(obj.detailedCaption || 'No description').replace(/"/g, '""')}}"`,
                'false',
                'true',
                obj.timestamp || '',
                episode.start_time,
                episode.end_time,
                episode.duration,
                `"${{episode.fixation_ids || ''}}"`
            ].join(',') + '\\n';
        }});
        
        (episodeExclusions.missingOutside || []).forEach((obj, index) => {{
            csvContent += [
                episode.id,
                `"${{episode.video_source}}"`,
                'missing_outside',
                index,
                `"${{(obj.objectIdentity || 'Unknown').replace(/"/g, '""')}}"`,
                `"${{(obj.detailedCaption || 'No description').replace(/"/g, '""')}}"`,
                'false',
                'true',
                obj.timestamp || '',
                episode.start_time,
                episode.end_time,
                episode.duration,
                `"${{episode.fixation_ids || ''}}"`
            ].join(',') + '\\n';
        }});
    }});
    
    return csvContent;
}}

// Export to CSV
function exportToCSV() {{
    try {{
        const csvData = generateCSVData();
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        const filename = `egtea_batch_{batch_num}_v2_${{timestamp}}.csv`;
        
        downloadFile(csvData, filename, 'text/csv');
        console.log('CSV export completed:', filename);
    }} catch (error) {{
        console.error('CSV export error:', error);
        alert('Error exporting CSV: ' + error.message);
    }}
}}

// Export to Excel
function exportToExcel() {{
    try {{
        if (typeof XLSX === 'undefined') {{
            alert('Excel export library not loaded. Please check your internet connection.');
            return;
        }}
        
        const wb = XLSX.utils.book_new();
        
        const mainData = generateExportData();
        const ws1 = XLSX.utils.json_to_sheet(mainData);
        XLSX.utils.book_append_sheet(wb, ws1, "Data");
        
        const summaryData = generateSummaryData();
        const ws2 = XLSX.utils.json_to_sheet(summaryData);
        XLSX.utils.book_append_sheet(wb, ws2, "Summary");
        
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        const filename = `egtea_batch_{batch_num}_v2_${{timestamp}}.xlsx`;
        
        XLSX.writeFile(wb, filename);
        console.log('Excel export completed:', filename);
    }} catch (error) {{
        console.error('Excel export error:', error);
        alert('Error exporting Excel: ' + error.message);
    }}
}}

// Generate export data for Excel
function generateExportData() {{
    const data = [];
    
    episodeData.forEach(episode => {{
        const episodeExclusions = exclusionData.episodes[episode.id] || {{}};
        
        const repState = episodeExclusions.representative || {{}};
        const repObj = episode.representative_object || {{}};
        data.push({{
            Episode_ID: episode.id,
            Video_Source: episode.video_source,
            Object_Type: 'representative',
            Object_Index: 0,
            Object_Identity: repObj.object_identity || 'Unknown',
            Object_Description: repObj.detailed_caption || 'No description',
            Excluded: repState.excluded || false,
            User_Added: false,
            Timestamp: repState.timestamp || '',
            Episode_Start_Time: episode.start_time,
            Episode_End_Time: episode.end_time,
            Episode_Duration: episode.duration,
            Fixation_IDs: episode.fixation_ids || ''
        }});
        
        (episode.other_objects_in_cropped_area || []).forEach((obj, index) => {{
            const objState = (episodeExclusions.cropped || [])[index] || {{}};
            data.push({{
                Episode_ID: episode.id,
                Video_Source: episode.video_source,
                Object_Type: 'cropped',
                Object_Index: index,
                Object_Identity: obj.object_identity || 'Unknown',
                Object_Description: obj.detailed_caption || 'No description',
                Excluded: objState.excluded || false,
                User_Added: false,
                Timestamp: objState.timestamp || '',
                Episode_Start_Time: episode.start_time,
                Episode_End_Time: episode.end_time,
                Episode_Duration: episode.duration,
                Fixation_IDs: episode.fixation_ids || ''
            }});
        }});
        
        (episode.other_objects_outside_fov || []).forEach((obj, index) => {{
            const objState = (episodeExclusions.outside || [])[index] || {{}};
            data.push({{
                Episode_ID: episode.id,
                Video_Source: episode.video_source,
                Object_Type: 'outside',
                Object_Index: index,
                Object_Identity: obj.object_identity || 'Unknown',
                Object_Description: obj.detailed_caption || 'No description',
                Excluded: objState.excluded || false,
                User_Added: false,
                Timestamp: objState.timestamp || '',
                Episode_Start_Time: episode.start_time,
                Episode_End_Time: episode.end_time,
                Episode_Duration: episode.duration,
                Fixation_IDs: episode.fixation_ids || ''
            }});
        }});
        
        // Missing objects
        (episodeExclusions.missingRepresentative || []).forEach((obj, index) => {{
            data.push({{
                Episode_ID: episode.id,
                Video_Source: episode.video_source,
                Object_Type: 'missing_representative',
                Object_Index: index,
                Object_Identity: obj.objectIdentity || 'Unknown',
                Object_Description: obj.detailedCaption || 'No description',
                Excluded: false,
                User_Added: true,
                Timestamp: obj.timestamp || '',
                Episode_Start_Time: episode.start_time,
                Episode_End_Time: episode.end_time,
                Episode_Duration: episode.duration,
                Fixation_IDs: episode.fixation_ids || ''
            }});
        }});
        
        (episodeExclusions.missingCropped || []).forEach((obj, index) => {{
            data.push({{
                Episode_ID: episode.id,
                Video_Source: episode.video_source,
                Object_Type: 'missing_cropped',
                Object_Index: index,
                Object_Identity: obj.objectIdentity || 'Unknown',
                Object_Description: obj.detailedCaption || 'No description',
                Excluded: false,
                User_Added: true,
                Timestamp: obj.timestamp || '',
                Episode_Start_Time: episode.start_time,
                Episode_End_Time: episode.end_time,
                Episode_Duration: episode.duration,
                Fixation_IDs: episode.fixation_ids || ''
            }});
        }});
        
        (episodeExclusions.missingOutside || []).forEach((obj, index) => {{
            data.push({{
                Episode_ID: episode.id,
                Video_Source: episode.video_source,
                Object_Type: 'missing_outside',
                Object_Index: index,
                Object_Identity: obj.objectIdentity || 'Unknown',
                Object_Description: obj.detailedCaption || 'No description',
                Excluded: false,
                User_Added: true,
                Timestamp: obj.timestamp || '',
                Episode_Start_Time: episode.start_time,
                Episode_End_Time: episode.end_time,
                Episode_Duration: episode.duration,
                Fixation_IDs: episode.fixation_ids || ''
            }});
        }});
    }});
    
    return data;
}}

// Generate summary data
function generateSummaryData() {{
    const summary = [];
    
    episodeData.forEach(episode => {{
        const episodeExclusions = exclusionData.episodes[episode.id] || {{}};
        
        let totalObjects = 1;
        let excludedObjects = episodeExclusions.representative?.excluded ? 1 : 0;
        let addedObjects = 0;
        
        totalObjects += (episode.other_objects_in_cropped_area || []).length;
        excludedObjects += (episodeExclusions.cropped || []).filter(obj => obj.excluded).length;
        
        totalObjects += (episode.other_objects_outside_fov || []).length;
        excludedObjects += (episodeExclusions.outside || []).filter(obj => obj.excluded).length;
        
        addedObjects += (episodeExclusions.missingRepresentative || []).length;
        addedObjects += (episodeExclusions.missingCropped || []).length;
        addedObjects += (episodeExclusions.missingOutside || []).length;
        
        const inclusionRate = totalObjects > 0 ? ((totalObjects - excludedObjects) / totalObjects * 100).toFixed(1) : 0;
        
        summary.push({{
            Episode_ID: episode.id,
            Video_Source: episode.video_source,
            Total_Objects: totalObjects,
            Excluded_Objects: excludedObjects,
            Included_Objects: totalObjects - excludedObjects,
            User_Added_Objects: addedObjects,
            Inclusion_Rate_Percent: inclusionRate,
            Episode_Duration: episode.duration,
            Start_Time: episode.start_time,
            End_Time: episode.end_time
        }});
    }});
    
    return summary;
}}

// Download file utility
function downloadFile(content, filename, contentType) {{
    const blob = new Blob([content], {{ type: contentType }});
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
}}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {{
    console.log('Batch {batch_num} V2 loading...');
    console.log('Episodes loaded:', episodeData.length);
    console.log('Videos embedded:', Object.keys(videoData).length);
    
    initializeExclusionData();
    
    if (episodeData.length > 0) {{
        displayEpisode(0);
    }} else {{
        console.error('No episode data found');
    }}
}});

// Display episode data
function displayEpisode(index) {{
    if (index < 0 || index >= episodeData.length) return;
    
    currentEpisodeIndex = index;
    const episode = episodeData[index];
    
    console.log('Displaying episode:', episode.id);
    
    // Update navigation
    document.getElementById('episodeInfo').textContent = `Episode ${{index + 1}} of ${{episodeData.length}}`;
    document.getElementById('prevBtn').disabled = index === 0;
    document.getElementById('nextBtn').disabled = index === episodeData.length - 1;
    
    // Update video source label
    document.getElementById('videoSourceLabel').textContent = episode.video_source;
    
    // Update video
    updateVideo(episode);
    
    // Update video info
    updateVideoInfo(episode);
    
    // Update objects
    updateObjects(episode);
    
    // Update captions
    updateCaptions(episode);
}}

// Update video with embedded data
function updateVideo(episode) {{
    const videoPlayer = document.getElementById('videoPlayer');
    const videoSource = document.getElementById('videoSource');
    
    if (episode.clip_filename && videoData[episode.clip_filename]) {{
        const dataUrl = videoData[episode.clip_filename];
        videoSource.src = dataUrl;
        videoPlayer.load();
        
        videoPlayer.onerror = function(e) {{
            console.error('Video loading error:', e);
        }};
        
        videoPlayer.onloadedmetadata = function() {{
            videoPlayer.currentTime = 0;
        }};
    }} else {{
        videoPlayer.style.display = 'none';
    }}
}}

// Update video information
function updateVideoInfo(episode) {{
    document.getElementById('duration').textContent = parseFloat(episode.duration).toFixed(2);
    document.getElementById('timeRange').textContent = 
        `${{parseFloat(episode.start_time).toFixed(2)}}s - ${{parseFloat(episode.end_time).toFixed(2)}}s`;
    document.getElementById('fixationIds').textContent = episode.fixation_ids;
}}

// Create object card HTML (V2 - corrected button logic)
function createObjectCard(obj, type, episodeId, index) {{
    if (!obj || typeof obj !== 'object') {{
        return '<p>No data available</p>';
    }}
    
    const identity = obj.object_identity || 'Unknown';
    const description = obj.detailed_caption || 'No description available';
    const exclusionState = getExclusionState(episodeId, type, index);
    const isExcluded = exclusionState.excluded;
    
    const cardId = `card_${{episodeId}}_${{type}}_${{index}}`;
    const btnId = `btn_${{episodeId}}_${{type}}_${{index}}`;
    
    return `
        <div class="object-card-container ${{isExcluded ? 'excluded-object' : ''}}" id="${{cardId}}">
            <div class="object-card-header">
                <div class="object-name">${{identity}}</div>
                <button class="exclude-btn ${{isExcluded ? 'exclude' : 'include'}}" 
                        id="${{btnId}}" 
                        onclick="toggleObjectExclusion(${{episodeId}}, '${{type}}', ${{index}})">
                    ${{isExcluded ? 'Exclude' : 'Include'}}
                </button>
            </div>
            <div class="object-description">${{description}}</div>
        </div>
    `;
}}

// Create missing object input section
function createMissingObjectSection(episodeId, section, sectionTitle) {{
    const episodeExclusions = exclusionData.episodes[episodeId];
    const missingKey = `missing${{section.charAt(0).toUpperCase() + section.slice(1)}}`;
    const missingObjects = episodeExclusions?.[missingKey] || [];
    
    return `
        <div class="missing-object-section">
            <h4>➕ If this data should be excluded, please describe what you see</h4>
            <div class="missing-object-inputs">
                <input type="text" 
                       id="missing-name-${{section}}-${{episodeId}}" 
                       placeholder="Object name (e.g., knife, cutting board)">
                <textarea 
                    id="missing-caption-${{section}}-${{episodeId}}" 
                    placeholder="1~2 sentences Description of the object including attributes (color, material) and location nearby specific objects"></textarea>
                <button class="add-missing-btn" onclick="addMissingObject(${{episodeId}}, '${{section}}')">
                    Add Object
                </button>
            </div>
            ${{missingObjects.length > 0 ? `
                <div class="added-missing-objects">
                    <strong>Added Objects:</strong>
                    ${{missingObjects.map((obj, idx) => `
                        <div class="added-missing-item">
                            <button class="remove-missing-btn" onclick="removeMissingObject(${{episodeId}}, '${{section}}', ${{idx}})">✕</button>
                            <div class="object-name">${{obj.objectIdentity}}</div>
                            <div class="object-description">${{obj.detailedCaption}}</div>
                        </div>
                    `).join('')}}
                </div>
            ` : ''}}
        </div>
    `;
}}

// Update objects display
function updateObjects(episode) {{
    // Representative object
    const repObjContainer = document.getElementById('representativeObject');
    const repObj = episode.representative_object;
    repObjContainer.innerHTML = createObjectCard(repObj, 'representative', episode.id, 0) +
                                 createMissingObjectSection(episode.id, 'representative', 'Green dot');
    
    // Other objects in cropped area
    const croppedContainer = document.getElementById('otherObjectsCropped');
    const croppedObjects = episode.other_objects_in_cropped_area;
    croppedContainer.innerHTML = '';
    if (Array.isArray(croppedObjects)) {{
        croppedObjects.forEach((obj, index) => {{
            const div = document.createElement('div');
            div.className = 'object-item';
            div.innerHTML = createObjectCard(obj, 'cropped', episode.id, index);
            croppedContainer.appendChild(div);
        }});
    }}
    
    // Objects outside FOV
    const outsideContainer = document.getElementById('objectsOutsideFov');
    const outsideObjects = episode.other_objects_outside_fov;
    outsideContainer.innerHTML = '';
    if (Array.isArray(outsideObjects)) {{
        outsideObjects.forEach((obj, index) => {{
            const div = document.createElement('div');
            div.className = 'outside-object-item';
            div.innerHTML = createObjectCard(obj, 'outside', episode.id, index);
            outsideContainer.appendChild(div);
        }});
    }}
}}

// Update captions
function updateCaptions(episode) {{
    const captionsContainer = document.getElementById('captions');
    const captions = episode.captions;
    
    captionsContainer.innerHTML = '';
    if (Array.isArray(captions)) {{
        captions.forEach(caption => {{
            const div = document.createElement('div');
            div.className = 'caption-item';
            div.textContent = caption;
            captionsContainer.appendChild(div);
        }});
    }}
}}

// Navigation functions
function previousEpisode() {{
    if (currentEpisodeIndex > 0) {{
        displayEpisode(currentEpisodeIndex - 1);
    }}
}}

function nextEpisode() {{
    if (currentEpisodeIndex < episodeData.length - 1) {{
        displayEpisode(currentEpisodeIndex + 1);
    }}
}}

// Keyboard navigation
document.addEventListener('keydown', function(event) {{
    if (event.key === 'ArrowLeft') {{
        previousEpisode();
    }} else if (event.key === 'ArrowRight') {{
        nextEpisode();
    }}
}});

// Auto-save exclusion data periodically
setInterval(saveExclusionData, 30000);

// Save on page unload
window.addEventListener('beforeunload', function() {{
    saveExclusionData();
}});
    </script>
</body>
</html>"""
    
    # Write HTML file
    output_filename = os.path.join(WORK_DIR, f"egtea_batch_{batch_num:02d}_v2.html")
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Also save JSON
    json_path = os.path.join(WORK_DIR, f"episodes_batch_{batch_num:02d}_v2.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)

def generate_batch_htmls(all_episodes):
    """Generate multiple HTML files, each containing a subset of episodes"""
    
    total_episodes = len(all_episodes)
    num_htmls = math.ceil(total_episodes / EPISODES_PER_HTML)
    
    print(f"\n🏗️  Generating {num_htmls} HTML files (V2 format, {EPISODES_PER_HTML} episodes per file)")
    
    for html_idx in range(num_htmls):
        start_idx = html_idx * EPISODES_PER_HTML
        end_idx = min(start_idx + EPISODES_PER_HTML, total_episodes)
        
        batch_episodes = all_episodes[start_idx:end_idx]
        
        generate_single_html_v2(batch_episodes, html_idx + 1, num_htmls)
        
        print(f"   ✓ Generated HTML batch {html_idx + 1}/{num_htmls} ({len(batch_episodes)} episodes)")

def main():
    print("=" * 70)
    print("🚀 EGTEA Batch Video Processing & HTML Generation (V2 Format)")
    print("=" * 70)
    
    # Load existing episodes_complete.json if it exists
    complete_json_path = os.path.join(WORK_DIR, "episodes_complete.json")
    
    if os.path.exists(complete_json_path):
        print("\n📋 Loading existing episodes data...")
        with open(complete_json_path, 'r', encoding='utf-8') as f:
            all_episodes = json.load(f)
        print(f"   Loaded {len(all_episodes)} episodes from cache")
    else:
        print("\n📋 Scanning video sources...")
        video_sources = get_all_video_sources()
        print(f"   Found {len(video_sources)} valid video sources")
        
        if len(video_sources) == 0:
            print("❌ No valid video sources found!")
            return
        
        print(f"\n🎬 Extracting clips from {len(video_sources)} videos...")
        all_episodes = []
        current_episode_id = 1
        
        for idx, video_info in enumerate(video_sources, 1):
            print(f"\n[{idx}/{len(video_sources)}]", end=" ")
            episodes, current_episode_id = extract_clips_for_video(video_info, current_episode_id)
            all_episodes.extend(episodes)
        
        print(f"\n✅ Extracted {len(all_episodes)} total episodes")
        
        with open(complete_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_episodes, f, indent=2, ensure_ascii=False)
        print(f"   Saved complete episodes list to: episodes_complete.json")
    
    # Generate batch HTMLs (V2 format)
    generate_batch_htmls(all_episodes)
    
    print("\n" + "=" * 70)
    print("✨ Processing Complete! (V2 Format)")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   • Total episodes: {len(all_episodes)}")
    print(f"   • Episodes per HTML: {EPISODES_PER_HTML}")
    print(f"   • Total HTML files: {math.ceil(len(all_episodes) / EPISODES_PER_HTML)}")
    print(f"\n📂 Output directory: {WORK_DIR}")
    print(f"   • HTML files: egtea_batch_01_v2.html, egtea_batch_02_v2.html, ...")
    print(f"   • JSON files: episodes_batch_01_v2.json, episodes_batch_02_v2.json, ...")
    print(f"   • Video clips: clips/ directory")
    print(f"\n🌐 Open any V2 HTML file in a browser to start annotation!")
    print(f"\n✨ V2 Features:")
    print(f"   • Corrected Include/Exclude button logic")
    print(f"   • Add missing objects functionality")
    print(f"   • Enhanced CSV/Excel export with user-added objects")

if __name__ == "__main__":
    main()








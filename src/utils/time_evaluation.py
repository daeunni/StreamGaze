import json
import statistics
import numpy as np
from typing import List, Dict, Any, Tuple


class TimeEvaluator:
    """
    Time evaluation utility for analyzing response times from model outputs
    """
    
    def __init__(self):
        self.response_times = []
        self.task_times = {}
    
    def add_response_time(self, response_time: float, task_name: str = "default"):
        """Add a response time measurement"""
        self.response_times.append(response_time)
        
        if task_name not in self.task_times:
            self.task_times[task_name] = []
        self.task_times[task_name].append(response_time)
    
    def get_statistics(self, task_name: str = None) -> Dict[str, float]:
        """Get statistical analysis of response times"""
        if task_name and task_name in self.task_times:
            times = self.task_times[task_name]
        else:
            times = self.response_times
            
        if not times:
            return {}
            
        return {
            "count": len(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "percentile_95": np.percentile(times, 95),
            "percentile_99": np.percentile(times, 99)
        }
    
    def get_all_task_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tasks"""
        results = {}
        for task_name in self.task_times.keys():
            results[task_name] = self.get_statistics(task_name)
        
        # Add overall statistics
        results["overall"] = self.get_statistics()
        return results
    
    def save_time_analysis(self, output_path: str):
        """Save time analysis to JSON file"""
        analysis = {
            "statistics": self.get_all_task_statistics(),
            "raw_data": {
                "all_times": self.response_times,
                "task_times": self.task_times
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=4)
    
    def print_summary(self):
        """Print summary of response time statistics"""
        stats = self.get_all_task_statistics()
        
        print("\n" + "="*50)
        print("RESPONSE TIME ANALYSIS SUMMARY")
        print("="*50)
        
        for task_name, task_stats in stats.items():
            if not task_stats:
                continue
                
            print(f"\n{task_name.upper()}:")
            print(f"  Count: {task_stats['count']}")
            print(f"  Mean: {task_stats['mean']:.3f}s")
            print(f"  Median: {task_stats['median']:.3f}s")
            print(f"  Std: {task_stats['std']:.3f}s")
            print(f"  Min: {task_stats['min']:.3f}s")
            print(f"  Max: {task_stats['max']:.3f}s")
            print(f"  95th percentile: {task_stats['percentile_95']:.3f}s")
            print(f"  99th percentile: {task_stats['percentile_99']:.3f}s")


def load_time_data_from_vispeak_output(output_files: List[str]) -> TimeEvaluator:
    """
    Load time data from ViSpeak_bench.py output files
    
    Args:
        output_files: List of JSON output files from ViSpeak_bench.py
        
    Returns:
        TimeEvaluator with loaded data
    """
    evaluator = TimeEvaluator()
    
    for file_path in output_files:
        # Extract task name from filename (e.g., "Visual_Reference_output.json" -> "Visual_Reference")
        task_name = file_path.split('/')[-1].replace('_output.json', '')
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for entry in data:
                for video_name, (response, response_time) in entry.items():
                    evaluator.add_response_time(response_time, task_name)
                    
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading {file_path}: {e}")
    
    return evaluator


def analyze_vispeak_times(output_dir: str, task_names: List[str] = None) -> TimeEvaluator:
    """
    Analyze response times from ViSpeak benchmark outputs
    
    Args:
        output_dir: Directory containing ViSpeak output files
        task_names: List of task names to analyze (if None, analyzes all found files)
        
    Returns:
        TimeEvaluator with analysis results
    """
    import os
    
    if task_names is None:
        # Default ViSpeak tasks
        task_names = [
            'Anomaly_Warning', 'Gesture_Understanding', 'Humor_Reaction',
            'Visual_Interruption', 'Visual_Reference', 'Visual_Termination', 
            'Visual_Wake-Up'
        ]
    
    output_files = []
    for task_name in task_names:
        file_path = os.path.join(output_dir, f"{task_name}_output.json")
        if os.path.exists(file_path):
            output_files.append(file_path)
    
    return load_time_data_from_vispeak_output(output_files)


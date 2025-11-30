import json
import numpy as np
from typing import List, Dict, Any, Tuple


class TimingAccuracyEvaluator:
    """
    Implements Time Accuracy evaluation as described in the paper:
    T_acc = (1/N) * sum(I(T_res^(i) ∈ [t1, t2 + T]))
    
    Where:
    - T_res^(i): Response time for question i
    - [t1, t2]: Ground-truth time span
    - T: Time margin
    - N: Number of questions in subtask
    """
    
    def __init__(self, time_margin_T: float = 3.0):
        """
        Args:
            time_margin_T: Time margin T in seconds (default 3.0s)
        """
        self.time_margin_T = time_margin_T
        self.results = []
    
    def evaluate_timing_accuracy(self, 
                                response_time: float, 
                                ground_truth_start: float, 
                                ground_truth_end: float) -> bool:
        """
        Check if response time falls within ground-truth time window + margin
        
        Args:
            response_time: T_res^(i) - when model responded
            ground_truth_start: t1 - start of ground-truth time span
            ground_truth_end: t2 - end of ground-truth time span
            
        Returns:
            bool: True if T_res ∈ [t1, t2 + T], False otherwise
        """
        window_start = ground_truth_start
        window_end = ground_truth_end + self.time_margin_T
        
        return window_start <= response_time <= window_end
    
    def add_evaluation(self, 
                      response_time: float,
                      ground_truth_start: float,
                      ground_truth_end: float,
                      subtask: str = "default",
                      question_id: str = None):
        """
        Add a timing evaluation result
        
        Args:
            response_time: Model response time
            ground_truth_start: Ground-truth start time
            ground_truth_end: Ground-truth end time  
            subtask: Subtask name for grouping
            question_id: Optional question identifier
        """
        is_accurate = self.evaluate_timing_accuracy(
            response_time, ground_truth_start, ground_truth_end
        )
        
        result = {
            "question_id": question_id,
            "subtask": subtask,
            "response_time": response_time,
            "ground_truth_start": ground_truth_start,
            "ground_truth_end": ground_truth_end,
            "time_margin": self.time_margin_T,
            "window_start": ground_truth_start,
            "window_end": ground_truth_end + self.time_margin_T,
            "is_timing_accurate": is_accurate
        }
        
        self.results.append(result)
        return is_accurate
    
    def calculate_timing_accuracy_by_subtask(self) -> Dict[str, float]:
        """
        Calculate T_acc^s for each subtask s
        
        Returns:
            Dict mapping subtask name to timing accuracy score
        """
        subtask_results = {}
        
        # Group by subtask
        subtasks = {}
        for result in self.results:
            subtask = result["subtask"]
            if subtask not in subtasks:
                subtasks[subtask] = []
            subtasks[subtask].append(result)
        
        # Calculate T_acc^s for each subtask
        for subtask, results in subtasks.items():
            N_s = len(results)  # Number of questions in subtask s
            correct_count = sum(1 for r in results if r["is_timing_accurate"])
            
            T_acc_s = correct_count / N_s if N_s > 0 else 0.0
            subtask_results[subtask] = T_acc_s
        
        return subtask_results
    
    def calculate_overall_timing_accuracy(self) -> float:
        """
        Calculate overall timing accuracy across all subtasks
        
        Returns:
            Overall T_acc score
        """
        if not self.results:
            return 0.0
            
        total_correct = sum(1 for r in self.results if r["is_timing_accurate"])
        total_questions = len(self.results)
        
        return total_correct / total_questions
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Get detailed evaluation results
        
        Returns:
            Dictionary with detailed timing accuracy analysis
        """
        subtask_accuracy = self.calculate_timing_accuracy_by_subtask()
        overall_accuracy = self.calculate_overall_timing_accuracy()
        
        # Calculate additional statistics
        response_times = [r["response_time"] for r in self.results]
        timing_errors = []
        
        for result in self.results:
            # Calculate how far off the response was from optimal timing
            optimal_time = (result["ground_truth_start"] + result["ground_truth_end"]) / 2
            error = abs(result["response_time"] - optimal_time)
            timing_errors.append(error)
        
        return {
            "timing_accuracy": {
                "overall": overall_accuracy,
                "by_subtask": subtask_accuracy
            },
            "time_margin_T": self.time_margin_T,
            "total_questions": len(self.results),
            "correct_timing_count": sum(1 for r in self.results if r["is_timing_accurate"]),
            "statistics": {
                "mean_response_time": np.mean(response_times) if response_times else 0,
                "std_response_time": np.std(response_times) if response_times else 0,
                "mean_timing_error": np.mean(timing_errors) if timing_errors else 0,
                "std_timing_error": np.std(timing_errors) if timing_errors else 0
            },
            "detailed_results": self.results
        }
    
    def print_timing_accuracy_summary(self):
        """
        Print timing accuracy evaluation summary
        """
        results = self.get_detailed_results()
        
        print("\n" + "="*60)
        print("TIMING ACCURACY EVALUATION RESULTS")
        print("="*60)
        print(f"Time Margin (T): {self.time_margin_T}s")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Timing: {results['correct_timing_count']}")
        print(f"Overall T_acc: {results['timing_accuracy']['overall']:.3f}")
        
        print(f"\nBy Subtask:")
        for subtask, accuracy in results['timing_accuracy']['by_subtask'].items():
            subtask_count = len([r for r in self.results if r['subtask'] == subtask])
            subtask_correct = len([r for r in self.results if r['subtask'] == subtask and r['is_timing_accurate']])
            print(f"  {subtask}: {accuracy:.3f} ({subtask_correct}/{subtask_count})")
        
        stats = results['statistics']
        print(f"\nTiming Statistics:")
        print(f"  Mean Response Time: {stats['mean_response_time']:.3f}s")
        print(f"  Std Response Time: {stats['std_response_time']:.3f}s")
        print(f"  Mean Timing Error: {stats['mean_timing_error']:.3f}s")
        print(f"  Std Timing Error: {stats['std_timing_error']:.3f}s")
    
    def save_results(self, output_path: str):
        """
        Save timing accuracy results to JSON file
        
        Args:
            output_path: Path to save results
        """
        results = self.get_detailed_results()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)


def parse_ground_truth_time_span(time_span_str: str) -> Tuple[float, float]:
    """
    Parse ground-truth time span string to start and end times
    
    Args:
        time_span_str: String like "01:23-01:28" or "83-88" (seconds)
        
    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    if '-' not in time_span_str:
        raise ValueError(f"Invalid time span format: {time_span_str}")
    
    start_str, end_str = time_span_str.split('-', 1)
    
    def parse_time(time_str):
        time_str = time_str.strip()
        if ':' in time_str:
            # Format: mm:ss or hh:mm:ss
            parts = list(map(int, time_str.split(':')))
            if len(parts) == 2:  # mm:ss
                return parts[0] * 60 + parts[1]
            elif len(parts) == 3:  # hh:mm:ss
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            # Format: seconds
            return float(time_str)
    
    start_time = parse_time(start_str)
    end_time = parse_time(end_str)
    
    return start_time, end_time


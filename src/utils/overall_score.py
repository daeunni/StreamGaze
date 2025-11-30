import json
import numpy as np
from typing import Dict, Any, List, Optional


class OverallScoreCalculator:
    """
    Calculate overall score O combining timing accuracy T_acc and text quality score S
    Based on the paper's evaluation metrics
    """
    
    def __init__(self, timing_weight: float = 0.5, text_weight: float = 0.5):
        """
        Args:
            timing_weight: Weight for timing accuracy component (default 0.5)
            text_weight: Weight for text quality component (default 0.5)
        """
        if abs(timing_weight + text_weight - 1.0) > 1e-6:
            raise ValueError("timing_weight + text_weight must equal 1.0")
            
        self.timing_weight = timing_weight
        self.text_weight = text_weight
        self.evaluations = []
    
    def add_evaluation(self, 
                      timing_accuracy: float,
                      text_quality_score: float, 
                      subtask: str = "default",
                      question_id: str = None):
        """
        Add an evaluation result
        
        Args:
            timing_accuracy: T_acc score (0.0 or 1.0 for individual question)
            text_quality_score: S score (0.0 to 1.0)
            subtask: Subtask name
            question_id: Question identifier
        """
        overall_score = (self.timing_weight * timing_accuracy + 
                        self.text_weight * text_quality_score)
        
        evaluation = {
            "question_id": question_id,
            "subtask": subtask,
            "timing_accuracy": timing_accuracy,
            "text_quality_score": text_quality_score,
            "overall_score": overall_score,
            "timing_weight": self.timing_weight,
            "text_weight": self.text_weight
        }
        
        self.evaluations.append(evaluation)
        return overall_score
    
    def calculate_subtask_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate average scores by subtask
        
        Returns:
            Dict mapping subtask to score components
        """
        subtask_results = {}
        
        # Group by subtask
        subtasks = {}
        for eval_result in self.evaluations:
            subtask = eval_result["subtask"]
            if subtask not in subtasks:
                subtasks[subtask] = []
            subtasks[subtask].append(eval_result)
        
        # Calculate averages for each subtask
        for subtask, results in subtasks.items():
            timing_scores = [r["timing_accuracy"] for r in results]
            text_scores = [r["text_quality_score"] for r in results]
            overall_scores = [r["overall_score"] for r in results]
            
            subtask_results[subtask] = {
                "count": len(results),
                "timing_accuracy": np.mean(timing_scores),
                "text_quality_score": np.mean(text_scores),
                "overall_score": np.mean(overall_scores),
                "timing_std": np.std(timing_scores),
                "text_std": np.std(text_scores),
                "overall_std": np.std(overall_scores)
            }
        
        return subtask_results
    
    def calculate_overall_scores(self) -> Dict[str, float]:
        """
        Calculate overall scores across all evaluations
        
        Returns:
            Dict with overall score components
        """
        if not self.evaluations:
            return {}
        
        timing_scores = [e["timing_accuracy"] for e in self.evaluations]
        text_scores = [e["text_quality_score"] for e in self.evaluations]
        overall_scores = [e["overall_score"] for e in self.evaluations]
        
        return {
            "count": len(self.evaluations),
            "timing_accuracy": np.mean(timing_scores),
            "text_quality_score": np.mean(text_scores),
            "overall_score": np.mean(overall_scores),
            "timing_std": np.std(timing_scores),
            "text_std": np.std(text_scores),
            "overall_std": np.std(overall_scores)
        }
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Get detailed results including all components
        
        Returns:
            Dict with comprehensive results
        """
        return {
            "weights": {
                "timing_weight": self.timing_weight,
                "text_weight": self.text_weight
            },
            "overall": self.calculate_overall_scores(),
            "by_subtask": self.calculate_subtask_scores(),
            "detailed_evaluations": self.evaluations
        }
    
    def print_summary(self):
        """
        Print summary of overall score evaluation
        """
        results = self.get_detailed_results()
        
        print("\n" + "="*60)
        print("OVERALL SCORE EVALUATION RESULTS")
        print("="*60)
        print(f"Timing Weight: {self.timing_weight:.2f}")
        print(f"Text Quality Weight: {self.text_weight:.2f}")
        
        overall = results["overall"]
        if overall:
            print(f"\nOverall Results (N={overall['count']}):")
            print(f"  Timing Accuracy: {overall['timing_accuracy']:.3f} ± {overall['timing_std']:.3f}")
            print(f"  Text Quality Score: {overall['text_quality_score']:.3f} ± {overall['text_std']:.3f}")
            print(f"  Overall Score (O): {overall['overall_score']:.3f} ± {overall['overall_std']:.3f}")
        
        print(f"\nBy Subtask:")
        for subtask, scores in results["by_subtask"].items():
            print(f"  {subtask} (N={scores['count']}):")
            print(f"    Timing Accuracy: {scores['timing_accuracy']:.3f} ± {scores['timing_std']:.3f}")
            print(f"    Text Quality: {scores['text_quality_score']:.3f} ± {scores['text_std']:.3f}")
            print(f"    Overall Score: {scores['overall_score']:.3f} ± {scores['overall_std']:.3f}")
    
    def save_results(self, output_path: str):
        """
        Save results to JSON file
        
        Args:
            output_path: Path to save results
        """
        results = self.get_detailed_results()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)


def calculate_text_quality_score(predicted_text: str, 
                                ground_truth_text: str,
                                method: str = "exact_match") -> float:
    """
    Calculate text quality score S
    
    Args:
        predicted_text: Model's response
        ground_truth_text: Ground truth answer
        method: Scoring method ("exact_match", "contains", "bleu", etc.)
        
    Returns:
        Text quality score (0.0 to 1.0)
    """
    if method == "exact_match":
        return 1.0 if predicted_text.lower().strip() == ground_truth_text.lower().strip() else 0.0
    
    elif method == "contains":
        return 1.0 if ground_truth_text.lower() in predicted_text.lower() else 0.0
    
    elif method == "first_char":
        # For multiple choice questions - check first character
        pred_first = predicted_text.strip().lower()
        gt_first = ground_truth_text.strip().lower()
        if pred_first and gt_first:
            return 1.0 if pred_first[0] == gt_first[0] else 0.0
        return 0.0
    
    elif method == "bleu":
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            
            reference = [word_tokenize(ground_truth_text.lower())]
            candidate = word_tokenize(predicted_text.lower())
            
            return sentence_bleu(reference, candidate)
        except ImportError:
            print("Warning: NLTK not available, falling back to exact match")
            return calculate_text_quality_score(predicted_text, ground_truth_text, "exact_match")
    
    else:
        raise ValueError(f"Unknown text quality scoring method: {method}")


def combine_timing_and_text_evaluations(timing_accuracy_results: Dict[str, Any],
                                       text_quality_results: Dict[str, Any],
                                       timing_weight: float = 0.5) -> OverallScoreCalculator:
    """
    Combine timing accuracy and text quality evaluations into overall score
    
    Args:
        timing_accuracy_results: Results from TimingAccuracyEvaluator
        text_quality_results: Results with text quality scores
        timing_weight: Weight for timing component (text weight = 1 - timing_weight)
        
    Returns:
        OverallScoreCalculator with combined results
    """
    calculator = OverallScoreCalculator(timing_weight, 1 - timing_weight)
    
    # Match evaluations by question_id
    timing_details = timing_accuracy_results.get("detailed_results", [])
    text_details = text_quality_results.get("detailed_results", [])
    
    # Create lookup for text quality results
    text_lookup = {result["question_id"]: result for result in text_details}
    
    for timing_result in timing_details:
        question_id = timing_result["question_id"]
        
        if question_id in text_lookup:
            text_result = text_lookup[question_id]
            
            calculator.add_evaluation(
                timing_accuracy=1.0 if timing_result["is_timing_accurate"] else 0.0,
                text_quality_score=text_result["text_quality_score"],
                subtask=timing_result["subtask"],
                question_id=question_id
            )
    
    return calculator


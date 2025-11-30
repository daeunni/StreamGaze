import json
import re
from typing import Tuple, Dict, Any, List


def parse_time_range(time_str: str) -> Tuple[float, float]:
    """
    Parse time range string like "[01:10 - 01:13]" to start and end seconds
    
    Args:
        time_str: Time range string in format "[mm:ss - mm:ss]" or "[hh:mm:ss - hh:mm:ss]"
        
    Returns:
        Tuple of (start_seconds, end_seconds)
    """
    # Remove brackets and split by " - "
    time_str = time_str.strip("[]")
    if " - " not in time_str:
        raise ValueError(f"Invalid time range format: {time_str}")
    
    start_str, end_str = time_str.split(" - ")
    
    def time_to_seconds(time_part: str) -> float:
        parts = time_part.strip().split(":")
        if len(parts) == 2:  # mm:ss
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # hh:mm:ss
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            raise ValueError(f"Invalid time format: {time_part}")
    
    start_seconds = time_to_seconds(start_str)
    end_seconds = time_to_seconds(end_str)
    
    return start_seconds, end_seconds


def generate_ground_truth_time_span(time_range_str: str, margin_seconds: int = 10) -> str:
    """
    Generate ground truth time span from time range with margin
    
    Args:
        time_range_str: Original time range like "[01:10 - 01:13]"
        margin_seconds: Additional margin in seconds (default 10)
        
    Returns:
        Ground truth time span string like "70-93" (start_seconds-end_seconds+margin)
    """
    start_seconds, end_seconds = parse_time_range(time_range_str)
    
    # Ground truth window: [start, end + margin]
    gt_start = start_seconds
    gt_end = end_seconds + margin_seconds
    
    return f"{gt_start}-{gt_end}"


def add_ground_truth_to_data(data: List[Dict[str, Any]], margin_seconds: int = 10) -> List[Dict[str, Any]]:
    """
    Add ground truth time spans to data entries
    
    Args:
        data: List of data entries with 'time' field
        margin_seconds: Margin to add to end time (default 10)
        
    Returns:
        Data with added 'ground_truth_time_span' fields
    """
    enhanced_data = []
    
    for entry in data:
        enhanced_entry = entry.copy()
        
        if "time" in entry and entry["time"].startswith("[") and " - " in entry["time"]:
            try:
                gt_time_span = generate_ground_truth_time_span(entry["time"], margin_seconds)
                enhanced_entry["ground_truth_time_span"] = gt_time_span
                
                # Also parse the individual components for debugging
                start_sec, end_sec = parse_time_range(entry["time"])
                enhanced_entry["_debug_info"] = {
                    "original_time": entry["time"],
                    "start_seconds": start_sec,
                    "end_seconds": end_sec,
                    "margin_seconds": margin_seconds,
                    "gt_window": f"[{start_sec}, {end_sec + margin_seconds}]"
                }
                
            except ValueError as e:
                print(f"Warning: Could not parse time range '{entry['time']}': {e}")
                enhanced_entry["ground_truth_time_span"] = None
        else:
            enhanced_entry["ground_truth_time_span"] = None
            
        enhanced_data.append(enhanced_entry)
    
    return enhanced_data


def process_egtea_data_file(input_file: str, output_file: str, margin_seconds: int = 10):
    """
    Process EGTEA data file to add ground truth time spans
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        margin_seconds: Margin to add to end time (default 10)
    """
    print(f"Processing {input_file}...")
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries")
    
    # Add ground truth time spans
    enhanced_data = add_ground_truth_to_data(data, margin_seconds)
    
    # Count entries with ground truth
    gt_count = sum(1 for entry in enhanced_data if entry.get("ground_truth_time_span"))
    print(f"Added ground truth time spans to {gt_count} entries")
    
    # Save enhanced data
    with open(output_file, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print(f"Saved enhanced data to {output_file}")
    
    # Print some examples
    print("\nExamples:")
    for i, entry in enumerate(enhanced_data[:3]):
        if entry.get("ground_truth_time_span"):
            print(f"  {i+1}. Original time: {entry['time']}")
            print(f"     Ground truth span: {entry['ground_truth_time_span']}")
            if "_debug_info" in entry:
                debug = entry["_debug_info"]
                print(f"     GT window: {debug['gt_window']}")
            print()


def validate_ground_truth_generation():
    """
    Test the ground truth generation with example data
    """
    print("="*60)
    print("GROUND TRUTH GENERATION VALIDATION")
    print("="*60)
    
    test_cases = [
        "[01:10 - 01:13]",  # 70-83 seconds -> GT: 70-93 (with +10 margin)
        "[01:27 - 01:30]",  # 87-90 seconds -> GT: 87-100 (with +10 margin)  
        "[00:05 - 00:08]",  # 5-8 seconds -> GT: 5-18 (with +10 margin)
    ]
    
    margin = 10
    print(f"Using margin: {margin} seconds")
    print()
    
    for time_range in test_cases:
        try:
            start_sec, end_sec = parse_time_range(time_range)
            gt_span = generate_ground_truth_time_span(time_range, margin)
            
            print(f"Original time: {time_range}")
            print(f"  Parsed: {start_sec}s - {end_sec}s")
            print(f"  Ground truth span: {gt_span}")
            print(f"  GT window: [{start_sec}, {end_sec + margin}]")
            print()
            
        except Exception as e:
            print(f"Error processing {time_range}: {e}")
            print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ground truth time spans for EGTEA data")
    parser.add_argument("--input_file", type=str, help="Input JSON file path")
    parser.add_argument("--output_file", type=str, help="Output JSON file path")
    parser.add_argument("--margin_seconds", type=int, default=10, help="Time margin in seconds")
    
    args = parser.parse_args()
    
    # If no arguments provided, run validation and default example
    if not args.input_file:
        # Validate the logic first
        validate_ground_truth_generation()
        
        # Example usage
        input_file = "../../../StreamingGaze/final_data/0815/egtea_past_memory_retrieval_tasks.json"
        output_file = "data/egtea_with_ground_truth.json"
        margin_seconds = 10
        
        print("="*60)
        print("PROCESSING EGTEA DATA (DEFAULT)")
        print("="*60)
        
        try:
            process_egtea_data_file(input_file, output_file, margin_seconds)
        except FileNotFoundError:
            print(f"File not found: {input_file}")
            print("Please provide the correct path to your EGTEA data file")
        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        # Use command line arguments
        print("="*60)
        print("PROCESSING EGTEA DATA (COMMAND LINE)")
        print("="*60)
        print(f"Input: {args.input_file}")
        print(f"Output: {args.output_file}")
        print(f"Margin: {args.margin_seconds}s")
        
        try:
            process_egtea_data_file(args.input_file, args.output_file, args.margin_seconds)
        except Exception as e:
            print(f"Error processing file: {e}")
            import sys
            sys.exit(1)

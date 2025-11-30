from utils.data_execution import load_data

from model.modelclass import Model
from benchmark.Benchmark import Benchmark

import argparse

def main(args):
    data = load_data(args.data_file)

    ####### BENCHMARK #######
    # StreamGaze format benchmarks (dataset/qa format)
    if args.benchmark_name == "StreamingBenchGaze_StreamGaze":
        from benchmark.StreamingBenchGaze_StreamGaze import StreamingBenchGaze_StreamGaze
        benchmark = StreamingBenchGaze_StreamGaze(data, args.video_root, args.use_gaze_instruction)
    
    elif args.benchmark_name == "StreamingBenchGaze_Past_StreamGaze":
        from benchmark.StreamingBenchGaze_Past_StreamGaze import StreamingBenchGaze_Past_StreamGaze
        benchmark = StreamingBenchGaze_Past_StreamGaze(data, args.video_root, args.use_gaze_instruction)
    
    elif args.benchmark_name == "StreamingBenchRemind_StreamGaze":
        from benchmark.StreamingBenchRemind_StreamGaze import StreamingBenchRemind_StreamGaze
        benchmark = StreamingBenchRemind_StreamGaze(data, args.video_root, args.use_gaze_instruction)
    
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark_name}")

    ##########################

    ####### MODEL ############

    model = Model()
    
    # Dynamic model loading based on model_name
    if args.model_name == "ViSpeak":
        from model.ViSpeak import ViSpeak
        model = ViSpeak()
    elif args.model_name == "InternVL":
        from model.InternVL import InternVL
        model = InternVL()
    elif args.model_name == "LLaVAOneVision":
        from model.LLaVAOneVision import LLaVAOneVision
        model = LLaVAOneVision()
    elif args.model_name == "Qwen2VL":
        from model.Qwen2VL import Qwen2VL
        model = Qwen2VL()
    elif args.model_name == "MiniCPMV":
        from model.MiniCPMV import MiniCPMV
        model = MiniCPMV()
    elif args.model_name == "LongVA":
        from model.LongVA import LongVA
        model = LongVA()
    elif args.model_name == "VideoLLaMA2":
        from model.VideoLLaMA2 import VideoLLaMA2
        model = VideoLLaMA2()
    elif args.model_name == "VILA":
        from model.VILA import VILA
        model = VILA()
    elif args.model_name == "VideollmOnline":
        from model.VideollmOnline import VideollmOnline
        model = VideollmOnline()
    elif args.model_name == "VideollmOnline_Streaming":
        from model.VideollmOnline_Streaming import VideollmOnline_Streaming
        model = VideollmOnline_Streaming()
    elif args.model_name == "FlashVstream":
        from model.FlashVstream_Streaming import FlashVstream_Streaming
        model = FlashVstream_Streaming()
    elif args.model_name == "Kangaroo":
        from model.Kangaroo import Kangaroo
        model = Kangaroo()
    elif args.model_name == "LLaVANextVideo32":
        from model.LLaVANextVideo32 import LLaVANextVideo32
        model = LLaVANextVideo32()
    elif args.model_name == "MiniCPMo":
        from model.MiniCPMo import MiniCPMo
        model = MiniCPMo()
    elif args.model_name == "VideoCCam":
        from model.VideoCCam import VideoCCam
        model = VideoCCam()
    elif args.model_name == "Qwen3VL":
        from model.Qwen3VL import Qwen3VL
        model = Qwen3VL()
    elif args.model_name == "Qwen25VL":
        from model.Qwen25VL import Qwen25VL
        model = Qwen25VL()
    elif args.model_name == "InternVL35":
        from model.InternVL35 import InternVL35
        model = InternVL35()
    elif args.model_name == "GPT4o":
        from model.GPT4o import GPT4o
        model = GPT4o()
    elif args.model_name == "Gemini":
        from model.Gemini import Gemini
        model = Gemini()
    elif args.model_name == "ClaudeOpus4":
        from model.ClaudeOpus4 import ClaudeOpus4
        model = ClaudeOpus4()
    elif args.model_name == "ClaudeSonnet4":
        from model.ClaudeSonnet4 import ClaudeSonnet4
        model = ClaudeSonnet4()
    elif args.model_name == "Dispider":
        from model.Dispider import Dispider
        model = Dispider()
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    ######################

    benchmark.eval(data, model, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--video_root", type=str, required=True, help="Path to the video dictionary")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--benchmark_name", type=str, required=True, help="Name of the benchmark")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--use_gaze_instruction", action="store_true", help="Add gaze instruction to prompts (green dot = gaze point, red circle = FOV)")
    parser.add_argument("--gaze_viz_video_root", type=str, default=None, help="Path to gaze visualization videos (used when use_gaze_instruction is True)")
    args = parser.parse_args()
    
    # If use_gaze_instruction is True and gaze_viz_video_root is provided, use it
    if args.use_gaze_instruction and args.gaze_viz_video_root:
        args.video_root = args.gaze_viz_video_root
        print(f"🎯 Using gaze visualization videos from: {args.video_root}")
    
    main(args)
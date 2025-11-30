"""
TEMPLATE for Creating Streaming Wrappers
Copy this file and replace MODEL_NAME with your model name

This wrapper makes any model compatible with StreamingGaze evaluation
by adding support for time-based video chunking.
"""
import time as time_module
from model.modelclass import Model

# Import your original model's functions
# from model.YourModel import YourModel_Init, YourModel_Run

model = None  # Global model instance

def MODEL_Init():
    """
    Initialize your model here
    Load weights, set device, etc.
    """
    global model
    # TODO: Add your model initialization code
    # Example:
    # model = YourModel.from_pretrained("path/to/model")
    # model = model.eval().cuda()
    pass

def MODEL_Run(video_path, inp, start_time, end_time):
    """
    Run your model with time-based video chunking
    
    Args:
        video_path: path to video file
        inp: input prompt/question
        start_time: start time in seconds
        end_time: end time in seconds
    
    Returns:
        response: model's text response
    """
    # TODO: Add your model inference code
    # 
    # Tips:
    # 1. Load video frames between start_time and end_time
    # 2. Process frames according to your model's requirements
    # 3. Run inference with the prompt
    # 4. Return the response text
    
    # Example using decord:
    # from decord import VideoReader, cpu
    # import numpy as np
    # 
    # vr = VideoReader(video_path, ctx=cpu(0))
    # fps = vr.get_avg_fps()
    # start_frame = int(start_time * fps)
    # end_frame = int(end_time * fps)
    # frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    # frames = vr.get_batch(frame_indices)
    # 
    # # Process frames and run your model
    # response = model.generate(frames, inp)
    # return response
    
    response = "TODO: Implement model inference"
    return response

class MODEL_Streaming(Model):
    """
    Streaming wrapper for MODEL
    Makes the model compatible with StreamingGaze evaluation
    """
    def __init__(self):
        """Initialize the model"""
        MODEL_Init()

    def Run(self, file, inp, start_time, end_time, question_time, omni=False, proactive=False):
        """
        Run method compatible with StreamingGaze framework
        
        Args:
            file: video file path
            inp: input prompt
            start_time: start time in seconds
            end_time: end time in seconds
            question_time: question timestamp (may not be used by all models)
            omni: multimodal flag (may not be used by all models)
            proactive: proactive response flag (may not be used by all models)
        
        Returns:
            tuple: (response, response_time)
        """
        start = time_module.time()
        response = MODEL_Run(file, inp, start_time, end_time)
        response_time = time_module.time() - start
        return response, response_time
    
    def name(self):
        """Return the model name"""
        return "MODEL_NAME"  # TODO: Change this to your model name


# =============================================================================
# USAGE INSTRUCTIONS:
# =============================================================================
# 
# 1. Copy this file to YourModel_Streaming.py
# 2. Replace MODEL with your model name everywhere
# 3. Implement MODEL_Init() - initialize your model
# 4. Implement MODEL_Run() - run inference with video chunking
# 5. Update the name() method to return your model's name
# 
# 6. Add to eval_multi_model.py:
#    elif args.model_name == "YourModel":
#        from model.YourModel_Streaming import YourModel_Streaming
#        model = YourModel_Streaming()
#
# 7. Add "YourModel" to the choices list in argparse
# 
# 8. Test with:
#    python eval_multi_model.py --model_name YourModel --benchmark_name StreamingGaze ...
# =============================================================================

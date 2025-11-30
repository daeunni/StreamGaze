import json

def get_timestamps(question_set):
    """
    """
    timestamps = []
    
    for question in question_set["questions"]:
        timestamps.append(question['time_stamp'])

    return timestamps

def load_data(EVAL_DATA_FILE):
    with open(EVAL_DATA_FILE, "r") as f:
        data = json.load(f)
    
    return data

def get_model_response(model, file, inp, start_time, end_time, question_time=None, omni=False, proactive=False, salience_map_path=None):
    """
    Get the model response for the given input
    Model: Model name
    file: Video file path
    inp: Input prompt
    salience_map_path: Optional path to salience map image
    """
    return model.Run(file, inp, start_time, end_time, question_time, omni, proactive, salience_map_path)
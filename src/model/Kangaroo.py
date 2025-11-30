import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.modelclass import Model
class Kangaroo(Model):
    def __init__(self):
        Kangaroo_Init()

    def Run(self, file, inp, start_time=None, end_time=None, question_time=None, omni=False, proactive=False, salience_map_path=None):
        return Kangaroo_Run(file, inp)
    
    def name(self):
        return "Kangaroo"
    
tokenizer, model, terminators = None, None, None

def Kangaroo_Init():
    global tokenizer, model, terminators
    tokenizer = AutoTokenizer.from_pretrained("KangarooGroup/kangaroo")
    model = AutoModelForCausalLM.from_pretrained(
        "KangarooGroup/kangaroo",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to("cuda")
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

def Kangaroo_Run(file, inp):
    # Reduce num_segments to avoid index out of bounds error on long videos
    # Kangaroo has an off-by-one bug in fuse_tokens_and_images
    # Using very small value (2) to safely avoid IndexError on any video length
    out, history = model.chat(video_path=file,
                            query=inp,
                            tokenizer=tokenizer,
                            num_segments=2,  # Minimal value to avoid off-by-one bug
                            max_new_tokens=512,
                            eos_token_id=terminators,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9,)
    print('Assitant: \n', out)
    return out, 0  # Return (response, response_time) - offline model has no timing
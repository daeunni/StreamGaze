<div align="center">

<h1><b>👁️ StreamGaze</b>: Gaze-Guided Temporal Reasoning<br/>and Proactive Understanding in Streaming Videos</h1>

<img src="assets/demo_video_gif.gif" width="100%"/>

<br />

<a href="https://arxiv.org/abs/2512.01707" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-StreamGaze-red?logo=arxiv" height="20" />
</a>
<a href="https://streamgaze.github.io" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-StreamGaze-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/danaleee/StreamGaze" target="_blank">
    <img alt="HF Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-StreamGaze--Bench-ffc107?color=ffc107&logoColor=white" height="20" />
</a>

<div>
    <a href="https://daeunni.github.io/" target="_blank">Daeun Lee</a><sup>1</sup>,
    <a href="https://subhojyoti.github.io/" target="_blank">Subhojyoti Mukherjee</a><sup>2</sup>,
    <a href="https://bkveton.com/" target="_blank">Branislav Kveton</a><sup>2</sup>,
    <a href="http://ryanrossi.com/" target="_blank">Ryan A. Rossi</a><sup>2</sup>,
    <a href="https://laiviet.github.io/" target="_blank">Viet Dac Lai</a><sup>2</sup>,
    <a href="https://david-yoon.github.io/" target="_blank">Seunghyun Yoon</a><sup>2</sup>,
    <a href="https://sites.google.com/site/trungbuistanford/" target="_blank">Trung Bui</a><sup>2</sup>,
    <a href="http://francky.me/" target="_blank">Franck Dernoncourt</a><sup>2</sup>,
    <a href="https://www.cs.unc.edu/~mbansal/" target="_blank">Mohit Bansal</a><sup>1</sup>
</div>

<div>
    <sup>1</sup>UNC Chapel Hill&emsp;
    <sup>2</sup>Adobe Research&emsp;
</div>

<br />


</div>

---

## 📰 News

- **`2025-12-01`** 🚀 StreamGaze benchmark and evaluation code released!

## 📋 Contents

- [📰 News](#-news)
- [📊 StreamGaze Benchmark](#-streamgaze-benchmark)
  - [Overview](#overview)
  - [Dataset Statistics](#dataset-statistics)
  - [Task Categories](#task-categories)
- [🚀 Quick Start](#-quick-start)
  - [Data Preparation](#data-preparation)
  - [Running Evaluation](#running-evaluation)
- [🔧 Adding Your Model](#-adding-your-model)
  - [Step 1: Implement Model Wrapper](#step-1-implement-model-wrapper)
  - [Step 2: Register Model](#step-2-register-model)
  - [Step 3: Create Evaluation Script](#step-3-create-evaluation-script)
- [📊 StreamGaze Data Generation Pipeline](#-streamgaze-data-generation-pipeline)
- [📖 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [📧 Contact](#-contact)

---

## 📊 StreamGaze Benchmark


### Dataset Statistics

<div align="center">

| Category | Metric | Count |
|:--------:|:------:|:-----:|
| 📹 **Videos** | Total Videos | **285** |
| 📝 **QA Pairs** | Total Questions | **8,521** |
| 🎯 **Tasks** | Task Categories | **10 tasks** (4 Past + 4 Present + 2 Proactive) |

</div>


### Task Categories

<img src="assets/main_streamgaze.png" width="100%"/>

#### 🔙 **Past Tasks**: Memory & Temporal Recall
Models must remember and reason about events that occurred earlier in the video stream.

- **Scene Recall (SR)**: What objects did the user interact with?
- **Object Transition Prediction (OTP)**: Which object will the user look at next, given past patterns?
- **Gaze Sequence Matching (GSM)**: Which gaze pattern matches the user's attention flow?
- **Non-Fixated Objects Identification (NFI)**: Which objects appeared but were never gazed at?

#### 👁️ **Present Tasks**: Real-time Perception & Reasoning
Models must identify and understand what is currently happening based on real-time gaze.

- **Object Identification (Easy/Hard)**: What is the user currently looking at?
- **Object Attribute Recognition (OAR)**: What are the characteristics of the gazed object?
- **Future Action Prediction (FAP)**: What action is the user about to perform?

#### 🔮 **Proactive Tasks**: Anticipation & Alerting
Models must anticipate future events and proactively respond—the most challenging category.

- **Gaze-Triggered Alert (GTA)**: Notify when the user gazes at a specific target object
- **Object Appearance Alert (OAA)**: Alert when a target object appears in the scene 


### Results 
<img src="assets/table.png" width="100%"/>


## 🚀 Quick Start

We share the same structure with [StreamingBench](https://github.com/THUNLP-MT/StreamingBench)!

### Data Preparation

Download our dataset from HuggingFace and locate like below: 

```
StreamGaze/
├── dataset/
│   ├── videos/
│   │   ├── original_video/        # Original egocentric videos
│   │   └── gaze_viz_video/        # Videos with gaze overlay
│   └── qa/
│       ├── past_*.json             # Past task QA pairs
│       ├── present_*.json          # Present task QA pairs
│       └── proactive_*.json        # Proactive task QA pairs
```

### Running Evaluation

**Quick evaluation** on existing models:

```bash
# Evaluate ViSpeak (without gaze visualization)
bash scripts/vispeak.sh

# Evaluate ViSpeak (with gaze visualization)
bash scripts/vispeak.sh --use_gaze_instruction

# Evaluate GPT-4o
bash scripts/gpt4o.sh --use_gaze_instruction

# Evaluate Qwen2.5-VL
bash scripts/qwen25vl.sh --use_gaze_instruction
```

**Results** will be automatically computed and saved to:
```
results/
├── ModelName/
│   ├── results/              # Without gaze visualization
│   │   ├── *_output.json
│   │   └── evaluation_summary.json
│   └── results_viz/          # With gaze visualization
│       ├── *_output.json
│       └── evaluation_summary.json
```
---

## 🔧 Adding Your Model

Want to evaluate your own model on StreamGaze? Follow our comprehensive guide [here](docs/model_guide.md)!

### Step 1: Implement Model Wrapper

Create `src/model/YourModel.py`:

```python
from model.modelclass import Model

class YourModel(Model):
    def __init__(self):
        # Load your model
        self.model = ...
        self.processor = ...
    
    def Run(self, file, inp, start_time, end_time, question_time, 
            omni=False, proactive=False, salience_map_path=None):
        # Process video and generate response
        return "Your model's response"
    
    def name(self):
        return "YourModel"
```

### Step 2: Register Model

Add to `src/eval.py`:

```python
elif args.model_name == "YourModel":
    from model.YourModel import YourModel
    model = YourModel()
```

### Step 3: Create Evaluation Script

Create `scripts/yourmodel.sh`:

```bash
#!/bin/bash
ROOT_DIR="/path/to/StreamGaze"
MODEL_NAME="YourModel"

# Run evaluation
bash scripts/yourmodel.sh --use_gaze_instruction
```

---

## 📊 StreamGaze Data Generation Pipeline 
<img src="assets/pipeline.png" width="100%"/>

We provide an end-to-end automatic data generation pipeline that processes raw gaze data from egocentric videos and generates high-quality temporal reasoning QA pairs. 

**Pipeline Stages:**
- **Steps 0-1**: Gaze projection & fixation extraction
- **Steps 1.5-2**: Quality filtering & object identification (InternVL-3.5 38B)
- **Step 2.5**: Sequence filtering & metadata merging
- **Step 3**: QA pair generation for 12 task types
- **Step 4**: QA validation & filtering (Qwen3VL 30B)

**Supported Datasets:**
EGTEA-Gaze+, Ego4D-Gaze, HoloAssist, EgoExoLearn

📂 **Full pipeline documentation**: [`pipeline/`](pipeline/)

```bash
# Quick start
cd pipeline
bash pipeline.sh --dataset egtea
``` 


---

## 📖 Citation

If you find StreamGaze useful in your research, please consider citing our work:

```bibtex
@misc{lee2025streamgazegazeguidedtemporalreasoning,
      title={StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos}, 
      author={Daeun Lee and Subhojyoti Mukherjee and Branislav Kveton and Ryan A. Rossi and Viet Dac Lai and Seunghyun Yoon and Trung Bui and Franck Dernoncourt and Mohit Bansal},
      year={2025},
      eprint={2512.01707},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.01707}, 
}
```

---

## 🙏 Acknowledgements

We thank the following projects and datasets that made StreamGaze possible:

- **EGTEA Gaze+** 
- **EgoExoLearn**
- **HoloAssist** 
- **StreamingBench** 
  
We also thank the open-source community for providing excellent multimodal models:
- ViSpeak, InternVL, Qwen-VL, LLaVA-OneVision, Video-LLaMA, and many others

---

## 📧 Contact

For questions, issues, or collaborations:

- 📧 Email: daeun@cs.unc.edu
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/StreamGaze/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-org/StreamGaze/discussions)

---

<div align="center">

**⭐ Star us on GitHub if you find StreamGaze useful!**

Made with ❤️ by UNC Chapel Hill & Adobe Research

</div>


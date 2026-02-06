<p align="center">
    <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/f867c49d9570b88e7bbce6e25583a0ad2417cdf7/icon.png" width="70"/>
</p>

# Pusa: Thousands Timesteps Video Diffusion Model (ICLR 2026)
<p align="center">
    <a href="https://yaofang-liu.github.io/Pusa_Web/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge"></a>
    <a href="https://arxiv.org/abs/2507.16116"><img alt="Technical Report" src="https://img.shields.io/badge/Technical_Report-📜-B31B1B?style=for-the-badge"></a>
    <a href="https://huggingface.co/RaphaelLiu/Pusa-Wan2.2-V1"><img alt="Model" src="https://img.shields.io/badge/Pusa_V1.0_Wan2.2-Model-FFD700?style=for-the-badge&logo=huggingface"></a>
    <a href="https://huggingface.co/RaphaelLiu/PusaV1"><img alt="Model" src="https://img.shields.io/badge/Pusa_V1.0_Wan2.1-Model-FFD700?style=for-the-badge&logo=huggingface"></a>
</p>
<p align="center">
    <a href="https://huggingface.co/datasets/RaphaelLiu/PusaV1_training"><img alt="Dataset" src="https://img.shields.io/badge/Pusa_V1.0-Dataset-6495ED"></a>
    <a href="https://github.com/Yaofang-Liu/Mochi-Full-Finetuner"><img alt="Code" src="https://img.shields.io/badge/Code-Training%20Scripts-32CD32?logo=github"></a>
    <a href="https://arxiv.org/abs/2410.03160"><img alt="Paper" src="https://img.shields.io/badge/📜-FVDM%20Paper-B31B1B?logo=arxiv"></a>
    <a href="https://x.com/stephenajason"><img alt="Twitter" src="https://img.shields.io/badge/🐦-Twitter-1DA1F2?logo=twitter"></a>
    <a href="https://www.xiaohongshu.com/discovery/item/67f898dc000000001c008339"><img alt="Xiaohongshu" src="https://img.shields.io/badge/📕-Xiaohongshu-FF2442"></a>
</p>


## **🔥🔥🔥🚀 New: Wan2.2 Support & LightX2V Acceleration🚀🔥🔥🔥**
- **MoE DiT Architecture**: Now supporting Wan2.2 models with separate high-noise and low-noise DiT models for enhanced quality
- **⚡ [LightX2V](https://github.com/ModelTC/LightX2V) Integration**: Achieve 4-step inference with lightning-fast generation while maintaining quality
- **Unified Framework**: Compatible with both Wan2.1 and Wan2.2 architectures
We are excited to release **Pusa V1.0**, a groundbreaking paradigm that leverages **vectorized timestep adaptation (VTA)** to enable fine-grained temporal control within a unified video diffusion framework. By finetuning the SOTA **Wan-T2V-14B** model with VTA, Pusa V1.0 achieves unprecedented efficiency, **surpassing Wan-I2V on Vbench-I2V with only $500 of training cost and 4k data**. 

## :sparkles: Highlights
- **[ComfyUI](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Pusa)**, supported by [Kijai](https://github.com/kijai), thanks a lot! 
- **[WAN2.2-14B-Rapid-AllInOne](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne)**, big salute to [Phr00t](https://huggingface.co/Phr00t)!


## Showcases
### **Various tasks in one model, all in 4 steps with LightX2V**
  **Text-to-Video, Image-to-Video, Start-End Frames, Video Completion, Video Extension, Video Transition, and more...**. Below are some showcases for Pusa-Wan2.2-V1. Please refer to **[Pusa V1.0 README](./PusaV1/README.md)** for detailed info.

**Example 1: Image-to-Video in 4 Steps**
<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
  <tr>
    <td align="center" style="padding: 8px;">
      <strong>noise: 0.2, high_lora_alpha 1.5</strong><br>
      <video src="https://github.com/user-attachments/assets/2eb2c158-fea7-4a7b-b5ee-239cea33ee01" width="100%" controls loop></video>
    </td>
    <td align="center" style="padding: 8px;">
      <strong>noise: 0.3, high_lora_alpha 1.4</strong><br>
      <video src="https://github.com/user-attachments/assets/5e0aedfa-d177-44cf-b707-ffd970952d33" width="100%" controls loop></video>
    </td>
    
  </tr>
  <tr>
    <td align="center" style="padding: 8px;">
      <strong>noise: 0.2, high_lora_alpha 1.5</strong><br>
      <video src="https://github.com/user-attachments/assets/a7347b4c-7618-45aa-8049-f91d2492eb03" width="100%" controls loop></video>
    </td>
    <td align="center" style="padding: 8px;">
      <strong>noise: 0.2, high_lora_alpha 1.5</strong><br>
      <video src="https://github.com/user-attachments/assets/27f51555-6298-4daf-8ccc-4afae4ce9eb5" width="100%" controls loop></video>
    </td>
  </tr>
</table>

**Example 2: Video Extension in 4 Steps**
<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
  <tr>
    <td align="center" style="padding: 8px;">
      <strong>noise: [0.0, 0.3, 0.5, 0.7], high_lora_alpha 1.5</strong><br>
      <video src="https://github.com/user-attachments/assets/1ee51dc5-4fb0-4fd7-806f-946a1d749dce" width="100%" controls loop></video>
    </td>
    <td align="center" style="padding: 8px;">
      <strong>noise: [0.2, 0.4, 0.4, 0.4], high_lora_alpha 1.4</strong><br>
      <video src="https://github.com/user-attachments/assets/09161d45-0893-4994-93f6-7b369f18ed7f" width="100%" controls loop></video>
    </td>
  </tr>
</table>


**Example 3: Start-End Frames in 4 Steps**
<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
  <tr>
    <td align="center" style="padding: 8px;">
      <strong>noise: [0.2, 0.5], high_lora_alpha 1.5</strong><br>
      <video src="https://github.com/user-attachments/assets/cabc0680-988f-4cf6-9413-e5483a93169c" width="100%" controls loop></video>
    </td>
    <td align="center" style="padding: 8px;">
      <strong>noise: [0.0, 0.4], high_lora_alpha 1.5</strong><br>
      <video src="https://github.com/user-attachments/assets/9f58f029-0337-4608-87dd-4a1dff7da336" width="100%" controls loop></video>
    </td>
  </tr>
</table>

**Example 4: Text-to-Video with in 4 Steps**
<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
  <tr>
    <td align="center" style="padding: 8px;">
      <video src="https://github.com/user-attachments/assets/2cd0ae7d-d20d-42b4-8877-bc39ced3767d" width="100%" controls loop></video>
    </td>
  </tr>
</table>


## News 
#### 🔥🔥🔥 2025.09: Pusa-Wan2.2 & LightX2V Acceleration Support Released!
- **🏗️ MoE DiT Support**: Added support for Wan2.2 models with separate high-noise and low-noise DiT architecture
- **⚡ LightX2V Integration**: Lightning-fast 4-step inference with maintained quality (great speed improvement)
- **🎯 New Model**: Released [Pusa-Wan2.2-V1](https://huggingface.co/RaphaelLiu/Pusa-Wan2.2-V1) with enhanced capabilities
- **📚 Updated Scripts**: All inference scripts now support `--lightx2v` acceleration flag
- **⚙️ Parameter Guidelines**: Added optimized settings for both standard and accelerated inference


#### 🔥🔥🔥 2025.07: Pusa V1.0 (Pusa-Wan) Code, Technical Report, and Dataset, all released!!! Check our [project page](https://yaofang-liu.github.io/Pusa_Web/) and [paper](https://github.com/Yaofang-Liu/Pusa-VidGen/blob/e99c3dcf866789a2db7fbe2686888ec398076a82/PusaV1/PusaV1.0_Report.pdf) for more info.

#### 🔥🔥🔥 2025.04: Pusa V0.5 (Pusa-Mochi) released.

#### 🔥🔥🔥 2024.10: FVDM [paper](https://arxiv.org/abs/2410.03160) released, initially introducing the vectorized tiemstep paradigm.


## Table of Contents
- [Overview](#overview)
- [Pusa V1.0 (Based on Wan)](#pusa-v10-based-on-wan)
- [Pusa V0.5 (Based on Mochi)](#pusa-v05-based-on-mochi)
- [Training](#training)
- [Limitations](#limitations)
- [Current Status and Roadmap](#current-status-and-roadmap)
- [Changelog](#changelog)
- [Related Work](#related-work)
- [BibTeX](#bibtex)

## Overview

Pusa (*pu: 'sA:*, from "Thousand-Hand Guanyin" in Chinese) introduces a paradigm shift in video diffusion modeling through frame-level noise control with vectorized timesteps, departing from conventional scalar timestep approaches. This shift was first presented in our [FVDM](https://arxiv.org/abs/2410.03160) paper. 

**Pusa V1.0** is based on the SOTA **Wan-T2V-14B** model and enhances it with our unique vectorized timestep adaptations (VTA), a non-destructive adaptation that fully preserves the capabilities of the base model. With the new Wan2.2 support, Pusa now leverages MoE DiT architecture for improved quality and efficiency.

The codebase has been integrated into the `PusaV1` directory, based on `DiffSynth-Studio`.

Pusa V1.0 not only sets a new standard for image-to-video generation but also unlocks many other zero-shot multi-task capabilities such as start-end frames and video extension, all without task-specific training while preserving the base model's T2V capabilities.

For detailed usage and examples for Pusa V1.0, please see the **[Pusa V1.0 README](./PusaV1/README.md)**.

**Pusa V0.5** leverages this architecture, and it is based on [Mochi1-Preview](https://huggingface.co/genmo/mochi-1-preview). We are open-sourcing this work to foster community collaboration, enhance methodologies, and expand capabilities.


### ✨ Key Features

- **Comprehensive Multi-task Support**:
  - Text-to-Video 
  - Image-to-Video 
  - Start-End Frames
  - Video completion/transitions
  - Video Extension
  - And more...

- **Unprecedented Efficiency**:
  - Surpasses Wan-I2V-14B with **≤ 1/200 of the training cost** (\$500 vs. ≥ \$100,000)
  - Trained on a dataset **≤ 1/2500 of the size** (4K vs. ≥ 10M samples)
  - Achieves a **VBench-I2V score of 87.32%** (vs. 86.86% for Wan-I2V-14B)
  - **⚡ Speed Improvement** with LightX2V acceleration (4 steps vs. 10 or more steps)

- **Advanced Architecture Options**:
  - **Wan2.1 Models**: Single DiT architecture for standard inference
  - **Wan2.2 Models**: MoE DiT architecture (high-noise + low-noise) for enhanced quality
  - **LightX2V Acceleration**: Compatible with both architectures for ultra-fast generation

- **Complete Open-Source Release**:
  - Full codebase and training/inference scripts
  - LoRA model weights and dataset for Pusa V1.0
  - [Pusa-Wan2.2-V1](https://huggingface.co/RaphaelLiu/Pusa-Wan2.2-V1) model weights
  - Detailed architecture specifications
  - Comprehensive training methodology

### 🔍 Unique Architecture

- **Novel Diffusion Paradigm**: Implements frame-level noise control with vectorized timesteps, originally introduced in the [FVDM paper](https://arxiv.org/abs/2410.03160), enabling unprecedented flexibility and scalability.

- **MoE DiT Support**: Wan2.2 models feature separate high-noise and low-noise DiT components, allowing for better noise handling and improved generation quality.

- **Lightning Acceleration**: LightX2V integration provides 4-step inference with maintained quality, dramatically reducing generation time.

- **Non-destructive Modification**: Our adaptations to the base model preserve its original Text-to-Video generation capabilities. After this adaptation, we only need a slight fine-tuning.

- **Universal Applicability**: The methodology can be readily applied to other leading video diffusion models including Hunyuan Video, Wan2.1, and others. *Collaborations enthusiastically welcomed!*

## Pusa V1.0 (Based on Wan)

Pusa V1.0 leverages the powerful Wan-Video models and enhances them with our custom LoRA models and training scripts. Now featuring both Wan2.1 and Wan2.2 architecture support with LightX2V acceleration capabilities.

**Model Options:**
- **Wan2.1 Models**: Traditional single DiT architecture ([Original Pusa V1.0](https://huggingface.co/RaphaelLiu/PusaV1))
- **Wan2.2 Models**: Advanced MoE DiT architecture ([Pusa-Wan2.2-V1](https://huggingface.co/RaphaelLiu/Pusa-Wan2.2-V1))

**Key Features:**
- **⚡ LightX2V Acceleration**: 4-step inference for both model types
- **🏗️ MoE DiT Architecture**: Enhanced quality with separate high/low noise models (Wan2.2)
- **🎯 Optimized Parameters**: `cfg_scale=1` for LightX2V, `cfg_scale=3` for standard inference
- **📈 Performance**: great speed improvement with maintained generation quality

For detailed instructions on installation, model preparation, usage examples, and training, please refer to the **[Pusa V1.0 README](./PusaV1/README.md)**.

## Pusa V0.5 (Based on Mochi)

<details>
<summary>Click to expand for Pusa V0.5 details</summary>

### Installation 

You may install using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/genmoai/models
cd models 
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install setuptools
uv pip install -e . --no-build-isolation
```

If you want to install flash attention, you can use:
```
uv pip install -e .[flash] --no-build-isolation
```

### Download Weights

**Option 1**: Use the Hugging Face CLI:
```bash
pip install huggingface_hub
huggingface-cli download RaphaelLiu/Pusa-V0.5 --local-dir <path_to_downloaded_directory>
```

**Option 2**: Download directly from [Hugging Face](https://huggingface.co/RaphaelLiu/Pusa-V0.5) to your local machine.


## Usage

### Image-to-Video Generation

```bash
python ./demos/cli_test_ti2v_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --prompt "Your_prompt_here" \
  --image_dir "/path/to/input/image.jpg" \
  --cond_position 0 \
  --num_steps 30 \
  --noise_multiplier 0
```
Note: We suggest you try different `con_position` here, and you may also modify the level of noise added to the condition image. You'd be likely to get some surprises.

Take `./demos/example.jpg` as an example and run with 4 GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./demos/cli_test_ti2v_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --prompt "The camera remains still, the man is surfing on a wave with his surfboard." \
  --image_dir "./demos/example.jpg" \
  --cond_position 0 \
  --num_steps 30 \
  --noise_multiplier 0.4
```
You can get this result:

<p align="center">
    <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/62526737953d9dc757414f2a368b94a0492ca6da/assets/example.gif" width="300" autoplay loop muted/>
    <br>
</p>

You may refer to the baselines' results from the [VideoGen-Eval](https://github.com/AILab-CVC/VideoGen-Eval) benchmark for comparison:

<p align="center">
    <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/62526737953d9dc757414f2a368b94a0492ca6da/assets/example_baseline.gif" width="1000" autoplay loop muted/>
    <br>
</p>

#### Processing A Group of Images
```bash
python ./demos/cli_test_ti2v_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --image_dir "/path/to/image/directory" \
  --prompt_dir "/path/to/prompt/directory" \
  --cond_position 1 \
  --num_steps 30
```

For group processing, each image should have a corresponding text file with the same name in the prompt directory.

#### Using the Provided Shell Script
We also provide a shell script for convenience:

```bash
# Edit cli_test_ti2v_release.sh to set your paths
# Then run:
bash ./demos/cli_test_ti2v_release.sh
```

### Multi-frame Condition

Pusa supports generating videos from multiple keyframes (2 or more) placed at specific positions in the sequence. This is useful for both start-end frame generation and multi-keyframe interpolation.

#### Start & End Frame Generation

```bash
python ./demos/cli_test_multi_frames_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --prompt "Drone view of waves crashing against the rugged cliffs along Big Sur’s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff’s edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway." \
  --multi_cond '{"0": ["./demos/example3.jpg", 0.3], "20": ["./demos/example5.jpg", 0.7]}' \
  --num_steps 30
```

The `multi_cond` parameter specifies frame condition positions and their corresponding image paths and noise multipliers. In this example, the first frame (position 0) uses `./demos/example3.jpg` with noise multiplier 0.3, and frame 20 uses `./demos/example5.jpg` with noise multiplier 0.5.

Alternatively, use the provided shell script:
```bash
# Edit parameters in cli_test_multi_frames_release.sh first
bash ./demos/cli_test_multi_frames_release.sh
```

#### Multi-keyframe Interpolation

To generate videos with more than two keyframes (e.g., start, middle, and end):

```bash
python ./demos/cli_test_multi_frames_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --prompt "Drone view of waves crashing against the rugged cliffs along Big Sur’s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff’s edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway." \
  --multi_cond '{"0": ["./demos/example3.jpg", 0.3], "13": ["./demos/example4.jpg", 0.7], "27": ["./demos/example5.jpg", 0.7]}' \
  --num_steps 30
```

### Video Transition

Create smooth transitions between two videos:

```bash
python ./demos/cli_test_transition_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --prompt "A fluffy Cockapoo, perched atop a vibrant pink flamingo jumps into a crystal-clear pool." \
  --video_start_dir "./demos/example1.mp4" \
  --video_end_dir "./demos/example2.mp4" \
  --cond_position_start "[0]" \
  --cond_position_end "[-3,-2,-1]" \
  --noise_multiplier "[0.3,0.8,0.8,0.8]" \
  --num_steps 30
```

Parameters:
- `cond_position_start`: Frame indices from the start video to use as conditioning
- `cond_position_end`: Frame indices from the end video to use as conditioning
- `noise_multiplier`: Noise level multipliers for each conditioning frame

Alternatively, use the provided shell script:
```bash
# Edit parameters in cli_test_transition_release.sh first
bash ./demos/cli_test_transition_release.sh
```

### Video Extension

Extend existing videos with generated content:

```bash
python ./demos/cli_test_extension_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --prompt "A cinematic shot captures a fluffy Cockapoo, perched atop a vibrant pink flamingo float, in a sun-drenched Los Angeles swimming pool. The crystal-clear water sparkles under the bright California sun, reflecting the playful scene." \
  --video_dir "./demos/example1.mp4" \
  --cond_position "[0,1,2,3]" \
  --noise_multiplier "[0.1,0.2,0.3,0.4]" \
  --num_steps 30
```

Parameters:
- `cond_position`: Frame indices from the input video to use as conditioning
- `noise_multiplier`: Noise level multipliers for each conditioning frame

Alternatively, use the provided shell script:
```bash
# Edit parameters in cli_test_v2v_release.sh first
bash ./demos/cli_test_v2v_release.sh
```

### Text-to-Video Generation
```bash
python ./demos/cli_test_ti2v_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --prompt "A man is playing basketball" \
  --num_steps 30
```

</details>

## Training

For Pusa V1.0, please find the training details in the **[Pusa V1.0 README](./PusaV1/README.md#training)**.

For Pusa V0.5, you can find our training code and details [here](https://github.com/Yaofang-Liu/Mochi-Full-Finetuner), which also supports training for the original Mochi model.

## Limitations

Pusa currently has several known limitations:
- Video generation quality is dependent on the base model (e.g., Wan-T2V-14B for V1.0).
- We anticipate significant quality improvements when applying our methodology to more advanced models.
- Sim2real gap between conditioning frames and generated frames, like color shifting in image-to-video task.
- We welcome community contributions to enhance model performance and extend its capabilities.


### Currently Available
- ✅ Model weights for Pusa V1.0 and V0.5
- ✅ Model weights for Pusa-Wan2.2-V1 with MoE DiT architecture
- ✅ LightX2V acceleration support for both Wan2.1 and Wan2.2
- ✅ Inference code for Text-to-Video generation
- ✅ Inference code for Image-to-Video generation
- ✅ Inference scripts for start & end frames, multi-frames, video transition, video extension
- ✅ Training code and details
- ✅ Model full fine-tuning guide (for Pusa V0.5)
- ✅ Training datasets
- ✅ Technical Report for Pusa V1.0
  
### TODO List
- 🔄 Release more advanced versions with SOTA models
- 🔄 More capabilities like long video generation
- 🔄 Enhanced LightX2V integration for even faster inference
- 🔄 ....

## Changelog

**v1.0.1 (September 1, 2025)**
- Added Wan2.2 MoE DiT architecture support
- Integrated LightX2V acceleration for 4-step inference
- Released [Pusa-Wan2.2-V1](https://huggingface.co/RaphaelLiu/Pusa-Wan2.2-V1) model weights
- Added `wan22_*` inference scripts with MoE LoRA support
- Updated all `wan_*` scripts with `--lightx2v` acceleration flag
- Added comprehensive parameter guidelines for accelerated inference
- Enhanced documentation with Wan2.2 usage examples

**v1.0 (July 15, 2025)**
- Released Pusa V1.0, based on the Wan-Video models.
- Released Technical Report, V1.0 model weights and dataset.
- Integrated codebase as `/PusaV1`.
- Added new examples and training scripts for Pusa V1.0 in `PusaV1/`.
- Updated documentation for the V1.0 release.

**v0.5 (June 3, 2025)**
- Released inference scripts for Start&End Frames Generation, Multi-Frames Generation, Video Transition, and Video Extension.

**v0.5 (April 10, 2025)**
- Released our training codes and details [here](https://github.com/Yaofang-Liu/Mochi-Full-Finetuner)
- Support multi-nodes/single-node full finetuning code for both Pusa and Mochi
- Released our training dataset [dataset](https://huggingface.co/datasets/RaphaelLiu/PusaV0.5_Training)


## Related Work

- [FVDM](https://arxiv.org/abs/2410.03160): Introduces the groundbreaking frame-level noise control with vectorized timestep approach that inspired Pusa.
- [Wan-Video](https://github.com/modelscope/DiffSynth-Studio): The foundation model for Pusa V1.0.
- [LightX2V](https://github.com/ModelTC/LightX2V): Lightning-fast inference acceleration integrated into Pusa for 4-step generation.
- [Mochi](https://huggingface.co/genmo/mochi-1-preview): The foundation model for Pusa V0.5, recognized as a leading open-source video generation system on the Artificial Analysis Leaderboard.

## BibTeX
If you use this work in your project, please cite the following references.

```
@article{liu2025pusa,
  title={PUSA V1. 0: Surpassing Wan-I2V with $500 Training Cost by Vectorized Timestep Adaptation},
  author={Liu, Yaofang and Ren, Yumeng and Artola, Aitor and Hu, Yuxuan and Cun, Xiaodong and Zhao, Xiaotong and Zhao, Alan and Chan, Raymond H and Zhang, Suiyun and Liu, Rui and others},
  journal={arXiv preprint arXiv:2507.16116},
  year={2025}
}
```

```
@misc{Liu2025pusa,
  title={Pusa: Thousands Timesteps Video Diffusion Model},
  author={Yaofang Liu and Rui Liu},
  year={2025},
  url={https://github.com/Yaofang-Liu/Pusa-VidGen},
}
```

```
@article{liu2024redefining,
  title={Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach},
  author={Liu, Yaofang and Ren, Yumeng and Cun, Xiaodong and Artola, Aitor and Liu, Yang and Zeng, Tieyong and Chan, Raymond H and Morel, Jean-michel},
  journal={arXiv preprint arXiv:2410.03160},
  year={2024}
}
```












